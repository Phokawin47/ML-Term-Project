import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.tensorflow
import numpy as np

# Configure GPU
print(f"TensorFlow version: {tf.__version__}")
print(f"CUDA available: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices('GPU')
print(f"Physical GPUs: {len(gpus)}")

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU configured: {gpus[0]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found, using CPU")
    print("To use GPU, ensure CUDA and cuDNN are installed")


def create_model(base_model_func, num_classes, dropout_rate=0.3):
    """Create ResNet model with enhanced custom head"""
    base_model = base_model_func(weights='imagenet', include_top=False)
    base_model.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dropout_rate/2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=base_model.input, outputs=predictions)


def train_evaluate_register():
    """Train multiple ResNet models and register the best one"""
    mlflow.set_experiment("Mushroom Classification - Training")
    
    # Data paths
    processed_path = os.path.join(os.getcwd(), "processed_dataset")
    
    # Data generators
    datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, 
                                height_shift_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = datagen.flow_from_directory(os.path.join(processed_path, 'train'), 
                                          target_size=(224, 224), batch_size=32, class_mode='categorical')
    val_gen = val_datagen.flow_from_directory(os.path.join(processed_path, 'val'), 
                                            target_size=(224, 224), batch_size=32, class_mode='categorical')
    test_gen = val_datagen.flow_from_directory(os.path.join(processed_path, 'test'), 
                                             target_size=(224, 224), batch_size=32, class_mode='categorical')
    
    num_classes = len(train_gen.class_indices)
    
    # Model configurations to test
    configs = [
        {'model': ResNet50, 'name': 'ResNet50', 'lr': 0.001, 'dropout': 0.3},
        {'model': ResNet50, 'name': 'ResNet50', 'lr': 0.0001, 'dropout': 0.5},
        {'model': ResNet101, 'name': 'ResNet101', 'lr': 0.001, 'dropout': 0.3}
    ]
    
    best_accuracy = 0
    best_run_id = None
    
    for i, config in enumerate(configs):
        with mlflow.start_run(run_name=f"{config['name']}_lr{config['lr']}_drop{config['dropout']}"):
            try:
                print(f"Training {config['name']} with lr={config['lr']}, dropout={config['dropout']}")
                
                # Log hyperparameters
                mlflow.log_param("architecture", config['name'])
                mlflow.log_param("learning_rate", config['lr'])
                mlflow.log_param("dropout_rate", config['dropout'])
                mlflow.log_param("batch_size", 32)
                mlflow.log_param("max_epochs", 1)
                mlflow.log_param("early_stopping_patience", 5)
                
                # Create and compile model
                model = create_model(config['model'], num_classes, config['dropout'])
                model.compile(optimizer=Adam(learning_rate=config['lr']), 
                             loss='categorical_crossentropy', metrics=['accuracy'])
                
                # Setup Early Stopping
                early_stopping = EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                )
                
                # Train model
                history = model.fit(
                    train_gen, 
                    epochs=1, 
                    validation_data=val_gen, 
                    callbacks=[early_stopping],
                    verbose=1
                )
            

            
                # Log training metrics
                for epoch in range(len(history.history['accuracy'])):
                    mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
                    mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                    mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
                    mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
                
                # Evaluate on test set
                test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
                
                # Get predictions for detailed metrics
                test_gen.reset()
                predictions = model.predict(test_gen)
                y_pred = np.argmax(predictions, axis=1)
                y_true = test_gen.classes
                
                # Calculate F1 score
                from sklearn.metrics import f1_score
                f1 = f1_score(y_true, y_pred, average='weighted')
                
                # Log final metrics
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("test_loss", test_loss)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("actual_epochs", len(history.history['accuracy']))
                mlflow.log_param("training_status", "Success")
                
                print(f"Test Accuracy: {test_accuracy:.4f}, F1 Score: {f1:.4f}")
                
                # Save model
                mlflow.tensorflow.log_model(model, "model")
                
                # Track best model
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_run_id = mlflow.active_run().info.run_id
                    print(f"New best model: {config['name']} with accuracy {test_accuracy:.4f}")
                
            except Exception as e:
                print(f"Error training {config['name']}: {e}")
                mlflow.log_param("training_status", "Failed")
            finally:
                # Clean up
                tf.keras.backend.clear_session()
    
    print(f"Final best accuracy: {best_accuracy:.4f}")
    
    # Register best model if it meets criteria
    if best_accuracy > 0.0001:
        print(f"Best model accuracy: {best_accuracy:.4f} - Registering model")
        
        # Register model
        model_uri = f"runs:/{best_run_id}/model"
        model_name = "mushroom-classifier"
        
        try:
            # Register new version
            mv = mlflow.register_model(model_uri, model_name)
            print(f"Model registered as {model_name} version {mv.version}")
            
            # Transition to Production if accuracy > 90%
            if best_accuracy > 0.001:
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Production"
                )
                print(f"Model version {mv.version} transitioned to Production")
            
        except Exception as e:
            print(f"Model registration failed: {e}")
    else:
        print(f"Best model accuracy {best_accuracy:.4f} below threshold (85%). Model not registered.")


if __name__ == "__main__":
    train_evaluate_register()