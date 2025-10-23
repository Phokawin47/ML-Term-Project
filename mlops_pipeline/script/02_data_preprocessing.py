import os
import shutil
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn


def preprocess_data():
    """
    Split mushroom dataset into train/test/val and prepare preprocessing artifacts.
    """
    mlflow.set_experiment("Mushroom Classification - Data Preprocessing")
    
    with mlflow.start_run():
        print("Starting data preprocessing run...")
        mlflow.set_tag("ml.step", "data_preprocessing")
        
        raw_path = os.path.join(os.getcwd(), "raw_dataset")
        processed_path = os.path.join(os.getcwd(), "processed_dataset")
        
        # Check if already processed
        if os.path.exists(processed_path):
            print("Processed dataset already exists. Skipping preprocessing.")
            mlflow.log_param("preprocessing_status", "Skipped")
            return
        
        # Create processed dataset structure
        for split in ['train', 'test', 'val']:
            os.makedirs(os.path.join(processed_path, split), exist_ok=True)
        
        # Get all classes and create label encoder
        classes = [d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))]
        label_encoder = LabelEncoder()
        label_encoder.fit(classes)
        
        total_images = 0
        
        # Process each class
        for class_name in classes:
            class_path = os.path.join(raw_path, class_name)
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if len(images) < 3:
                print(f"Skipping {class_name}: insufficient images ({len(images)})")
                continue
            
            # Split images: 70% train, 20% test, 10% val
            train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
            test_imgs, val_imgs = train_test_split(temp_imgs, test_size=0.33, random_state=42)
            
            # Create class folders in each split
            for split in ['train', 'test', 'val']:
                os.makedirs(os.path.join(processed_path, split, class_name), exist_ok=True)
            
            # Copy images to respective splits
            for img in train_imgs:
                shutil.copy2(os.path.join(class_path, img), 
                           os.path.join(processed_path, 'train', class_name, img))
            
            for img in test_imgs:
                shutil.copy2(os.path.join(class_path, img), 
                           os.path.join(processed_path, 'test', class_name, img))
            
            for img in val_imgs:
                shutil.copy2(os.path.join(class_path, img), 
                           os.path.join(processed_path, 'val', class_name, img))
            
            total_images += len(images)
            print(f"{class_name}: {len(train_imgs)} train, {len(test_imgs)} test, {len(val_imgs)} val")
        
        # Save label encoder as artifact
        encoder_path = "label_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        # Log metrics and artifacts
        mlflow.log_metric("total_classes", len(classes))
        mlflow.log_metric("total_images", total_images)
        mlflow.log_param("train_ratio", 0.7)
        mlflow.log_param("test_ratio", 0.2)
        mlflow.log_param("val_ratio", 0.1)
        mlflow.log_artifact(encoder_path, "preprocessing")
        mlflow.log_param("preprocessing_status", "Success")
        
        # Clean up local file
        os.remove(encoder_path)
        
        print("Data preprocessing completed successfully.")


if __name__ == "__main__":
    preprocess_data()