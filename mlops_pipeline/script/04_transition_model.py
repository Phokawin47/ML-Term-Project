import mlflow
from mlflow.tracking import MlflowClient


def transition_best_model():
    """
    Find the best performing mushroom classifier model and set appropriate alias
    """
    client = MlflowClient()
    model_name = "mushroom-classifier"
    
    try:
        # Get all versions of the mushroom classifier model
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"No model versions found for {model_name}")
            return
        
        # Find best model based on test_accuracy
        best_version = None
        best_accuracy = 0
        
        for version in versions:
            run_id = version.run_id
            run = client.get_run(run_id)
            test_accuracy = run.data.metrics.get('test_accuracy', 0)
            
            print(f"Version {version.version}: accuracy {test_accuracy:.4f}")
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_version = version
        
        if best_version is None:
            print("No valid model version found")
            return
        
        print(f"Best model: Version {best_version.version} with accuracy {best_accuracy:.4f}")
        
        # Set alias based on performance
        if best_accuracy >= 0.001:
            alias = "production"
        elif best_accuracy >= 0.0001:
            alias = "staging"
        else:
            alias = "candidate"
            print(f"Model accuracy {best_accuracy:.4f} below production threshold")
        
        # Set the alias
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=best_version.version
        )
        
        print(f"Model version {best_version.version} set to alias '{alias}'")
        
        # Update description
        client.update_model_version(
            name=model_name,
            version=best_version.version,
            description=f"ResNet model with {best_accuracy:.4f} test accuracy - {alias} ready"
        )
        
    except Exception as e:
        print(f"Error transitioning model: {e}")


if __name__ == "__main__":
    transition_best_model()
