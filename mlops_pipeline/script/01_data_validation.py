import os
import mlflow


def validate_data():
    """
    Validates the mushroom image dataset structure and logs results to MLflow.
    """
    mlflow.set_experiment("Mushroom Classification - Data Validation")

    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        # 1. Check dataset structure
        raw_dataset_path = os.path.join(os.getcwd(), "raw_dataset")
        
        if not os.path.exists(raw_dataset_path):
            print("Dataset path not found!")
            mlflow.log_param("validation_status", "Failed")
            return

        # 2. Count classes and images
        num_classes = 0
        total_images = 0
        
        for folder_name in os.listdir(raw_dataset_path):
            folder_path = os.path.join(raw_dataset_path, folder_name)
            if os.path.isdir(folder_path):
                num_classes += 1
                image_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                total_images += image_count
                print(f"{folder_name}: {image_count} images")

        print(f"Total classes: {num_classes}")
        print(f"Total images: {total_images}")

        # 3. Log validation results
        mlflow.log_metric("num_classes", num_classes)
        mlflow.log_metric("total_images", total_images)
        mlflow.log_metric("avg_images_per_class", total_images / num_classes if num_classes > 0 else 0)

        # 4. Validation check
        validation_status = "Success" if num_classes > 0 and total_images > 0 else "Failed"
        mlflow.log_param("validation_status", validation_status)
        print(f"Validation status: {validation_status}")
        print("Data validation run finished.")


if __name__ == "__main__":
    validate_data()
