import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient
import pickle
from datetime import datetime, timedelta
import json


def calculate_psi(expected, actual, buckets=10):
    """Calculate Population Stability Index"""
    def scale_range(input_array, new_min=0, new_max=1):
        return (input_array - input_array.min()) / (input_array.max() - input_array.min()) * (new_max - new_min) + new_min
    
    expected_scaled = scale_range(expected)
    actual_scaled = scale_range(actual)
    
    expected_counts, bin_edges = np.histogram(expected_scaled, bins=buckets)
    actual_counts, _ = np.histogram(actual_scaled, bins=bin_edges)
    
    expected_percents = expected_counts / len(expected_scaled)
    actual_percents = actual_counts / len(actual_scaled)
    
    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi


def data_drift_detection():
    """Detect data drift using PSI and KS test"""
    mlflow.set_experiment("Mushroom Classification - Monitoring")
    
    with mlflow.start_run(run_name="data_drift_detection"):
        print("Starting data drift detection...")
        
        # Load reference data (training data features)
        reference_data_path = os.path.join(os.getcwd(), "processed_dataset", "train")
        production_data_path = os.path.join(os.getcwd(), "production_data")  # Simulated production data
        
        if not os.path.exists(production_data_path):
            print("No production data found. Creating sample data...")
            os.makedirs(production_data_path, exist_ok=True)
            # Simulate production data with some drift
            np.random.seed(42)
            production_features = np.random.normal(0.5, 0.2, 1000)  # Simulated features
        else:
            production_features = np.random.normal(0.6, 0.3, 1000)  # Drifted data
        
        # Reference features (simulated from training)
        reference_features = np.random.normal(0.5, 0.15, 5000)
        
        # Calculate PSI
        psi_score = calculate_psi(reference_features, production_features)
        
        # Calculate KS test
        ks_statistic, ks_p_value = stats.ks_2samp(reference_features, production_features)
        
        # Log metrics
        mlflow.log_metric("psi_score", psi_score)
        mlflow.log_metric("ks_statistic", ks_statistic)
        mlflow.log_metric("ks_p_value", ks_p_value)
        
        # Determine drift status
        data_drift_detected = psi_score > 0.2 or ks_statistic > 0.1
        mlflow.log_param("data_drift_detected", data_drift_detected)
        
        print(f"PSI Score: {psi_score:.4f}")
        print(f"KS Statistic: {ks_statistic:.4f}")
        print(f"Data Drift Detected: {data_drift_detected}")
        
        return data_drift_detected, psi_score


def concept_drift_detection():
    """Detect concept drift using performance degradation"""
    mlflow.set_experiment("Mushroom Classification - Monitoring")
    
    with mlflow.start_run(run_name="concept_drift_detection"):
        print("Starting concept drift detection...")
        
        # Simulate current performance vs baseline
        baseline_accuracy = 0.92  # From training
        
        # Simulate rolling window accuracy (last 7 days)
        current_accuracies = np.random.normal(0.85, 0.05, 7)  # Degraded performance
        current_avg_accuracy = np.mean(current_accuracies)
        
        # Calculate performance drop
        performance_drop = (baseline_accuracy - current_avg_accuracy) / baseline_accuracy
        
        # Log metrics
        mlflow.log_metric("baseline_accuracy", baseline_accuracy)
        mlflow.log_metric("current_avg_accuracy", current_avg_accuracy)
        mlflow.log_metric("performance_drop_percent", performance_drop * 100)
        
        # Detect concept drift
        concept_drift_detected = performance_drop > 0.1  # 10% threshold
        mlflow.log_param("concept_drift_detected", concept_drift_detected)
        
        print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
        print(f"Current Average Accuracy: {current_avg_accuracy:.4f}")
        print(f"Performance Drop: {performance_drop*100:.2f}%")
        print(f"Concept Drift Detected: {concept_drift_detected}")
        
        return concept_drift_detected, performance_drop


def check_new_data_volume():
    """Check if new data volume exceeds threshold"""
    # Simulate data volume check
    training_data_size = 19165  # From original training
    new_data_size = 2500  # Simulated new data
    
    new_data_ratio = new_data_size / training_data_size
    volume_threshold_exceeded = new_data_ratio > 0.1  # 10% threshold
    
    print(f"New data ratio: {new_data_ratio:.2f}")
    print(f"Volume threshold exceeded: {volume_threshold_exceeded}")
    
    return volume_threshold_exceeded, new_data_ratio


def automated_retraining_trigger():
    """Main monitoring function that triggers retraining if needed"""
    mlflow.set_experiment("Mushroom Classification - Monitoring")
    
    with mlflow.start_run(run_name="monitoring_pipeline"):
        print("=== Mushroom Classifier Monitoring Pipeline ===")
        
        # Run all monitoring checks
        data_drift, psi_score = data_drift_detection()
        concept_drift, perf_drop = concept_drift_detection()
        volume_exceeded, data_ratio = check_new_data_volume()
        
        # Log overall monitoring results
        mlflow.log_param("monitoring_timestamp", datetime.now().isoformat())
        mlflow.log_metric("psi_score", psi_score)
        mlflow.log_metric("performance_drop", perf_drop)
        mlflow.log_metric("new_data_ratio", data_ratio)
        
        # Determine if retraining is needed
        retrain_triggers = []
        
        if data_drift:
            retrain_triggers.append("data_drift")
            print("🚨 Data Drift detected - PSI > 0.2")
        
        if concept_drift:
            retrain_triggers.append("concept_drift")
            print("🚨 Concept Drift detected - Performance drop > 10%")
        
        if volume_exceeded:
            retrain_triggers.append("new_data_volume")
            print("🚨 New data volume > 10% of training set")
        
        # Log triggers
        mlflow.log_param("retrain_triggers", json.dumps(retrain_triggers))
        
        if retrain_triggers:
            print(f"\n🔄 RETRAINING TRIGGERED by: {', '.join(retrain_triggers)}")
            mlflow.log_param("retraining_needed", True)
            
            # Simulate triggering retraining pipeline
            print("Triggering automated retraining pipeline...")
            print("1. Running 02_data_preprocessing.py")
            print("2. Running 03_train_evaluate_register.py")
            print("3. Running 04_transition_model.py")
            
            # In real implementation, this would call:
            # subprocess.run(["python", "02_data_preprocessing.py"])
            # subprocess.run(["python", "03_train_evaluate_register.py"])
            # subprocess.run(["python", "04_transition_model.py"])
            
        else:
            print("\n✅ No drift detected - Model performance stable")
            mlflow.log_param("retraining_needed", False)
        
        return len(retrain_triggers) > 0


if __name__ == "__main__":
    automated_retraining_trigger()