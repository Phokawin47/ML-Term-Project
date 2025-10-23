# Mushroom Classification MLOps Pipeline

Automated machine learning pipeline for mushroom classification using ResNet models with comprehensive monitoring and drift detection.

## 🚀 Features

- **Automated Data Validation** - Dataset structure and quality checks
- **Data Preprocessing** - Train/test/validation split with artifact logging
- **Model Training** - Multiple ResNet architectures with hyperparameter tuning
- **Model Registry** - Automatic model versioning and staging
- **Drift Monitoring** - Data and concept drift detection
- **Automated Retraining** - Trigger retraining based on performance degradation
- **GitHub Actions CI/CD** - Fully automated pipeline execution

## 📁 Project Structure

```
ML/
├── .github/workflows/
│   └── main.yml                    # GitHub Actions workflow
├── mlops_pipeline/script/
│   ├── 01_data_validation.py       # Dataset validation
│   ├── 02_data_preprocessing.py    # Data splitting & preprocessing
│   ├── 03_train_evaluate_register.py # Model training & registration
│   ├── 04_transition_model.py      # Model staging & promotion
│   └── 05_monitoring_strategy.py   # Drift detection & monitoring
├── raw_dataset/                    # Raw mushroom images
├── processed_dataset/              # Processed train/test/val splits
└── README.md
```

## 🛠️ Setup

### 1. Repository Setup
```bash
git clone <your-repo-url>
cd ML
```

### 2. GitHub Secrets Configuration
Add these secrets in your GitHub repository settings:

```
MLFLOW_TRACKING_URI=<your-mlflow-server-url>
MLFLOW_TRACKING_USERNAME=<username>
MLFLOW_TRACKING_PASSWORD=<password>
```

### 3. Local Development
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1

# Install dependencies
pip install tensorflow scikit-learn mlflow pandas numpy scipy
```

## 🔄 Pipeline Execution

### Automatic Triggers
- **Push to main/develop** - Runs full training pipeline
- **Pull Request** - Runs validation and training
- **Daily Schedule (2 AM)** - Runs monitoring pipeline
- **Manual Trigger** - Available in GitHub Actions tab

### Manual Execution
```bash
# Run individual scripts
python mlops_pipeline/script/01_data_validation.py
python mlops_pipeline/script/02_data_preprocessing.py
python mlops_pipeline/script/03_train_evaluate_register.py
python mlops_pipeline/script/04_transition_model.py
python mlops_pipeline/script/05_monitoring_strategy.py
```

## 📊 Pipeline Stages

### 1. Data Validation
- Validates dataset structure
- Counts classes and images
- Logs metrics to MLflow

### 2. Data Preprocessing  
- Splits data (70% train, 20% test, 10% val)
- Creates processed dataset folders
- Saves label encoder artifact

### 3. Model Training
- Tests multiple ResNet architectures (ResNet50, ResNet101)
- Hyperparameter tuning (learning rate, dropout)
- Evaluates on test set
- Registers best model (accuracy > 85%)

### 4. Model Transition
- Finds best performing model version
- Sets appropriate alias (production/staging/candidate)
- Updates model descriptions

### 5. Monitoring
- **Data Drift**: PSI > 0.2 or KS test
- **Concept Drift**: Performance drop > 10%
- **Volume Check**: New data > 10% of training set
- **Auto Retraining**: Triggers pipeline if drift detected

## 🎯 Model Performance Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| Test Accuracy | > 90% | Production alias |
| Test Accuracy | > 85% | Staging alias |
| Test Accuracy | < 85% | Candidate alias |
| PSI Score | > 0.2 | Trigger retraining |
| Performance Drop | > 10% | Trigger retraining |
| New Data Volume | > 10% | Trigger retraining |

## 📈 MLflow Tracking

Access your MLflow UI to monitor:
- Experiment runs and metrics
- Model versions and stages
- Drift detection results
- Training artifacts and parameters

## 🔧 Configuration

### Model Configurations
```python
configs = [
    {'model': ResNet50, 'lr': 0.001, 'dropout': 0.3},
    {'model': ResNet50, 'lr': 0.0001, 'dropout': 0.5},
    {'model': ResNet101, 'lr': 0.001, 'dropout': 0.3}
]
```

### Monitoring Thresholds
```python
PSI_THRESHOLD = 0.2
PERFORMANCE_DROP_THRESHOLD = 0.1
NEW_DATA_RATIO_THRESHOLD = 0.1
```

## 🚨 Troubleshooting

### Common Issues
1. **GPU not detected**: Install CUDA drivers or run on CPU
2. **MLflow connection**: Check tracking URI and credentials
3. **Data not found**: Ensure raw_dataset folder exists
4. **Memory issues**: Reduce batch size or model complexity

### GitHub Actions Debugging
- Check Actions tab for pipeline logs
- Verify secrets are properly configured
- Ensure repository has necessary permissions

## 📝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🤝 Support

For issues and questions:
- Create GitHub Issues
- Check MLflow logs for debugging
- Review GitHub Actions workflow logs