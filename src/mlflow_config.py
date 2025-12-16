"""
MLflow Configuration

Set MLflow tracking URI here. 
For local development, use local file storage.
For production, use remote tracking server.
"""

import os

# MLflow Configuration
# For local development (bef# Local tracking (default)
# MLFLOW_TRACKING_URI = "file:./mlruns"

# Remote tracking (AWS EC2 setup)
MLFLOW_TRACKING_URI = "http://13.217.233.217:5000"
# MLFLOW_TRACKING_URI = "http://<EC2_PUBLIC_IP>:5000"

# MLflow Experiment Name
EXPERIMENT_NAME = "prsa_pm25_prediction"

# AWS S3 Configuration (for remote artifact storage)
# Uncomment after AWS setup
# AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# S3_BUCKET = "mlops-team10-artifacts"

def setup_mlflow():
    """Setup MLflow tracking"""
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")
    return mlflow
