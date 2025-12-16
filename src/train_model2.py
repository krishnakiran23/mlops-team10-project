"""
Model 2: Random Forest Training Script

This script trains a Random Forest model using scikit-learn and logs everything to MLflow.
Based on H2O AutoML second-best performer.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import sys

# Import MLflow config
sys.path.append(str(Path(__file__).parent))
from mlflow_config import setup_mlflow

def load_data():
    """Load train, validation, and test datasets"""
    print("Loading datasets...")
    
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/validate.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    print(f"Train: {len(train_df)} rows")
    print(f"Validation: {len(val_df)} rows")
    print(f"Test: {len(test_df)} rows")
    
    return train_df, val_df, test_df

def prepare_features(train_df, val_df, test_df):
    """Prepare features and target"""
    print("\nPreparing features...")
    
    # Define features to exclude
    exclude_cols = ['No', 'pm2.5', 'datetime', 'cbwd', 'hour_category']
    
    # Get feature columns
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Separate features and target
    X_train = train_df[feature_cols]
    y_train = train_df['pm2.5']
    
    X_val = val_df[feature_cols]
    y_val = val_df['pm2.5']
    
    X_test = test_df[feature_cols]
    y_test = test_df['pm2.5']
    
    print(f"Number of features: {len(feature_cols)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

def train_model(X_train, y_train):
    """Train Random Forest model"""
    print("\nTraining Random Forest model...")
    
    # Model parameters
    params = {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 0
    }
    
    # Create and train model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    
    print("✓ Model training completed")
    
    return model, params

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evaluate model on all datasets"""
    print("\nEvaluating model...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'val_mae': mean_absolute_error(y_val, y_val_pred),
        'val_r2': r2_score(y_val, y_val_pred),
        
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_r2': r2_score(y_test, y_test_pred),
    }
    
    print("\nMetrics:")
    print(f"Train - RMSE: {metrics['train_rmse']:.4f}, MAE: {metrics['train_mae']:.4f}, R²: {metrics['train_r2']:.4f}")
    print(f"Val   - RMSE: {metrics['val_rmse']:.4f}, MAE: {metrics['val_mae']:.4f}, R²: {metrics['val_r2']:.4f}")
    print(f"Test  - RMSE: {metrics['test_rmse']:.4f}, MAE: {metrics['test_mae']:.4f}, R²: {metrics['test_r2']:.4f}")
    
    return metrics, y_test_pred

def create_plots(model, feature_cols, y_test, y_test_pred):
    """Create visualization plots"""
    print("\nCreating plots...")
    
    # Create output directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Feature Importance Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    sns.barplot(data=importance_df, y='feature', x='importance', ax=ax)
    ax.set_title('Top 15 Feature Importances - Random Forest Model')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    feature_importance_path = plots_dir / "model2_rf_feature_importance.png"
    plt.savefig(feature_importance_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Residual Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    residuals = y_test - y_test_pred
    ax.scatter(y_test_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted PM2.5')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot - Random Forest Model')
    plt.tight_layout()
    residual_path = plots_dir / "model2_rf_residuals.png"
    plt.savefig(residual_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Actual vs Predicted Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_test_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual PM2.5')
    ax.set_ylabel('Predicted PM2.5')
    ax.set_title('Actual vs Predicted - Random Forest Model')
    plt.tight_layout()
    pred_path = plots_dir / "model2_rf_predictions.png"
    plt.savefig(pred_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plots saved to {plots_dir}/")
    
    return feature_importance_path, residual_path, pred_path

def log_to_mlflow(model, params, metrics, feature_importance_path, residual_path, pred_path):
    """Log model, parameters, metrics, and artifacts to MLflow"""
    print("\nLogging to MLflow...")
    
    mlflow = setup_mlflow()
    
    with mlflow.start_run(run_name="RandomForest_Model"):
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="PM25_RandomForest"
        )
        
        # Log plots
        mlflow.log_artifact(str(feature_importance_path))
        mlflow.log_artifact(str(residual_path))
        mlflow.log_artifact(str(pred_path))
        
        # Log model info
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("algorithm", "sklearn.RandomForestRegressor")
        mlflow.set_tag("dataset", "PRSA Air Quality")
        
        run_id = mlflow.active_run().info.run_id
        print(f"✓ Logged to MLflow (Run ID: {run_id})")
    
    return run_id

def main():
    """Main training pipeline"""
    print("="*70)
    print("MODEL 2: RANDOM FOREST")
    print("="*70)
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    # Prepare features
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_features(
        train_df, val_df, test_df
    )
    
    # Train model
    model, params = train_model(X_train, y_train)
    
    # Evaluate model
    metrics, y_test_pred = evaluate_model(
        model, X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Create plots
    feature_importance_path, residual_path, pred_path = create_plots(
        model, feature_cols, y_test, y_test_pred
    )
    
    # Log to MLflow
    run_id = log_to_mlflow(
        model, params, metrics, feature_importance_path, residual_path, pred_path
    )
    
    print("\n" + "="*70)
    print("RANDOM FOREST MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"MLflow Run ID: {run_id}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Test MAE: {metrics['test_mae']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")

if __name__ == "__main__":
    main()
