"""
Model Drift Analysis with Evidently v0.2.8

This script uses Evidently library to detect data drift and performance drift.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# Import Evidently v0.2.8
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, RegressionPerformanceTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, RegressionPerformanceProfileSection
from evidently.pipeline.column_mapping import ColumnMapping

def main():
    """Main drift analysis pipeline using Evidently"""
    print("="*70)
    print("MODEL DRIFT ANALYSIS WITH EVIDENTLY v0.2.8")
    print("="*70)
    
    # Load data
    print("\nLoading datasets...")
    reference_df = pd.read_csv("data/processed/validate.csv")
    current_df = pd.read_csv("data/processed/test.csv")
    
    print(f"Reference data: {len(reference_df)} rows ({reference_df['datetime'].min()} to {reference_df['datetime'].max()})")
    print(f"Current data: {len(current_df)} rows ({current_df['datetime'].min()} to {current_df['datetime'].max()})")
    
    # Create output directory
    output_dir = Path("drift_reports")
    output_dir.mkdir(exist_ok=True)
    
    # ===== LOAD MODEL AND MAKE PREDICTIONS =====
    print("\n" + "="*70)
    print("LOADING MODEL AND MAKING PREDICTIONS")
    print("="*70)
    
    mlflow.set_tracking_uri("http://13.217.233.217:5000")
    print("Loading champion model from remote MLflow (Production stage)...")
    model = mlflow.sklearn.load_model("models:/PM25_GBM/Production")
    print("✓ Model loaded")
    
    # Prepare features
    exclude_cols = ['No', 'pm2.5', 'datetime', 'cbwd', 'hour_category']
    feature_cols = [col for col in reference_df.columns if col not in exclude_cols]
    
    # Make predictions
    print("\nMaking predictions...")
    X_ref = reference_df[feature_cols]
    y_ref = reference_df['pm2.5']
    y_ref_pred = model.predict(X_ref)
    
    X_curr = current_df[feature_cols]
    y_curr = current_df['pm2.5']
    y_curr_pred = model.predict(X_curr)
    
    # Prepare data for Evidently (needs target and prediction columns)
    ref_data = reference_df[feature_cols].copy()
    ref_data['target'] = y_ref
    ref_data['prediction'] = y_ref_pred
    
    curr_data = current_df[feature_cols].copy()
    curr_data['target'] = y_curr
    curr_data['prediction'] = y_curr_pred
    
    # ===== DATA DRIFT DASHBOARD =====
    print("\n" + "="*70)
    print("GENERATING DATA DRIFT DASHBOARD")
    print("="*70)
    
    data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
    data_drift_dashboard.calculate(ref_data, curr_data)
    
    drift_dashboard_path = output_dir / "evidently_data_drift_dashboard.html"
    data_drift_dashboard.save(str(drift_dashboard_path))
    print(f"✓ Data drift dashboard saved to {drift_dashboard_path}")
    
    # ===== REGRESSION PERFORMANCE DASHBOARD =====
    print("\n" + "="*70)
    print("GENERATING REGRESSION PERFORMANCE DASHBOARD")
    print("="*70)
    
    regression_dashboard = Dashboard(tabs=[RegressionPerformanceTab()])
    
    column_mapping = ColumnMapping()
    column_mapping.target = 'target'
    column_mapping.prediction = 'prediction'
    
    regression_dashboard.calculate(ref_data, curr_data, column_mapping=column_mapping)
    
    regression_dashboard_path = output_dir / "evidently_regression_performance_dashboard.html"
    regression_dashboard.save(str(regression_dashboard_path))
    print(f"✓ Regression performance dashboard saved to {regression_dashboard_path}")
    
    # ===== DATA DRIFT PROFILE (JSON) - SKIPPED DUE TO PYDANTIC COMPATIBILITY =====
    # Note: JSON profiles have compatibility issues with pydantic v1/v2
    # HTML dashboards above provide all necessary visualizations
    
    # ===== REGRESSION PERFORMANCE PROFILE (JSON) - SKIPPED =====
    # Note: JSON profiles have compatibility issues with pydantic v1/v2
    # HTML dashboards above provide all necessary visualizations
    
    # ===== CALCULATE METRICS MANUALLY =====
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    ref_metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_ref, y_ref_pred))),
        'mae': float(mean_absolute_error(y_ref, y_ref_pred)),
        'r2': float(r2_score(y_ref, y_ref_pred))
    }
    
    curr_metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_curr, y_curr_pred))),
        'mae': float(mean_absolute_error(y_curr, y_curr_pred)),
        'r2': float(r2_score(y_curr, y_curr_pred))
    }
    
    print("\nReference Period (Validation Set):")
    print(f"  RMSE: {ref_metrics['rmse']:.4f}")
    print(f"  MAE: {ref_metrics['mae']:.4f}")
    print(f"  R²: {ref_metrics['r2']:.4f}")
    
    print("\nCurrent Period (Test Set):")
    print(f"  RMSE: {curr_metrics['rmse']:.4f}")
    print(f"  MAE: {curr_metrics['mae']:.4f}")
    print(f"  R²: {curr_metrics['r2']:.4f}")
    
    # Calculate changes
    rmse_change = ((curr_metrics['rmse'] - ref_metrics['rmse']) / ref_metrics['rmse']) * 100
    mae_change = ((curr_metrics['mae'] - ref_metrics['mae']) / ref_metrics['mae']) * 100
    r2_change = ((curr_metrics['r2'] - ref_metrics['r2']) / ref_metrics['r2']) * 100
    
    print("\nPerformance Change:")
    print(f"  RMSE: {rmse_change:+.2f}%")
    print(f"  MAE: {mae_change:+.2f}%")
    print(f"  R²: {r2_change:+.2f}%")
    
    # ===== GENERATE SUMMARY =====
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'tool_used': 'Evidently',
        'evidently_version': '0.2.8',
        'data_periods': {
            'reference': {
                'start': str(reference_df['datetime'].min()),
                'end': str(reference_df['datetime'].max()),
                'rows': len(reference_df)
            },
            'current': {
                'start': str(current_df['datetime'].min()),
                'end': str(current_df['datetime'].max()),
                'rows': len(current_df)
            }
        },
        'performance_metrics': {
            'reference': ref_metrics,
            'current': curr_metrics,
            'change_pct': {
                'rmse': float(rmse_change),
                'mae': float(mae_change),
                'r2': float(r2_change)
            }
        },
        'reports_generated': {
            'data_drift_dashboard': str(drift_dashboard_path),
            'regression_performance_dashboard': str(regression_dashboard_path),
            'note': 'JSON profiles skipped due to pydantic compatibility issues'
        },
        'recommendations': []
    }
    
    # Add recommendations
    if abs(rmse_change) < 5:
        summary['recommendations'].append("✓ Model performance is stable")
    elif rmse_change > 10:
        summary['recommendations'].append("⚠ Significant performance degradation - consider retraining")
    elif rmse_change > 5:
        summary['recommendations'].append("⚠ Model performance degraded - monitor closely")
    else:
        summary['recommendations'].append("✓ Model performance improved on recent data")
    
    # Save summary
    summary_path = output_dir / "evidently_drift_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("DRIFT ANALYSIS COMPLETED WITH EVIDENTLY!")
    print("="*70)
    print("\nGenerated Reports:")
    print(f"  1. Data Drift Dashboard (HTML): {drift_dashboard_path}")
    print(f"  2. Regression Performance Dashboard (HTML): {regression_dashboard_path}")
    print(f"  3. Summary (JSON): {summary_path}")
    print("\nRecommendations:")
    for rec in summary['recommendations']:
        print(f"  {rec}")
    print("\n✓ Open the HTML dashboards in a browser to view detailed visualizations")
    print("="*70)

if __name__ == "__main__":
    main()
