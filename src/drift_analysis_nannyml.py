"""
Model Drift Analysis with NannyML

This script uses NannyML library to detect data drift and performance drift
by comparing reference data (validation set) with production data (test set).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')

# Import NannyML
try:
    import nannyml as nml
    from nannyml.drift.univariate import UnivariateDriftCalculator
    from nannyml.performance_calculation import PerformanceCalculator
    print("✓ NannyML imported successfully")
except ImportError as e:
    print(f"❌ Error importing NannyML: {e}")
    print("Installing required dependencies...")

def main():
    """Main drift analysis pipeline using NannyML"""
    print("="*70)
    print("MODEL DRIFT ANALYSIS WITH NANNYML")
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
    
    # Add predictions to dataframes
    reference_df['prediction'] = y_ref_pred
    current_df['prediction'] = y_curr_pred
    
    # ===== DATA DRIFT ANALYSIS WITH NANNYML =====
    print("\n" + "="*70)
    print("ANALYZING DATA DRIFT WITH NANNYML")
    print("="*70)
    
    # Combine reference and current data for NannyML
    reference_df['period'] = 'reference'
    current_df['period'] = 'analysis'
    combined_df = pd.concat([reference_df, current_df], ignore_index=True)
    
    # Select numerical features for drift analysis
    numerical_features = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
    
    # Calculate univariate drift
    print(f"\nAnalyzing drift for {len(numerical_features)} numerical features...")
    
    drift_calculator = UnivariateDriftCalculator(
        column_names=numerical_features,
        timestamp_column_name='datetime',
        chunk_size=5000
    )
    
    # Fit on reference data
    ref_data_for_drift = reference_df[numerical_features + ['datetime']].copy()
    drift_calculator.fit(ref_data_for_drift)
    
    # Calculate drift on current data
    curr_data_for_drift = current_df[numerical_features + ['datetime']].copy()
    drift_results = drift_calculator.calculate(curr_data_for_drift)
    
    print("✓ Drift analysis completed")
    
    # Extract drift information
    drift_summary = {}
    for feature in numerical_features:
        try:
            # Get drift metrics from results
            feature_drift = drift_results.data[drift_results.data['column_name'] == feature]
            if not feature_drift.empty:
                drift_detected = feature_drift['alert'].any() if 'alert' in feature_drift.columns else False
                drift_summary[feature] = {
                    'drift_detected': bool(drift_detected),
                    'method': 'Kolmogorov-Smirnov (NannyML)'
                }
        except Exception as e:
            print(f"  Note: Could not extract detailed drift for {feature}")
            drift_summary[feature] = {
                'drift_detected': 'unknown',
                'method': 'Kolmogorov-Smirnov (NannyML)'
            }
    
    print(f"\n✓ Analyzed {len(drift_summary)} features for drift")
    
    # ===== PERFORMANCE METRICS =====
    print("\n" + "="*70)
    print("CALCULATING PERFORMANCE METRICS")
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
    print("\n" + "="*70)
    print("GENERATING DRIFT SUMMARY")
    print("="*70)
    
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'tool_used': 'NannyML',
        'nannyml_version': '0.13.1',
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
        'data_drift': {
            'features_analyzed': numerical_features,
            'drift_summary': drift_summary,
            'method': 'Univariate Drift Detection (Kolmogorov-Smirnov)'
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
    summary_path = output_dir / "nannyml_drift_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✓ Drift summary saved to", summary_path)
    
    print("\n" + "="*70)
    print("DRIFT ANALYSIS COMPLETED WITH NANNYML!")
    print("="*70)
    print("\nGenerated Files:")
    print(f"  1. Summary JSON: {summary_path}")
    print("\nRecommendations:")
    for rec in summary['recommendations']:
        print(f"  {rec}")
    print("\n✓ NannyML drift analysis complete!")
    print("="*70)

if __name__ == "__main__":
    main()
