"""
Model Drift Analysis Script

This script detects data drift and performance drift by comparing
reference data (validation set) with production data (test set).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load validation and test datasets"""
    print("Loading datasets...")
    
    # Use validation set as reference (earlier time period)
    reference_df = pd.read_csv("data/processed/validate.csv")
    
    # Use test set as production/current data (later time period)
    current_df = pd.read_csv("data/processed/test.csv")
    
    print(f"Reference data: {len(reference_df)} rows")
    print(f"Current data: {len(current_df)} rows")
    print(f"Reference period: {reference_df['datetime'].min()} to {reference_df['datetime'].max()}")
    print(f"Current period: {current_df['datetime'].min()} to {current_df['datetime'].max()}")
    
    return reference_df, current_df

def analyze_data_drift(reference_df, current_df):
    """Analyze data drift using statistical tests"""
    print("\n" + "="*70)
    print("ANALYZING DATA DRIFT")
    print("="*70)
    
    # Select numerical features
    numerical_features = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir', 'pm2.5']
    
    drift_results = {}
    
    for feature in numerical_features:
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(
            reference_df[feature].dropna(),
            current_df[feature].dropna()
        )
        
        # Calculate distribution statistics
        ref_mean = reference_df[feature].mean()
        curr_mean = current_df[feature].mean()
        mean_change = ((curr_mean - ref_mean) / ref_mean) * 100 if ref_mean != 0 else 0
        
        drift_results[feature] = {
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'drift_detected': bool(ks_pvalue < 0.05),  # Convert to Python bool
            'ref_mean': float(ref_mean),
            'curr_mean': float(curr_mean),
            'mean_change_pct': float(mean_change)
        }
        
        print(f"\n{feature}:")
        print(f"  KS Statistic: {ks_stat:.4f}")
        print(f"  P-value: {ks_pvalue:.4f}")
        print(f"  Drift Detected: {'Yes' if ks_pvalue < 0.05 else 'No'}")
        print(f"  Mean Change: {mean_change:+.2f}%")
    
    return drift_results

def analyze_model_performance(reference_df, current_df):
    """Analyze model performance on different time periods"""
    print("\n" + "="*70)
    print("ANALYZING MODEL PERFORMANCE")
    print("="*70)
    
    # Load GBM model (champion model)
    import mlflow
    import mlflow.sklearn
    
    mlflow.set_tracking_uri("file:./mlruns")
    model = mlflow.sklearn.load_model("models:/PM25_GBM/1")
    
    # Prepare features
    exclude_cols = ['No', 'pm2.5', 'datetime', 'cbwd', 'hour_category']
    feature_cols = [col for col in reference_df.columns if col not in exclude_cols]
    
    # Make predictions
    X_ref = reference_df[feature_cols]
    y_ref = reference_df['pm2.5']
    y_ref_pred = model.predict(X_ref)
    
    X_curr = current_df[feature_cols]
    y_curr = current_df['pm2.5']
    y_curr_pred = model.predict(X_curr)
    
    # Calculate metrics
    ref_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_ref, y_ref_pred)),
        'mae': mean_absolute_error(y_ref, y_ref_pred),
        'r2': r2_score(y_ref, y_ref_pred)
    }
    
    curr_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_curr, y_curr_pred)),
        'mae': mean_absolute_error(y_curr, y_curr_pred),
        'r2': r2_score(y_curr, y_curr_pred)
    }
    
    print("\nReference Period Performance:")
    print(f"  RMSE: {ref_metrics['rmse']:.4f}")
    print(f"  MAE: {ref_metrics['mae']:.4f}")
    print(f"  R²: {ref_metrics['r2']:.4f}")
    
    print("\nCurrent Period Performance:")
    print(f"  RMSE: {curr_metrics['rmse']:.4f}")
    print(f"  MAE: {curr_metrics['mae']:.4f}")
    print(f"  R²: {curr_metrics['r2']:.4f}")
    
    # Calculate performance degradation
    rmse_change = ((curr_metrics['rmse'] - ref_metrics['rmse']) / ref_metrics['rmse']) * 100
    mae_change = ((curr_metrics['mae'] - ref_metrics['mae']) / ref_metrics['mae']) * 100
    r2_change = ((curr_metrics['r2'] - ref_metrics['r2']) / ref_metrics['r2']) * 100
    
    print("\nPerformance Change:")
    print(f"  RMSE: {rmse_change:+.2f}%")
    print(f"  MAE: {mae_change:+.2f}%")
    print(f"  R²: {r2_change:+.2f}%")
    
    # Create visualization
    create_drift_plots(reference_df, current_df, y_ref, y_ref_pred, y_curr, y_curr_pred)
    
    return {
        'reference': ref_metrics,
        'current': curr_metrics,
        'change_pct': {
            'rmse': rmse_change,
            'mae': mae_change,
            'r2': r2_change
        }
    }

def create_drift_plots(reference_df, current_df, y_ref, y_ref_pred, y_curr, y_curr_pred):
    """Create drift visualization plots"""
    output_dir = Path("drift_reports")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Target Distribution Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(y_ref, bins=50, alpha=0.7, label='Reference', color='blue')
    axes[0].hist(y_curr, bins=50, alpha=0.7, label='Current', color='orange')
    axes[0].set_xlabel('PM2.5')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Target Distribution Comparison')
    axes[0].legend()
    
    # 2. Prediction Error Comparison
    ref_errors = y_ref - y_ref_pred
    curr_errors = y_curr - y_curr_pred
    
    axes[1].hist(ref_errors, bins=50, alpha=0.7, label='Reference', color='blue')
    axes[1].hist(curr_errors, bins=50, alpha=0.7, label='Current', color='orange')
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Prediction Error Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'drift_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Drift plots saved to {output_dir}/drift_distributions.png")

def generate_drift_summary(data_drift_results, performance_results):
    """Generate drift analysis summary"""
    print("\n" + "="*70)
    print("DRIFT ANALYSIS SUMMARY")
    print("="*70)
    
    # Count features with drift
    features_with_drift = sum(1 for result in data_drift_results.values() if result['drift_detected'])
    total_features = len(data_drift_results)
    
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'data_drift': {
            'features_with_drift': features_with_drift,
            'total_features': total_features,
            'drift_percentage': (features_with_drift / total_features) * 100,
            'details': data_drift_results
        },
        'performance_drift': performance_results,
        'recommendations': []
    }
    
    # Add recommendations
    if features_with_drift > total_features * 0.5:
        summary['recommendations'].append("⚠ Significant data drift detected - monitor model performance closely")
    else:
        summary['recommendations'].append("✓ Data distribution is relatively stable")
    
    rmse_change = performance_results['change_pct']['rmse']
    if abs(rmse_change) < 5:
        summary['recommendations'].append("✓ Model performance is stable")
    elif rmse_change > 10:
        summary['recommendations'].append("⚠ Significant performance degradation - consider retraining")
    elif rmse_change > 5:
        summary['recommendations'].append("⚠ Model performance degraded - monitor closely")
    else:
        summary['recommendations'].append("✓ Model performance improved on recent data")
    
    # Save summary
    output_dir = Path("drift_reports")
    summary_path = output_dir / "drift_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nData Drift: {features_with_drift}/{total_features} features ({summary['data_drift']['drift_percentage']:.1f}%)")
    print("\nKey Findings:")
    for rec in summary['recommendations']:
        print(f"  {rec}")
    
    print(f"\n✓ Drift summary saved to {summary_path}")
    
    return summary

def main():
    """Main drift analysis pipeline"""
    print("="*70)
    print("MODEL DRIFT ANALYSIS")
    print("="*70)
    
    # Load data
    reference_df, current_df = load_data()
    
    # Analyze data drift
    drift_results = analyze_data_drift(reference_df, current_df)
    
    # Analyze model performance
    performance_results = analyze_model_performance(reference_df, current_df)
    
    # Generate summary
    summary = generate_drift_summary(drift_results, performance_results)
    
    print("\n" + "="*70)
    print("DRIFT ANALYSIS COMPLETED!")
    print("="*70)
    print("\nGenerated Files:")
    print("  1. Drift Plots: drift_reports/drift_distributions.png")
    print("  2. Summary: drift_reports/drift_summary.json")

if __name__ == "__main__":
    main()

    """Load test and validation datasets"""
    print("Loading datasets...")
    
    # Use validation set as reference (earlier time period)
    reference_df = pd.read_csv("data/processed/validate.csv")
    
    # Use test set as production/current data (later time period)
    current_df = pd.read_csv("data/processed/test.csv")
    
    print(f"Reference data: {len(reference_df)} rows")
    print(f"Current data: {len(current_df)} rows")
    print(f"Reference period: {reference_df['datetime'].min()} to {reference_df['datetime'].max()}")
    print(f"Current period: {current_df['datetime'].min()} to {current_df['datetime'].max()}")
    
    return reference_df, current_df

def prepare_data_for_drift(df):
    """Prepare data for drift analysis"""
    # Select relevant columns (exclude datetime, No, and categorical originals)
    exclude_cols = ['No', 'datetime', 'cbwd', 'hour_category']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    return df[feature_cols]

def analyze_data_drift(reference_df, current_df):
    """Analyze data drift using Evidently"""
    print("\n" + "="*70)
    print("ANALYZING DATA DRIFT")
    print("="*70)
    
    # Prepare data
    ref_data = prepare_data_for_drift(reference_df)
    curr_data = prepare_data_for_drift(current_df)
    
    # Create drift report
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    data_drift_report.run(
        reference_data=ref_data,
        current_data=curr_data
    )
    
    # Save report
    output_dir = Path("drift_reports")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "data_drift_report.html"
    data_drift_report.save_html(str(report_path))
    print(f"\n✓ Data drift report saved to {report_path}")
    
    # Extract drift metrics
    drift_results = data_drift_report.as_dict()
    
    return drift_results, report_path

def analyze_model_performance(reference_df, current_df):
    """Analyze model performance on different time periods"""
    print("\n" + "="*70)
    print("ANALYZING MODEL PERFORMANCE")
    print("="*70)
    
    # Load GBM model (champion model)
    import mlflow
    import mlflow.sklearn
    
    mlflow.set_tracking_uri("file:./mlruns")
    model = mlflow.sklearn.load_model("models:/PM25_GBM/1")
    
    # Prepare features
    exclude_cols = ['No', 'pm2.5', 'datetime', 'cbwd', 'hour_category']
    feature_cols = [col for col in reference_df.columns if col not in exclude_cols]
    
    # Make predictions
    X_ref = reference_df[feature_cols]
    y_ref = reference_df['pm2.5']
    y_ref_pred = model.predict(X_ref)
    
    X_curr = current_df[feature_cols]
    y_curr = current_df['pm2.5']
    y_curr_pred = model.predict(X_curr)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    ref_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_ref, y_ref_pred)),
        'mae': mean_absolute_error(y_ref, y_ref_pred),
        'r2': r2_score(y_ref, y_ref_pred)
    }
    
    curr_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_curr, y_curr_pred)),
        'mae': mean_absolute_error(y_curr, y_curr_pred),
        'r2': r2_score(y_curr, y_curr_pred)
    }
    
    print("\nReference Period Performance:")
    print(f"  RMSE: {ref_metrics['rmse']:.4f}")
    print(f"  MAE: {ref_metrics['mae']:.4f}")
    print(f"  R²: {ref_metrics['r2']:.4f}")
    
    print("\nCurrent Period Performance:")
    print(f"  RMSE: {curr_metrics['rmse']:.4f}")
    print(f"  MAE: {curr_metrics['mae']:.4f}")
    print(f"  R²: {curr_metrics['r2']:.4f}")
    
    # Calculate performance degradation
    rmse_change = ((curr_metrics['rmse'] - ref_metrics['rmse']) / ref_metrics['rmse']) * 100
    mae_change = ((curr_metrics['mae'] - ref_metrics['mae']) / ref_metrics['mae']) * 100
    r2_change = ((curr_metrics['r2'] - ref_metrics['r2']) / ref_metrics['r2']) * 100
    
    print("\nPerformance Change:")
    print(f"  RMSE: {rmse_change:+.2f}%")
    print(f"  MAE: {mae_change:+.2f}%")
    print(f"  R²: {r2_change:+.2f}%")
    
    # Create regression performance report
    ref_data_with_pred = reference_df[feature_cols + ['pm2.5']].copy()
    ref_data_with_pred['prediction'] = y_ref_pred
    
    curr_data_with_pred = current_df[feature_cols + ['pm2.5']].copy()
    curr_data_with_pred['prediction'] = y_curr_pred
    
    regression_report = Report(metrics=[
        RegressionPreset(),
    ])
    
    regression_report.run(
        reference_data=ref_data_with_pred,
        current_data=curr_data_with_pred,
        column_mapping={'target': 'pm2.5', 'prediction': 'prediction'}
    )
    
    # Save report
    output_dir = Path("drift_reports")
    report_path = output_dir / "performance_drift_report.html"
    regression_report.save_html(str(report_path))
    print(f"\n✓ Performance drift report saved to {report_path}")
    
    return {
        'reference': ref_metrics,
        'current': curr_metrics,
        'change_pct': {
            'rmse': rmse_change,
            'mae': mae_change,
            'r2': r2_change
        }
    }, report_path

def generate_drift_summary(data_drift_results, performance_results):
    """Generate drift analysis summary"""
    print("\n" + "="*70)
    print("DRIFT ANALYSIS SUMMARY")
    print("="*70)
    
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'data_drift': {
            'detected': 'Yes' if data_drift_results else 'No',
            'details': 'See data_drift_report.html for detailed analysis'
        },
        'performance_drift': performance_results,
        'recommendations': []
    }
    
    # Add recommendations based on performance change
    rmse_change = performance_results['change_pct']['rmse']
    
    if abs(rmse_change) < 5:
        summary['recommendations'].append("✓ Model performance is stable")
    elif rmse_change > 5:
        summary['recommendations'].append("⚠ Model performance degraded - consider retraining")
    else:
        summary['recommendations'].append("✓ Model performance improved on recent data")
    
    # Save summary
    output_dir = Path("drift_reports")
    summary_path = output_dir / "drift_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nKey Findings:")
    for rec in summary['recommendations']:
        print(f"  {rec}")
    
    print(f"\n✓ Drift summary saved to {summary_path}")
    
    return summary

def main():
    """Main drift analysis pipeline"""
    print("="*70)
    print("MODEL DRIFT ANALYSIS")
    print("="*70)
    
    # Load data
    reference_df, current_df = load_data()
    
    # Analyze data drift
    drift_results, data_drift_path = analyze_data_drift(reference_df, current_df)
    
    # Analyze model performance
    performance_results, perf_drift_path = analyze_model_performance(reference_df, current_df)
    
    # Generate summary
    summary = generate_drift_summary(drift_results, performance_results)
    
    print("\n" + "="*70)
    print("DRIFT ANALYSIS COMPLETED!")
    print("="*70)
    print("\nGenerated Reports:")
    print(f"  1. Data Drift: {data_drift_path}")
    print(f"  2. Performance Drift: {perf_drift_path}")
    print(f"  3. Summary: drift_reports/drift_summary.json")
    print("\nOpen the HTML reports in a browser to view detailed visualizations.")

if __name__ == "__main__":
    main()
