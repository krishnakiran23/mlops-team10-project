"""
H2O AutoML Analysis Script

This script:
1. Initializes H2O cluster
2. Loads training data
3. Runs H2O AutoML
4. Identifies top 3 model types
5. Saves leaderboard and results
"""

import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from pathlib import Path
import json

def initialize_h2o(max_mem_size="4G"):
    """Initialize H2O cluster"""
    print("Initializing H2O cluster...")
    h2o.init(max_mem_size=max_mem_size)
    print(f"H2O cluster initialized: {h2o.cluster().cloud_name}")
    print(f"H2O version: {h2o.__version__}")
    return h2o

def load_training_data(train_path):
    """Load training data into H2O frame"""
    print(f"\nLoading training data from {train_path}...")
    
    # Load with pandas first to inspect
    df_pandas = pd.read_csv(train_path)
    print(f"Loaded {len(df_pandas)} rows with {len(df_pandas.columns)} columns")
    
    # Convert to H2O frame
    h2o_frame = h2o.H2OFrame(df_pandas)
    
    return h2o_frame, df_pandas

def prepare_features(h2o_frame):
    """Prepare features and target for modeling"""
    print("\nPreparing features and target...")
    
    # Define target variable
    target = 'pm2.5'
    
    # Define features to exclude
    exclude_cols = [
        'No',  # Row number
        'pm2.5',  # Target variable
        'datetime',  # Datetime string
        'cbwd',  # Original categorical (we have one-hot encoded versions)
        'hour_category',  # Original categorical (we have one-hot encoded versions)
    ]
    
    # Get all column names
    all_cols = h2o_frame.columns
    
    # Select features (all columns except excluded ones)
    features = [col for col in all_cols if col not in exclude_cols]
    
    print(f"Target variable: {target}")
    print(f"Number of features: {len(features)}")
    print(f"Features: {features[:10]}... (showing first 10)")
    
    return features, target

def run_automl(h2o_frame, features, target, max_runtime_secs=600, max_models=20):
    """
    Run H2O AutoML
    
    Args:
        h2o_frame: H2O frame with data
        features: List of feature column names
        target: Target column name
        max_runtime_secs: Maximum runtime in seconds (default 600 = 10 minutes)
        max_models: Maximum number of models to train (default 20)
    """
    print(f"\nRunning H2O AutoML...")
    print(f"Max runtime: {max_runtime_secs} seconds")
    print(f"Max models: {max_models}")
    
    # Initialize AutoML
    aml = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        max_models=max_models,
        seed=42,
        sort_metric='RMSE',  # Primary metric for regression
        nfolds=5,  # 5-fold cross-validation
        keep_cross_validation_predictions=True,
        keep_cross_validation_models=True,
        verbosity='info'
    )
    
    # Train AutoML
    print("\nTraining AutoML models...")
    aml.train(x=features, y=target, training_frame=h2o_frame)
    
    print("\n✓ AutoML training completed!")
    
    return aml

def analyze_leaderboard(aml, output_dir):
    """Analyze and save AutoML leaderboard"""
    print("\n" + "="*70)
    print("AUTOML LEADERBOARD")
    print("="*70)
    
    # Get leaderboard
    lb = aml.leaderboard
    print("\nTop 10 Models:")
    print(lb.head(10))
    
    # Convert to pandas for easier manipulation
    lb_df = lb.as_data_frame()
    
    # Save full leaderboard
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    leaderboard_path = output_dir / "automl_leaderboard.csv"
    lb_df.to_csv(leaderboard_path, index=False)
    print(f"\n✓ Saved leaderboard to {leaderboard_path}")
    
    return lb_df

def identify_top_model_types(lb_df, top_n=3):
    """Identify top N model types from leaderboard"""
    print("\n" + "="*70)
    print(f"TOP {top_n} MODEL TYPES")
    print("="*70)
    
    # Extract model type from model_id
    # H2O model IDs typically start with model type (e.g., "GBM_", "XGBoost_", "DRF_", "GLM_")
    def extract_model_type(model_id):
        """Extract model type from H2O model ID"""
        if 'StackedEnsemble' in model_id:
            return 'StackedEnsemble'
        elif 'GBM' in model_id:
            return 'GBM'
        elif 'XGBoost' in model_id or 'XRT' in model_id:
            return 'XGBoost'
        elif 'DRF' in model_id:
            return 'RandomForest'
        elif 'GLM' in model_id:
            return 'GLM'
        elif 'DeepLearning' in model_id:
            return 'DeepLearning'
        else:
            # Try to extract from the beginning of model_id
            return model_id.split('_')[0]
    
    lb_df['model_type'] = lb_df['model_id'].apply(extract_model_type)
    
    # Get top models (excluding stacked ensembles for base model selection)
    base_models = lb_df[lb_df['model_type'] != 'StackedEnsemble'].copy()
    
    # Get unique model types in order of performance
    top_model_types = []
    seen_types = set()
    
    for idx, row in base_models.iterrows():
        model_type = row['model_type']
        if model_type not in seen_types and len(top_model_types) < top_n:
            top_model_types.append({
                'rank': len(top_model_types) + 1,
                'model_type': model_type,
                'model_id': row['model_id'],
                'rmse': row['rmse'],
                'mae': row['mae'] if 'mae' in row else None,
                'r2': row.get('mean_residual_deviance', None)
            })
            seen_types.add(model_type)
    
    print("\nTop Model Types (for manual training):")
    for i, model_info in enumerate(top_model_types, 1):
        print(f"\n{i}. {model_info['model_type']}")
        print(f"   Model ID: {model_info['model_id']}")
        print(f"   RMSE: {model_info['rmse']:.4f}")
        if model_info['mae']:
            print(f"   MAE: {model_info['mae']:.4f}")
    
    return top_model_types

def save_results(top_model_types, output_dir):
    """Save AutoML results"""
    output_dir = Path(output_dir)
    
    results_path = output_dir / "automl_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'top_model_types': top_model_types,
            'timestamp': pd.Timestamp.now().isoformat()
        }, f, indent=2)
    
    print(f"\n✓ Saved results to {results_path}")

def main():
    """Main AutoML pipeline"""
    # Define paths
    train_path = Path("data/processed/train.csv")
    output_dir = Path("data/processed")
    
    try:
        # Initialize H2O
        h2o_instance = initialize_h2o(max_mem_size="4G")
        
        # Load training data
        h2o_frame, df_pandas = load_training_data(train_path)
        
        # Prepare features
        features, target = prepare_features(h2o_frame)
        
        # Run AutoML (10 minutes max runtime)
        aml = run_automl(h2o_frame, features, target, 
                        max_runtime_secs=600, 
                        max_models=20)
        
        # Analyze leaderboard
        lb_df = analyze_leaderboard(aml, output_dir)
        
        # Identify top 3 model types
        top_model_types = identify_top_model_types(lb_df, top_n=3)
        
        # Save results
        save_results(top_model_types, output_dir)
        
        print("\n" + "="*70)
        print("H2O AutoML analysis completed successfully!")
        print("="*70)
        
        print("\nNext steps:")
        print("1. Review the leaderboard: data/processed/automl_leaderboard.csv")
        print("2. Check top model types: data/processed/automl_results.json")
        print("3. Create manual training scripts for the top 3 model types")
        
    finally:
        # Shutdown H2O cluster
        print("\nShutting down H2O cluster...")
        h2o.cluster().shutdown()
        print("✓ H2O cluster shut down")

if __name__ == "__main__":
    main()
