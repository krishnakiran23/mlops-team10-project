"""
Time-Based Data Splitting Script

This script:
1. Loads the cleaned dataset
2. Sorts by datetime
3. Splits into train (35%), validate (35%), test (30%)
4. Saves each split to separate CSV files
5. Logs split statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_cleaned_data(filepath):
    """Load the cleaned dataset"""
    print(f"Loading cleaned data from {filepath}...")
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"Loaded {len(df)} rows")
    return df

def perform_time_based_split(df, train_pct=0.35, val_pct=0.35, test_pct=0.30):
    """
    Perform time-based split
    
    Args:
        df: DataFrame sorted by datetime
        train_pct: Percentage for training set (default 0.35)
        val_pct: Percentage for validation set (default 0.35)
        test_pct: Percentage for test set (default 0.30)
    
    Returns:
        train_df, val_df, test_df
    """
    print("\nPerforming time-based split...")
    print(f"Split ratios: Train={train_pct*100}%, Validate={val_pct*100}%, Test={test_pct*100}%")
    
    # Ensure data is sorted by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    total_rows = len(df)
    
    # Calculate split indices
    train_end_idx = int(total_rows * train_pct)
    val_end_idx = int(total_rows * (train_pct + val_pct))
    
    # Split the data
    train_df = df.iloc[:train_end_idx].copy()
    val_df = df.iloc[train_end_idx:val_end_idx].copy()
    test_df = df.iloc[val_end_idx:].copy()
    
    return train_df, val_df, test_df

def print_split_statistics(train_df, val_df, test_df):
    """Print statistics for each split"""
    print("\n" + "="*70)
    print("SPLIT STATISTICS")
    print("="*70)
    
    total_rows = len(train_df) + len(val_df) + len(test_df)
    
    # Training set
    print("\nTRAINING SET:")
    print(f"  Rows: {len(train_df)} ({len(train_df)/total_rows*100:.1f}%)")
    print(f"  Date range: {train_df['datetime'].min()} to {train_df['datetime'].max()}")
    print(f"  PM2.5 stats: mean={train_df['pm2.5'].mean():.2f}, "
          f"median={train_df['pm2.5'].median():.2f}, "
          f"std={train_df['pm2.5'].std():.2f}")
    print(f"  PM2.5 range: [{train_df['pm2.5'].min():.2f}, {train_df['pm2.5'].max():.2f}]")
    
    # Validation set
    print("\nVALIDATION SET:")
    print(f"  Rows: {len(val_df)} ({len(val_df)/total_rows*100:.1f}%)")
    print(f"  Date range: {val_df['datetime'].min()} to {val_df['datetime'].max()}")
    print(f"  PM2.5 stats: mean={val_df['pm2.5'].mean():.2f}, "
          f"median={val_df['pm2.5'].median():.2f}, "
          f"std={val_df['pm2.5'].std():.2f}")
    print(f"  PM2.5 range: [{val_df['pm2.5'].min():.2f}, {val_df['pm2.5'].max():.2f}]")
    
    # Test set
    print("\nTEST SET:")
    print(f"  Rows: {len(test_df)} ({len(test_df)/total_rows*100:.1f}%)")
    print(f"  Date range: {test_df['datetime'].min()} to {test_df['datetime'].max()}")
    print(f"  PM2.5 stats: mean={test_df['pm2.5'].mean():.2f}, "
          f"median={test_df['pm2.5'].median():.2f}, "
          f"std={test_df['pm2.5'].std():.2f}")
    print(f"  PM2.5 range: [{test_df['pm2.5'].min():.2f}, {test_df['pm2.5'].max():.2f}]")
    
    print("\n" + "="*70)
    
    # Verify no overlap in dates
    assert train_df['datetime'].max() < val_df['datetime'].min(), "Train and validation sets overlap!"
    assert val_df['datetime'].max() < test_df['datetime'].min(), "Validation and test sets overlap!"
    print("✓ Verified: No temporal overlap between splits")
    print("="*70)

def save_splits(train_df, val_df, test_df, output_dir):
    """Save each split to separate CSV files"""
    print(f"\nSaving splits to {output_dir}...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.csv"
    val_path = output_dir / "validate.csv"
    test_path = output_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    print(f"  Saved training set to {train_path}")
    
    val_df.to_csv(val_path, index=False)
    print(f"  Saved validation set to {val_path}")
    
    test_df.to_csv(test_path, index=False)
    print(f"  Saved test set to {test_path}")
    
    print("\n✓ All splits saved successfully!")

def main():
    """Main data splitting pipeline"""
    # Define paths
    cleaned_data_path = Path("data/processed/cleaned_data.csv")
    output_dir = Path("data/processed")
    
    # Load cleaned data
    df = load_cleaned_data(cleaned_data_path)
    
    # Perform time-based split (35% train, 35% validate, 30% test)
    train_df, val_df, test_df = perform_time_based_split(df, 
                                                          train_pct=0.35, 
                                                          val_pct=0.35, 
                                                          test_pct=0.30)
    
    # Print statistics
    print_split_statistics(train_df, val_df, test_df)
    
    # Save splits
    save_splits(train_df, val_df, test_df, output_dir)
    
    print("\n" + "="*70)
    print("Time-based data splitting completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()
