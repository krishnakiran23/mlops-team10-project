"""
Data Cleaning Script for PRSA Air Quality Dataset

This script:
1. Loads the raw PRSA dataset
2. Handles missing values
3. Creates datetime column
4. Encodes categorical variables
5. Handles outliers
6. Saves cleaned dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_data(filepath):
    """Load the raw PRSA dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def create_datetime_column(df):
    """Create datetime column from year, month, day, hour"""
    print("\nCreating datetime column...")
    df['datetime'] = pd.to_datetime(
        df[['year', 'month', 'day', 'hour']].rename(
            columns={'year': 'year', 'month': 'month', 'day': 'day', 'hour': 'hour'}
        )
    )
    print(f"Datetime range: {df['datetime'].min()} to {df['datetime'].max()}")
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    print("\nHandling missing values...")
    print("Missing values before cleaning:")
    print(df.isnull().sum())
    
    # Drop rows where target variable (pm2.5) is missing
    # This is critical as we cannot train without target values
    initial_rows = len(df)
    df = df.dropna(subset=['pm2.5'])
    dropped_rows = initial_rows - len(df)
    print(f"\nDropped {dropped_rows} rows with missing pm2.5 values")
    
    # Handle missing values in features
    # For numeric features, use median imputation
    numeric_features = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
    for col in numeric_features:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled {col} missing values with median: {median_val:.2f}")
    
    # For categorical wind direction, forward fill (temporal continuity)
    if df['cbwd'].isnull().sum() > 0:
        df['cbwd'].fillna(method='ffill', inplace=True)
        # If any still remain (at the beginning), backfill
        df['cbwd'].fillna(method='bfill', inplace=True)
        print(f"Forward-filled missing wind direction (cbwd) values")
    
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    
    return df

def encode_categorical_variables(df):
    """Encode categorical variables (wind direction)"""
    print("\nEncoding categorical variables...")
    print(f"Wind direction (cbwd) unique values: {df['cbwd'].unique()}")
    
    # One-hot encode wind direction
    cbwd_dummies = pd.get_dummies(df['cbwd'], prefix='cbwd')
    df = pd.concat([df, cbwd_dummies], axis=1)
    
    print(f"Created {len(cbwd_dummies.columns)} one-hot encoded columns for wind direction")
    
    return df

def handle_outliers(df, column='pm2.5', method='iqr', threshold=3):
    """
    Handle outliers in the target variable
    
    Args:
        df: DataFrame
        column: Column to check for outliers
        method: 'iqr' or 'zscore'
        threshold: IQR multiplier or z-score threshold
    """
    print(f"\nHandling outliers in {column}...")
    print(f"Original {column} stats:")
    print(df[column].describe())
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        print(f"Found {outliers.sum()} outliers using IQR method (threshold={threshold})")
        print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # For air quality data, we'll keep outliers as they represent real pollution events
        # But we'll log them for awareness
        print(f"Keeping outliers as they represent real pollution events")
        
    return df

def create_additional_features(df):
    """Create additional time-based features"""
    print("\nCreating additional features...")
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
    df['season'] = df['month'].apply(lambda x: 
        1 if x in [12, 1, 2] else
        2 if x in [3, 4, 5] else
        3 if x in [6, 7, 8] else 4
    )
    
    # Hour of day categories
    df['hour_category'] = df['hour'].apply(lambda x:
        'night' if x in range(0, 6) else
        'morning' if x in range(6, 12) else
        'afternoon' if x in range(12, 18) else
        'evening'
    )
    
    # One-hot encode hour category
    hour_cat_dummies = pd.get_dummies(df['hour_category'], prefix='hour_cat')
    df = pd.concat([df, hour_cat_dummies], axis=1)
    
    print(f"Created features: day_of_week, season, hour_category")
    
    return df

def save_cleaned_data(df, output_path):
    """Save cleaned dataset"""
    print(f"\nSaving cleaned data to {output_path}...")
    
    # Sort by datetime to ensure temporal order
    df = df.sort_values('datetime').reset_index(drop=True)
    
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    
    # Print final statistics
    print("\nFinal dataset statistics:")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Target variable (pm2.5) range: {df['pm2.5'].min():.2f} to {df['pm2.5'].max():.2f}")
    print(f"Mean pm2.5: {df['pm2.5'].mean():.2f}")
    print(f"Median pm2.5: {df['pm2.5'].median():.2f}")

def main():
    """Main data cleaning pipeline"""
    # Define paths
    raw_data_path = Path("data/raw/PRSA_data_2010.1.1-2014.12.31.csv")
    cleaned_data_path = Path("data/processed/cleaned_data.csv")
    
    # Ensure output directory exists
    cleaned_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(raw_data_path)
    
    # Create datetime column
    df = create_datetime_column(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode categorical variables
    df = encode_categorical_variables(df)
    
    # Handle outliers (log only, keep data)
    df = handle_outliers(df, column='pm2.5', method='iqr', threshold=3)
    
    # Create additional features
    df = create_additional_features(df)
    
    # Save cleaned data
    save_cleaned_data(df, cleaned_data_path)
    
    print("\n" + "="*50)
    print("Data cleaning completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
