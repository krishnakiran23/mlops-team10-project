# Model Comparison Summary

## Overview
This document compares the three trained models for PM2.5 prediction on the PRSA Air Quality dataset.

## Model Performance Comparison

| Model | Test RMSE | Test MAE | Test R² | Training Time |
|-------|-----------|----------|---------|---------------|
| **GBM (Gradient Boosting)** | **66.19** | **46.60** | **0.465** | Fast |
| Random Forest | 68.60 | 49.21 | 0.426 | Medium |
| AdaBoost | 85.81 | 70.47 | 0.102 | Fast |

## Champion Model: Gradient Boosting Machine (GBM)

**Rationale:**
- **Best RMSE**: 66.19 (lowest prediction error)
- **Best MAE**: 46.60 (best average absolute error)
- **Best R²**: 0.465 (explains 46.5% of variance)
- **Balanced Performance**: Good performance across all metrics
- **Generalization**: Reasonable gap between train and test performance

## Model Details

### 1. Gradient Boosting Machine (GBM)
- **Algorithm**: sklearn.GradientBoostingRegressor
- **Key Parameters**:
  - n_estimators: 100
  - learning_rate: 0.1
  - max_depth: 6
  - subsample: 0.8
- **Strengths**: Best overall performance, good generalization
- **MLflow Model Name**: PM25_GBM

### 2. Random Forest
- **Algorithm**: sklearn.RandomForestRegressor
- **Key Parameters**:
  - n_estimators: 100
  - max_depth: 20
  - min_samples_split: 10
  - max_features: sqrt
- **Strengths**: Second-best performance, robust to outliers
- **MLflow Model Name**: PM25_RandomForest

### 3. AdaBoost
- **Algorithm**: sklearn.AdaBoostRegressor
- **Key Parameters**:
  - n_estimators: 100
  - learning_rate: 0.1
  - loss: linear
- **Strengths**: Fast training, simple implementation
- **Weaknesses**: Poorest performance among the three models
- **MLflow Model Name**: PM25_AdaBoost


## Feature Importance (GBM Model)

Top 5 most important features:
1. DEWP (Dew Point)
2. TEMP (Temperature)
3. hour
4. PRES (Pressure)
5. Iws (Wind Speed)

