# MLOps Team 10 - PRSA Air Quality PM2.5 Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Deployment-green)](https://fastapi.tiangolo.com/)

Complete MLOps pipeline for predicting PM2.5 air quality levels using the PRSA Beijing Air Quality dataset.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Drift Analysis](#drift-analysis)
- [Team Members](#team-members)

## 🎯 Project Overview

This project implements a complete MLOps pipeline including:

- ✅ **Data Cleaning & Preprocessing**: Handling missing values, feature engineering, temporal features
- ✅ **Time-Based Splitting**: 35% train, 35% validate, 30% test (chronological order)
- ✅ **H2O AutoML**: Automated model selection identifying top 3 algorithms
- ✅ **Manual Model Training**: GBM, Random Forest, AdaBoost with MLflow tracking
- ✅ **MLflow Integration**: Experiment tracking, model registry, artifact storage
- ✅ **FastAPI Deployment**: REST API with three model endpoints
- ✅ **Drift Analysis**: Data drift and performance drift monitoring

## 📊 Dataset

**Name**: PRSA Beijing PM2.5 Air Quality Dataset  
**Period**: January 1, 2010 - December 31, 2014  
**Frequency**: Hourly measurements  
**Size**: 43,824 rows (41,757 after cleaning)

### Features

- **Temporal**: year, month, day, hour
- **Meteorological**: DEWP (dew point), TEMP (temperature), PRES (pressure)
- **Wind**: cbwd (combined wind direction), Iws (wind speed)
- **Precipitation**: Is (snow hours), Ir (rain hours)
- **Target**: pm2.5 (PM2.5 concentration in µg/m³)

## 📁 Project Structure

```
MLOps_Team10/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned and split data
│       ├── cleaned_data.csv
│       ├── train.csv
│       ├── validate.csv
│       └── test.csv
├── src/
│   ├── data_cleaning.py        # Data preprocessing
│   ├── data_splitting.py       # Time-based splitting
│   ├── automl_analysis.py      # H2O AutoML
│   ├── train_model1.py         # GBM training
│   ├── train_model2.py         # Random Forest training
│   ├── train_model3.py         # AdaBoost training
│   ├── drift_analysis.py       # Drift detection
│   └── mlflow_config.py        # MLflow configuration
├── api/
│   ├── main.py                 # FastAPI application
│   ├── models.py               # Pydantic models
│   └── client_example.py       # API client examples
├── infrastructure/
│   └── mlflow_setup.md         # MLflow infrastructure guide
├── plots/                      # Model visualization plots
├── drift_reports/              # Drift analysis reports
├── mlruns/                     # MLflow tracking data
├── MODEL_COMPARISON.md         # Model comparison summary
├── pyproject.toml              # UV dependencies
└── README.md                   # This file
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.9+
- UV package manager ([installation guide](https://github.com/astral-sh/uv))
- AWS account (for remote MLflow setup - optional)

### Installation

1. **Clone the repository**
   ```bash
   cd MLOps_Team10
   ```

2. **Install dependencies using UV**
   ```bash
   uv sync
   ```

3. **Verify installation**
   ```bash
   uv run python --version
   ```

## 📖 Usage Guide

### 1. Data Preparation

```bash
# Clean the raw data
uv run python src/data_cleaning.py

# Perform time-based splitting
uv run python src/data_splitting.py
```

**Output**:
- `data/processed/cleaned_data.csv` (41,757 rows)
- `data/processed/train.csv` (14,614 rows, 35%)
- `data/processed/validate.csv` (14,615 rows, 35%)
- `data/processed/test.csv` (12,528 rows, 30%)

### 2. H2O AutoML Analysis

```bash
uv run python src/automl_analysis.py
```

**Output**:
- `data/processed/automl_leaderboard.csv`
- `data/processed/automl_results.json`
- Top 3 model types: GBM, Random Forest, XGBoost

### 3. Train Models with MLflow

```bash
# Train all three models
uv run python src/train_model1.py  # GBM
uv run python src/train_model2.py  # Random Forest
uv run python src/train_model3.py  # AdaBoost
```

**Output**:
- Models logged to MLflow
- Registered in Model Registry
- Plots saved to `plots/`

### 4. View MLflow UI

```bash
uv run mlflow ui
```

Navigate to `http://localhost:5000` to view:
- Experiments and runs
- Model metrics (RMSE, MAE, R²)
- Registered models
- Artifacts (plots, models)

### 5. Deploy FastAPI

```bash
# Start the API server
uv run uvicorn api.main:app --reload
```

API available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

### 6. Test API

```bash
# Run example client
uv run python api/client_example.py
```

Or use curl:
```bash
curl -X POST http://localhost:8000/predict_model1 \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2014, "month": 12, "day": 15, "hour": 14,
    "DEWP": -15.0, "TEMP": -5.0, "PRES": 1025.0,
    "Iws": 10.5, "Is": 0, "Ir": 0, "cbwd": "NW"
  }'
```

### 7. Drift Analysis

```bash
uv run python src/drift_analysis.py
```

**Output**:
- `drift_reports/drift_distributions.png`
- `drift_reports/drift_summary.json`

## 🏆 Model Performance

| Model | Test RMSE | Test MAE | Test R² | Status |
|-------|-----------|----------|---------|--------|
| **GBM** | **66.19** | **46.60** | **0.465** | ✅ **Champion** |
| Random Forest | 68.60 | 49.21 | 0.426 | ✅ Production |
| AdaBoost | 85.81 | 70.47 | 0.102 | ⚠️ Baseline |

### Champion Model: Gradient Boosting Machine (GBM)

- **Best RMSE**: 66.19 (lowest prediction error)
- **Best R²**: 0.465 (explains 46.5% of variance)
- **Algorithm**: sklearn.GradientBoostingRegressor
- **Key Parameters**:
  - n_estimators: 100
  - learning_rate: 0.1
  - max_depth: 6
  - subsample: 0.8

### Top Features (by importance)

1. DEWP (Dew Point)
2. TEMP (Temperature)
3. hour
4. PRES (Pressure)
5. Iws (Wind Speed)

## 🔌 API Documentation

### Endpoints

#### `GET /`
Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "message": "PM2.5 Prediction API is running",
  "models_loaded": {
    "model1": "loaded",
    "model2": "loaded",
    "model3": "loaded"
  },
  "timestamp": "2024-12-10T15:30:00"
}
```

#### `GET /models`
List all available models

#### `POST /predict_model1`
Predict using GBM model (champion)

**Request Body**:
```json
{
  "year": 2014,
  "month": 12,
  "day": 15,
  "hour": 14,
  "DEWP": -15.0,
  "TEMP": -5.0,
  "PRES": 1025.0,
  "Iws": 10.5,
  "Is": 0,
  "Ir": 0,
  "cbwd": "NW"
}
```

**Response**:
```json
{
  "prediction": 95.5,
  "model_name": "PM25_GBM",
  "model_version": "1",
  "timestamp": "2024-12-10T15:30:00",
  "input_features": {...}
}
```

#### `POST /predict_model2`
Predict using Random Forest model

#### `POST /predict_model3`
Predict using AdaBoost model

## 📈 Drift Analysis

### Data Drift Results

- **Features with Drift**: 5/7 (71.4%)
- **Drifted Features**: DEWP, TEMP, PRES, Iws, pm2.5
- **Stable Features**: Is (snow), Ir (rain)

### Performance Drift

| Metric | Reference Period | Current Period | Change |
|--------|-----------------|----------------|--------|
| RMSE | 73.23 | 66.19 | **-9.62%** ✅ |
| MAE | 50.60 | 46.60 | **-7.91%** ✅ |
| R² | 0.389 | 0.465 | **+19.50%** ✅ |

**Key Finding**: Despite significant data drift, model performance actually **improved** on the test set, indicating good generalization.

## 🏗️ MLflow Infrastructure

### Local Setup (Current)

- **Tracking URI**: `file:./mlruns`
- **Backend Store**: Local filesystem
- **Artifact Store**: Local filesystem

### Remote Setup (Production)

See [infrastructure/mlflow_setup.md](infrastructure/mlflow_setup.md) for detailed instructions on:
- AWS EC2 setup for MLflow tracking server
- PostgreSQL backend configuration (Neon.tech)
- S3 artifact storage setup
- Security best practices

## 👥 Team Members

- Team Member 1- Krishna Kiran Sristi
- Team Member 2- Devika Akula
- Team Member 3- Narendra Kumar Kottala



## 📚 Additional Resources

- [H2O AutoML Documentation](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)



---

