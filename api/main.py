"""
FastAPI Application for PM2.5 Prediction

This application serves three trained models:
- Model 1: GBM (Gradient Boosting)
- Model 2: Random Forest
- Model 3: AdaBoost

Each model is loaded from MLflow Model Registry and exposed via REST API endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from api.models import PredictionInput, PredictionOutput, HealthResponse

# Initialize FastAPI app
app = FastAPI(
    title="PM2.5 Prediction API",
    description="API for predicting PM2.5 air quality levels using multiple ML models",
    version="1.0.0"
)

# Global variables for loaded models
models = {}
MODEL_INFO = {
    "model1": {"name": "PM25_GBM", "stage": "Production", "type": "GBM"},
    "model2": {"name": "PM25_RandomForest", "version": "2", "type": "RandomForest"},
    "model3": {"name": "PM25_AdaBoost", "version": "1", "type": "AdaBoost"}
}

def load_models():
    """Load all models from remote MLflow"""
    global models
    
    # Set MLflow tracking URI to remote EC2 server
    mlflow.set_tracking_uri("http://13.217.233.217:5000")
    
    for model_key, info in MODEL_INFO.items():
        try:
            # Load champion model from Production stage, others by version
            if "stage" in info:
                model_uri = f"models:/{info['name']}/{info['stage']}"
                models[model_key] = mlflow.sklearn.load_model(model_uri)
                print(f"✓ Loaded {info['name']} ({info['stage']} stage)")
            else:
                model_uri = f"models:/{info['name']}/{info['version']}"
                models[model_key] = mlflow.sklearn.load_model(model_uri)
                print(f"✓ Loaded {info['name']} v{info['version']}")
        except Exception as e:
            print(f"✗ Failed to load {info['name']}: {e}")
            models[model_key] = None

def prepare_features(input_data: PredictionInput) -> pd.DataFrame:
    """
    Prepare features from input data
    
    Converts input data to the same format used during training
    """
    # Create base features
    features = {
        'year': input_data.year,
        'month': input_data.month,
        'day': input_data.day,
        'hour': input_data.hour,
        'DEWP': input_data.DEWP,
        'TEMP': input_data.TEMP,
        'PRES': input_data.PRES,
        'Iws': input_data.Iws,
        'Is': input_data.Is,
        'Ir': input_data.Ir,
    }
    
    # One-hot encode wind direction
    wind_directions = ['NE', 'NW', 'SE', 'cv']
    for direction in wind_directions:
        features[f'cbwd_{direction}'] = 1 if input_data.cbwd == direction else 0
    
    # Calculate day of week (0=Monday, 6=Sunday)
    from datetime import date
    try:
        dt = date(input_data.year, input_data.month, input_data.day)
        features['day_of_week'] = dt.weekday()
    except ValueError:
        features['day_of_week'] = 0  # Default to Monday if invalid date
    
    # Calculate season
    month = input_data.month
    if month in [12, 1, 2]:
        features['season'] = 1  # Winter
    elif month in [3, 4, 5]:
        features['season'] = 2  # Spring
    elif month in [6, 7, 8]:
        features['season'] = 3  # Summer
    else:
        features['season'] = 4  # Fall
    
    # Hour category one-hot encoding
    hour = input_data.hour
    features['hour_cat_afternoon'] = 1 if 12 <= hour < 18 else 0
    features['hour_cat_evening'] = 1 if 18 <= hour < 24 else 0
    features['hour_cat_morning'] = 1 if 6 <= hour < 12 else 0
    features['hour_cat_night'] = 1 if 0 <= hour < 6 else 0
    
    # Convert to DataFrame with correct column order
    df = pd.DataFrame([features])
    
    # Ensure column order matches training data
    expected_columns = [
        'year', 'month', 'day', 'hour', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir',
        'cbwd_NE', 'cbwd_NW', 'cbwd_SE', 'cbwd_cv',
        'day_of_week', 'season',
        'hour_cat_afternoon', 'hour_cat_evening', 'hour_cat_morning', 'hour_cat_night'
    ]
    
    return df[expected_columns]

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("Loading models...")
    load_models()
    print("API ready!")

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    models_status = {
        key: "loaded" if model is not None else "failed"
        for key, model in models.items()
    }
    
    return HealthResponse(
        status="healthy" if all(m is not None for m in models.values()) else "degraded",
        message="PM2.5 Prediction API is running",
        models_loaded=models_status,
        timestamp=datetime.now()
    )

@app.post("/predict_model1", response_model=PredictionOutput)
async def predict_model1(input_data: PredictionInput):
    """
    Predict PM2.5 using Model 1 (GBM - Gradient Boosting)
    
    This is the champion model with best performance.
    """
    if models.get("model1") is None:
        raise HTTPException(status_code=503, detail="Model 1 (GBM) not loaded")
    
    try:
        # Prepare features
        features_df = prepare_features(input_data)
        
        # Make prediction
        prediction = models["model1"].predict(features_df)[0]
        
        # Ensure prediction is non-negative
        prediction = max(0, prediction)
        
        return PredictionOutput(
            prediction=float(prediction),
            model_name=MODEL_INFO["model1"]["name"],
            model_version=MODEL_INFO["model1"].get("stage", MODEL_INFO["model1"].get("version")),
            timestamp=datetime.now(),
            input_features=input_data.dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_model2", response_model=PredictionOutput)
async def predict_model2(input_data: PredictionInput):
    """
    Predict PM2.5 using Model 2 (Random Forest)
    
    Second-best performer, good for ensemble predictions.
    """
    if models.get("model2") is None:
        raise HTTPException(status_code=503, detail="Model 2 (Random Forest) not loaded")
    
    try:
        # Prepare features
        features_df = prepare_features(input_data)
        
        # Make prediction
        prediction = models["model2"].predict(features_df)[0]
        
        # Ensure prediction is non-negative
        prediction = max(0, prediction)
        
        return PredictionOutput(
            prediction=float(prediction),
            model_name=MODEL_INFO["model2"]["name"],
            model_version=MODEL_INFO["model2"]["version"],
            timestamp=datetime.now(),
            input_features=input_data.dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_model3", response_model=PredictionOutput)
async def predict_model3(input_data: PredictionInput):
    """
    Predict PM2.5 using Model 3 (AdaBoost)
    
    Alternative boosting model for comparison.
    """
    if models.get("model3") is None:
        raise HTTPException(status_code=503, detail="Model 3 (AdaBoost) not loaded")
    
    try:
        # Prepare features
        features_df = prepare_features(input_data)
        
        # Make prediction
        prediction = models["model3"].predict(features_df)[0]
        
        # Ensure prediction is non-negative
        prediction = max(0, prediction)
        
        return PredictionOutput(
            prediction=float(prediction),
            model_name=MODEL_INFO["model3"]["name"],
            model_version=MODEL_INFO["model3"]["version"],
            timestamp=datetime.now(),
            input_features=input_data.dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/models")
async def list_models():
    """List all available models and their status"""
    return {
        "models": [
            {
                "endpoint": "/predict_model1",
                "name": MODEL_INFO["model1"]["name"],
                "version": MODEL_INFO["model1"].get("stage", MODEL_INFO["model1"].get("version")),
                "type": MODEL_INFO["model1"]["type"],
                "status": "loaded" if models.get("model1") is not None else "failed",
                "description": "Gradient Boosting Machine (Champion Model - Production)"
            },
            {
                "endpoint": "/predict_model2",
                "name": MODEL_INFO["model2"]["name"],
                "version": MODEL_INFO["model2"]["version"],
                "type": MODEL_INFO["model2"]["type"],
                "status": "loaded" if models.get("model2") is not None else "failed",
                "description": "Random Forest (Second-best - Staging)"
            },
            {
                "endpoint": "/predict_model3",
                "name": MODEL_INFO["model3"]["name"],
                "version": MODEL_INFO["model3"]["version"],
                "type": MODEL_INFO["model3"]["type"],
                "status": "loaded" if models.get("model3") is not None else "failed",
                "description": "AdaBoost (Alternative - Archived)"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
