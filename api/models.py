"""
Pydantic models for FastAPI request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class PredictionInput(BaseModel):
    """Input features for PM2.5 prediction"""
    
    year: int = Field(..., ge=2010, le=2030, description="Year")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    day: int = Field(..., ge=1, le=31, description="Day of month (1-31)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    DEWP: float = Field(..., description="Dew Point temperature (°C)")
    TEMP: float = Field(..., description="Temperature (°C)")
    PRES: float = Field(..., description="Pressure (hPa)")
    Iws: float = Field(..., ge=0, description="Cumulated wind speed (m/s)")
    Is: int = Field(..., ge=0, description="Cumulated hours of snow")
    Ir: int = Field(..., ge=0, description="Cumulated hours of rain")
    cbwd: str = Field(..., description="Combined wind direction (NE, NW, SE, cv)")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }

class PredictionOutput(BaseModel):
    """Output prediction with metadata"""
    
    prediction: float = Field(..., description="Predicted PM2.5 concentration")
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(..., description="Version of the model")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    input_features: dict = Field(..., description="Echo of input features")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 95.5,
                "model_name": "PM25_GBM",
                "model_version": "1",
                "timestamp": "2024-12-10T15:30:00",
                "input_features": {
                    "year": 2014,
                    "month": 12,
                    "day": 15,
                    "hour": 14,
                    "DEWP": -15.0,
                    "TEMP": -5.0,
                    "PRES": 1025.0
                }
            }
        }

class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    models_loaded: dict = Field(..., description="Loaded models status")
    timestamp: datetime = Field(..., description="Response timestamp")
