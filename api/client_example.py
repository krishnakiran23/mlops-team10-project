"""
Example client for PM2.5 Prediction API

This script demonstrates how to interact with the FastAPI endpoints.
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("="*70)
    print("TESTING HEALTH CHECK")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_list_models():
    """Test the list models endpoint"""
    print("="*70)
    print("LISTING AVAILABLE MODELS")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_prediction(endpoint, model_name):
    """Test a prediction endpoint"""
    print("="*70)
    print(f"TESTING {model_name}")
    print("="*70)
    
    # Example input data (winter day in Beijing)
    input_data = {
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
    
    print(f"Input Data:")
    print(json.dumps(input_data, indent=2))
    print()
    
    response = requests.post(f"{BASE_URL}/{endpoint}", json=input_data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Result:")
        print(f"  PM2.5 Prediction: {result['prediction']:.2f} µg/m³")
        print(f"  Model: {result['model_name']} v{result['model_version']}")
        print(f"  Timestamp: {result['timestamp']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_multiple_scenarios():
    """Test multiple weather scenarios"""
    print("="*70)
    print("TESTING MULTIPLE SCENARIOS")
    print("="*70)
    
    scenarios = [
        {
            "name": "Cold Winter Day",
            "data": {
                "year": 2014, "month": 1, "day": 15, "hour": 10,
                "DEWP": -20.0, "TEMP": -10.0, "PRES": 1030.0,
                "Iws": 5.0, "Is": 0, "Ir": 0, "cbwd": "NW"
            }
        },
        {
            "name": "Warm Summer Day",
            "data": {
                "year": 2014, "month": 7, "day": 15, "hour": 14,
                "DEWP": 20.0, "TEMP": 30.0, "PRES": 1010.0,
                "Iws": 3.0, "Is": 0, "Ir": 0, "cbwd": "SE"
            }
        },
        {
            "name": "Rainy Day",
            "data": {
                "year": 2014, "month": 5, "day": 10, "hour": 16,
                "DEWP": 10.0, "TEMP": 15.0, "PRES": 1015.0,
                "Iws": 8.0, "Is": 0, "Ir": 5, "cbwd": "cv"
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Input: {scenario['data']}")
        
        response = requests.post(f"{BASE_URL}/predict_model1", json=scenario['data'])
        if response.status_code == 200:
            result = response.json()
            print(f"  Predicted PM2.5: {result['prediction']:.2f} µg/m³")
        else:
            print(f"  Error: {response.status_code}")
    print()

def generate_curl_examples():
    """Generate curl command examples"""
    print("="*70)
    print("CURL COMMAND EXAMPLES")
    print("="*70)
    
    print("\n1. Health Check:")
    print("curl -X GET http://localhost:8000/")
    
    print("\n2. List Models:")
    print("curl -X GET http://localhost:8000/models")
    
    print("\n3. Predict with Model 1 (GBM):")
    print("""curl -X POST http://localhost:8000/predict_model1 \\
  -H "Content-Type: application/json" \\
  -d '{
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
  }'""")
    
    print("\n4. Predict with Model 2 (Random Forest):")
    print("curl -X POST http://localhost:8000/predict_model2 \\")
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"year": 2014, "month": 7, "day": 15, "hour": 14, "DEWP": 20.0, "TEMP": 30.0, "PRES": 1010.0, "Iws": 3.0, "Is": 0, "Ir": 0, "cbwd": "SE"}\'')
    print()

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("PM2.5 PREDICTION API CLIENT")
    print("="*70 + "\n")
    
    try:
        # Test health check
        test_health_check()
        
        # List models
        test_list_models()
        
        # Test each model
        test_prediction("predict_model1", "Model 1 (GBM)")
        test_prediction("predict_model2", "Model 2 (Random Forest)")
        test_prediction("predict_model3", "Model 3 (AdaBoost)")
        
        # Test multiple scenarios
        test_multiple_scenarios()
        
        # Show curl examples
        generate_curl_examples()
        
        print("="*70)
        print("ALL TESTS COMPLETED!")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("Make sure the API is running:")
        print("  uv run uvicorn api.main:app --reload")
        print()

if __name__ == "__main__":
    main()
