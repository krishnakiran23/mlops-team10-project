"""
Simple Model Promotion Script

Promotes GBM to Production stage in MLflow Model Registry.
"""

import mlflow
from mlflow.tracking import MlflowClient

# Set tracking URI
mlflow.set_tracking_uri("http://13.217.233.217:5000")

# Create client
client = MlflowClient()

print("Promoting PM25_GBM to Production...")

try:
    # Get latest version of GBM model
    versions = client.search_model_versions("name='PM25_GBM'")
    if versions:
        version = versions[0].version
        
        # Promote to Production
        client.transition_model_version_stage(
            name="PM25_GBM",
            version=version,
            stage="Production"
        )
        
        print(f"✅ PM25_GBM version {version} promoted to Production!")
        print(f"🔗 View at: http://13.217.233.217:5000/#/models/PM25_GBM")
    else:
        print("❌ No versions found for PM25_GBM")
        
except Exception as e:
    print(f"❌ Error: {e}")
