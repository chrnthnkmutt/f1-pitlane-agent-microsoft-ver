"""
deploy_model.py
===============
Run this ONCE before your session to:
  1. Register the trained model in Azure ML
  2. Deploy it as a real-time REST endpoint

Prerequisites:
  pip install azure-ai-ml azure-identity scikit-learn joblib

Set these environment variables (or use a .env file):
  AZURE_SUBSCRIPTION_ID
  AZURE_RESOURCE_GROUP
  AZURE_ML_WORKSPACE

Usage:
  python deploy_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── 1. Train & save the model locally first ──────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("[INFO] Training model from f1_telemetry.csv ...")

df = pd.read_csv("f1_telemetry.csv")
le = LabelEncoder()
df["compound_encoded"] = le.fit_transform(df["tyre_compound"])

FEATURES = [
    "tyre_age", "lap_time_delta", "compound_encoded",
    "gap_to_car_behind", "gap_to_car_ahead",
    "fuel_effect", "safety_car"
]

X = df[FEATURES]
y = df["pit_recommended"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100, max_depth=8,
    random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

# Save model + encoder together
model_dir = Path("model")
model_dir.mkdir(exist_ok=True)
joblib.dump(model, model_dir / "pit_model.pkl")
joblib.dump(le, model_dir / "label_encoder.pkl")
print(f"[OK] Model saved to {model_dir}/")

# ── 2. Create scoring script ─────────────────────────────────────────────────
score_script = '''
import json
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

FEATURES = [
    "tyre_age", "lap_time_delta", "compound_encoded",
    "gap_to_car_behind", "gap_to_car_ahead",
    "fuel_effect", "safety_car"
]

def init():
    global model
    model_path = Path(os.environ.get("AZUREML_MODEL_DIR", ".")) / "pit_model.pkl"
    model = joblib.load(model_path)
    print("Model loaded successfully")

def run(raw_data):
    try:
        data = json.loads(raw_data)
        df = pd.DataFrame([data["input"]])
        pred = int(model.predict(df[FEATURES])[0])
        proba = model.predict_proba(df[FEATURES])[0].tolist()
        return {
            "pit_recommended": pred,
            "confidence": round(max(proba) * 100, 1),
            "stay_out_probability": round(proba[0] * 100, 1),
            "pit_probability": round(proba[1] * 100, 1)
        }
    except Exception as e:
        return {"error": str(e)}
'''

score_path = model_dir / "score.py"
score_path.write_text(score_script)
print("[OK] Scoring script created")

# ── 3. Deploy to Azure ML ────────────────────────────────────────────────────
# NOTE: Uncomment this block when you have an Azure ML workspace ready.
# For the demo, you can also run the agent locally using the mock endpoint below.

"""
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Model, ManagedOnlineEndpoint,
    ManagedOnlineDeployment, Environment, CodeConfiguration
)
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZURE_ML_WORKSPACE"]
)

# Register model
registered_model = ml_client.models.create_or_update(
    Model(
        path=str(model_dir / "pit_model.pkl"),
        name="f1-pit-strategy-model",
        description="F1 pit stop recommendation model — Global Azure demo",
        type="custom_model"
    )
)
print(f"✅ Model registered: {registered_model.name} v{registered_model.version}")

# Create endpoint
endpoint_name = "f1-pit-strategy-endpoint"
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="F1 pit stop strategy prediction endpoint",
    auth_mode="key"
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print(f"✅ Endpoint created: {endpoint_name}")

# Create deployment
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=registered_model,
    environment=Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04",
        conda_file="conda.yml"
    ),
    code_configuration=CodeConfiguration(
        code=str(model_dir),
        scoring_script="score.py"
    ),
    instance_type="Standard_DS2_v2",
    instance_count=1
)
ml_client.online_deployments.begin_create_or_update(deployment).result()
print(f"✅ Deployment complete!")

# Get endpoint URL and key
endpoint_details = ml_client.online_endpoints.get(endpoint_name)
keys = ml_client.online_endpoints.get_keys(endpoint_name)
print(f"\\n📡 Endpoint URL: {endpoint_details.scoring_uri}")
print(f"🔑 API Key:      {keys.primary_key[:8]}...")
print("\\nAdd these to your .env file:")
print(f"  AZURE_ML_ENDPOINT_URL={endpoint_details.scoring_uri}")
print(f"  AZURE_ML_ENDPOINT_KEY={keys.primary_key}")
"""

# ── 4. Local mock endpoint (for demo without Azure ML) ───────────────────────
print("\n" + "="*55)
print("  LOCAL MOCK ENDPOINT (no Azure subscription needed)")
print("="*55)
print("""
For demo purposes without Azure ML, the agent notebook
includes a local mock that calls the model directly.
This means you can demo the full agent flow with zero
Azure configuration — just run the notebook!

When you have an Azure ML workspace, set:
  AZURE_ML_ENDPOINT_URL=<your endpoint URL>
  AZURE_ML_ENDPOINT_KEY=<your endpoint key>

The agent notebook auto-detects which mode to use.
""")
print("[OK] Setup complete. Open agent_pitstop.ipynb to run the demo.")
