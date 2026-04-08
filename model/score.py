
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
