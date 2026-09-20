from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path


# -----------------------------
# Load trained model
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"

model = joblib.load(MODEL_PATH)


# -----------------------------
# Create FastAPI application
# -----------------------------

app = FastAPI(
    title="Predictive Maintenance API",
    description="Machine failure risk prediction using Machine Learning",
    version="1.0"
)


# -----------------------------
# Enable CORS
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Input data format
# -----------------------------

class MachineData(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


# -----------------------------
# Home endpoint
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API is running"
    }


# -----------------------------
# Health check
# -----------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Random Forest"
    }


# -----------------------------
# Prediction endpoint
# -----------------------------

@app.post("/predict")
def predict(data: MachineData):

    input_data = pd.DataFrame([{
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        result = "Machine Failure Risk Detected"
        recommendation = "Inspect the machine and schedule maintenance."
    else:
        result = "Machine Operating Normally"
        recommendation = "Continue monitoring the machine."

    return {
        "prediction": int(prediction),
        "failure_probability": round(float(probability) * 100, 2),
        "result": result,
        "recommendation": recommendation
    }