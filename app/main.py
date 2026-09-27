from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import sqlite3
import io
import csv
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

import json as _json

BASE_DIR       = Path(__file__).resolve().parent.parent
MODEL_PATH     = BASE_DIR / "models" / "improved_random_forest_model.pkl"
METADATA_PATH  = BASE_DIR / "models" / "model_metadata.json"
DB_PATH        = BASE_DIR / "data"   / "predictions.db"


# ─────────────────────────────────────────────
# MODEL METADATA  (loaded from model_metadata.json)
# Contains: hyperparameters, threshold, feature names, metrics.
# ─────────────────────────────────────────────

with open(METADATA_PATH, "r", encoding="utf-8") as _f:
    MODEL_METADATA = _json.load(_f)

# Classification threshold read from metadata — do NOT hard-code this value.
# Currently trained threshold: 0.49 (stored in models/model_metadata.json).
CLASSIFICATION_THRESHOLD: float = float(MODEL_METADATA["selected_threshold"])


# ─────────────────────────────────────────────
# RISK LEVEL THRESHOLDS
# Edit these values to change the risk boundaries.
# ─────────────────────────────────────────────

RISK_THRESHOLDS = {
    "low_max":    30.0,   #   0 – 30  %  → Low Risk
    "medium_max": 70.0,   #  30 – 70  %  → Medium Risk
                          #  70 – 100 %  → High Risk
}


# ─────────────────────────────────────────────
# FEATURE NAMES  (must match model training)
# ─────────────────────────────────────────────

FEATURE_NAMES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

model = joblib.load(MODEL_PATH)


# ─────────────────────────────────────────────
# FASTAPI APPLICATION
# ─────────────────────────────────────────────

app = FastAPI(
    title="Predictive Maintenance API",
    description="Machine failure risk prediction using Machine Learning",
    version="2.0",
)


# ─────────────────────────────────────────────
# CORS  (unchanged – keep wildcard for Render frontend)
# ─────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# SQLITE – create table on startup
# ─────────────────────────────────────────────

def init_db():
    """Create the predictions history table if it does not already exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp           TEXT    NOT NULL,
                air_temperature     REAL    NOT NULL,
                process_temperature REAL    NOT NULL,
                rotational_speed    REAL    NOT NULL,
                torque              REAL    NOT NULL,
                tool_wear           REAL    NOT NULL,
                prediction          INTEGER NOT NULL,
                failure_probability REAL    NOT NULL,
                risk_level          TEXT    NOT NULL,
                result              TEXT    NOT NULL,
                recommendation      TEXT    NOT NULL
            )
        """)
        conn.commit()


init_db()


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_risk_level(probability: float) -> str:
    """Map failure_probability (0–100) to a named risk tier."""
    if probability < RISK_THRESHOLDS["low_max"]:
        return "Low Risk"
    elif probability < RISK_THRESHOLDS["medium_max"]:
        return "Medium Risk"
    else:
        return "High Risk"


def get_recommendation(prediction: int, risk_level: str) -> str:
    """Return a maintenance recommendation based on prediction outcome and risk level."""
    if prediction == 0:
        if risk_level == "Low Risk":
            return "Machine is operating normally. Continue routine monitoring."
        else:
            return (
                "No failure predicted yet, but probability is elevated. "
                "Increase monitoring frequency."
            )
    else:
        if risk_level == "Medium Risk":
            return (
                "Failure risk detected. Schedule a maintenance inspection soon "
                "to prevent unexpected downtime."
            )
        else:  # High Risk
            return (
                "High failure risk! Stop the machine and perform an immediate "
                "inspection. Critical maintenance required."
            )


def get_feature_importance_dict() -> dict:
    """Extract global feature importances from the loaded Random Forest model."""
    return {
        name: round(float(importance), 6)
        for name, importance in zip(FEATURE_NAMES, model.feature_importances_)
    }


def save_prediction_to_db(record: dict):
    """Persist one prediction record to the SQLite database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                timestamp, air_temperature, process_temperature,
                rotational_speed, torque, tool_wear,
                prediction, failure_probability, risk_level,
                result, recommendation
            ) VALUES (
                :timestamp, :air_temperature, :process_temperature,
                :rotational_speed, :torque, :tool_wear,
                :prediction, :failure_probability, :risk_level,
                :result, :recommendation
            )
            """,
            record,
        )
        conn.commit()


def run_prediction(
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
) -> dict:
    """
    Core prediction logic shared by /predict and /predict/batch.

    Uses the improved Random Forest model with a calibrated classification
    threshold read from model_metadata.json (CLASSIFICATION_THRESHOLD).

    The failure_probability is the raw model probability (0–100 %).
    The binary prediction is derived by applying CLASSIFICATION_THRESHOLD
    to that probability — NOT by using model.predict() directly.

    Returns a dict with prediction, failure_probability, risk_level,
    result, and recommendation.
    """
    input_df = pd.DataFrame([{
        "Air temperature [K]":     air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]":  rotational_speed,
        "Torque [Nm]":             torque,
        "Tool wear [min]":         tool_wear,
    }])

    # Raw failure probability (0.0 – 1.0) from the improved RF model
    raw_probability = float(model.predict_proba(input_df)[0][1])

    # Apply the optimised threshold (read from metadata, currently 0.49)
    # failure_probability is kept as the actual model probability %, unchanged.
    probability = round(raw_probability * 100, 2)
    prediction  = int(raw_probability >= CLASSIFICATION_THRESHOLD)
    risk_level  = get_risk_level(probability)

    result_text = (
        "Machine Failure Risk Detected" if prediction == 1
        else "Machine Operating Normally"
    )
    recommendation = get_recommendation(prediction, risk_level)

    return {
        "prediction":          prediction,
        "failure_probability": probability,
        "risk_level":          risk_level,
        "result":              result_text,
        "recommendation":      recommendation,
    }


# ─────────────────────────────────────────────
# INPUT SCHEMA  (unchanged – same five fields)
# ─────────────────────────────────────────────

class MachineData(BaseModel):
    air_temperature:     float
    process_temperature: float
    rotational_speed:    float
    torque:              float
    tool_wear:           float


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

# ── Root ──────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Predictive Maintenance API is running"}


# ── Health ────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":                 "healthy",
        "model":                  "Improved Random Forest",
        "model_file":             MODEL_PATH.name,
        "model_loaded":           True,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "threshold_source":       "models/model_metadata.json",
        "n_estimators":           int(model.n_estimators),
        "features":               FEATURE_NAMES,
        "baseline_f1":            MODEL_METADATA.get("baseline_rf_metrics", {}).get("F1"),
        "improved_f1":            MODEL_METADATA.get("improved_rf_metrics", {}).get("F1"),
        "db_path":                str(DB_PATH),
    }


# ── Single Prediction ─────────────────────────

@app.post("/predict")
def predict(data: MachineData):
    """
    Predict machine failure for a single set of sensor readings.

    Returns:
        prediction          – 0 (normal) or 1 (failure)
        failure_probability – percentage probability of failure
        risk_level          – Low Risk / Medium Risk / High Risk
        result              – human-readable status
        recommendation      – maintenance action to take
        feature_importance  – global feature importances from the RF model
    """
    result = run_prediction(
        air_temperature     = data.air_temperature,
        process_temperature = data.process_temperature,
        rotational_speed    = data.rotational_speed,
        torque              = data.torque,
        tool_wear           = data.tool_wear,
    )

    # Attach global feature importances to the prediction response
    result["feature_importance"] = get_feature_importance_dict()

    # Save to history
    save_prediction_to_db({
        "timestamp":           datetime.utcnow().isoformat(),
        "air_temperature":     data.air_temperature,
        "process_temperature": data.process_temperature,
        "rotational_speed":    data.rotational_speed,
        "torque":              data.torque,
        "tool_wear":           data.tool_wear,
        "prediction":          result["prediction"],
        "failure_probability": result["failure_probability"],
        "risk_level":          result["risk_level"],
        "result":              result["result"],
        "recommendation":      result["recommendation"],
    })

    return result


# ── Feature Importance ────────────────────────

@app.get("/feature-importance")
def feature_importance():
    """
    Return the global feature importances from the trained Random Forest model.
    Values sum to 1.0 and represent each feature's contribution to predictions.
    """
    return {
        "model":              "Random Forest",
        "feature_importance": get_feature_importance_dict(),
    }


# ── Prediction History ────────────────────────

@app.get("/history")
def get_history(limit: int = Query(default=50, ge=1, le=500)):
    """
    Return recent prediction history, newest first.
    Optional query parameter: limit (1–500, default 50).
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return {
        "count":       len(rows),
        "predictions": [dict(r) for r in rows],
    }


@app.delete("/history")
def clear_history():
    """Delete all prediction history records (useful during development/testing)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM predictions")
        conn.commit()
    return {"message": "Prediction history cleared."}


# ── Batch CSV Prediction ──────────────────────

@app.get("/predict/batch/template")
def batch_template():
    """
    Download a CSV template showing the required column names and example rows.
    Use this as a starting point for preparing your batch prediction file.
    """
    rows = [
        FEATURE_NAMES,
        [298.1, 308.6, 1551, 42.8,   0],
        [305.0, 315.0, 1350, 68.5, 220],
        [302.3, 312.1, 1700, 35.2, 180],
        [299.0, 309.5, 1480, 55.0,  95],
    ]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(rows)
    output.seek(0)

    return StreamingResponse(
        iter([output.read()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_template.csv"},
    )


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Run predictions for every row in an uploaded CSV file.

    Required CSV columns (exact names):
        Air temperature [K]
        Process temperature [K]
        Rotational speed [rpm]
        Torque [Nm]
        Tool wear [min]

    Returns a JSON list of predictions, one entry per input row.
    """
    contents = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse the uploaded CSV file: {exc}",
        )

    # Validate required columns are present
    missing_cols = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail={
                "error":            "Missing required columns in CSV.",
                "missing_columns":  missing_cols,
                "required_columns": FEATURE_NAMES,
            },
        )

    if len(df) == 0:
        raise HTTPException(status_code=422, detail="The uploaded CSV file is empty.")

    # Run prediction for each row
    predictions = []
    for idx, row in df[FEATURE_NAMES].iterrows():
        try:
            pred = run_prediction(
                air_temperature     = float(row["Air temperature [K]"]),
                process_temperature = float(row["Process temperature [K]"]),
                rotational_speed    = float(row["Rotational speed [rpm]"]),
                torque              = float(row["Torque [Nm]"]),
                tool_wear           = float(row["Tool wear [min]"]),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Error processing row {idx}: {exc}",
            )

        # Include the input values alongside each prediction
        pred["row"]                 = int(idx)
        pred["air_temperature"]     = float(row["Air temperature [K]"])
        pred["process_temperature"] = float(row["Process temperature [K]"])
        pred["rotational_speed"]    = float(row["Rotational speed [rpm]"])
        pred["torque"]              = float(row["Torque [Nm]"])
        pred["tool_wear"]           = float(row["Tool wear [min]"])
        predictions.append(pred)

    return {
        "total_rows":  len(predictions),
        "predictions": predictions,
    }