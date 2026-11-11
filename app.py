"""FastAPI inference service for 30-day readmission risk.

Serves the MLflow model promoted to the `production` alias. The model is a complete
preprocessing + classifier sklearn Pipeline, so the API only applies the same row-wise
feature engineering as training (data_prep.engineer_features) before scoring.

    GET  /health   -> liveness + which model version is loaded
    POST /predict  -> validated encounter in, readmission probability out

Configuration (environment variables):
    MODEL_URI            local model directory or MLflow URI (default: ./model, exported from the registry)
    MLFLOW_TRACKING_URI  only needed when MODEL_URI is a models:/ or runs:/ URI
    PREDICTION_LOG_PATH  JSON-lines prediction log (default: ./logs/predictions.jsonl)
    DECISION_THRESHOLD   probability threshold for the positive class (default: 0.5)

Run locally:  uvicorn app:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

import data_prep as dp

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_URI = os.getenv("MODEL_URI", str(PROJECT_ROOT / "model"))
PREDICTION_LOG_PATH = Path(os.getenv("PREDICTION_LOG_PATH", PROJECT_ROOT / "logs" / "predictions.jsonl"))
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("readmission_api")

Medication = Literal["No", "Steady", "Up", "Down"]
AgeBand = Literal[
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]
ICD9_PATTERN = r"^[VvEe]?\d{1,3}(\.\d{1,2})?$"


class Encounter(BaseModel):
    """One inpatient encounter at discharge, using the raw dataset's column names and codes."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    race: Optional[Literal["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]] = None
    gender: Literal["Female", "Male"]
    age: AgeBand
    admission_type_id: int = Field(ge=1, le=8)
    discharge_disposition_id: int = Field(ge=1, le=30)
    admission_source_id: int = Field(ge=1, le=26)
    time_in_hospital: int = Field(ge=1, le=14, description="Length of stay in days")
    payer_code: Optional[str] = Field(default=None, max_length=5)
    medical_specialty: Optional[str] = Field(default=None, max_length=60)
    num_lab_procedures: int = Field(ge=0, le=200)
    num_procedures: int = Field(ge=0, le=20)
    num_medications: int = Field(ge=0, le=150)
    number_outpatient: int = Field(ge=0, le=100)
    number_emergency: int = Field(ge=0, le=100)
    number_inpatient: int = Field(ge=0, le=100)
    diag_1: Optional[str] = Field(default=None, pattern=ICD9_PATTERN)
    diag_2: Optional[str] = Field(default=None, pattern=ICD9_PATTERN)
    diag_3: Optional[str] = Field(default=None, pattern=ICD9_PATTERN)
    number_diagnoses: int = Field(ge=1, le=20)
    max_glu_serum: Literal["None", "Norm", ">200", ">300"] = "None"
    A1Cresult: Literal["None", "Norm", ">7", ">8"] = "None"
    metformin: Medication = "No"
    repaglinide: Medication = "No"
    nateglinide: Medication = "No"
    chlorpropamide: Medication = "No"
    glimepiride: Medication = "No"
    acetohexamide: Medication = "No"
    glipizide: Medication = "No"
    glyburide: Medication = "No"
    tolbutamide: Medication = "No"
    pioglitazone: Medication = "No"
    rosiglitazone: Medication = "No"
    acarbose: Medication = "No"
    miglitol: Medication = "No"
    troglitazone: Medication = "No"
    tolazamide: Medication = "No"
    insulin: Medication = "No"
    glyburide_metformin: Medication = Field(default="No", alias="glyburide-metformin")
    glipizide_metformin: Medication = Field(default="No", alias="glipizide-metformin")
    glimepiride_pioglitazone: Medication = Field(default="No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: Medication = Field(default="No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: Medication = Field(default="No", alias="metformin-pioglitazone")
    change: Literal["No", "Ch"]
    diabetesMed: Literal["Yes", "No"]

    @field_validator("discharge_disposition_id")
    @classmethod
    def not_expired_or_hospice(cls, value):
        if value in dp.EXPIRED_HOSPICE_IDS:
            raise ValueError(
                "expired/hospice discharges (11, 13, 14, 19, 20, 21) cannot be readmitted and are out of scope"
            )
        return value


class PredictionResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    readmission_probability: float
    prediction: int
    risk_label: Literal["high risk", "low risk"]
    threshold: float


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_name: str
    model_version: str
    model_uri: str


def load_model_info(model_uri):
    """Registry metadata written next to an exported model (see monitoring.export_production_model)."""
    info_path = Path(model_uri) / "model_info.json"
    if info_path.is_file():
        return json.loads(info_path.read_text(encoding="utf-8"))
    return {"model_name": "unregistered", "model_version": "unknown"}


def get_prediction_logger(path):
    """JSON-lines file logger: one record per prediction, for monitoring and audit."""
    prediction_logger = logging.getLogger("readmission_api.predictions")
    prediction_logger.setLevel(logging.INFO)
    prediction_logger.propagate = False
    path.parent.mkdir(parents=True, exist_ok=True)
    if not any(getattr(h, "baseFilename", None) == str(path.resolve()) for h in prediction_logger.handlers):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        prediction_logger.addHandler(handler)
    return prediction_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_info = {"model_name": "unloaded", "model_version": "unknown"}
    try:
        if os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        app.state.model = mlflow.sklearn.load_model(MODEL_URI)
        app.state.model_info = load_model_info(MODEL_URI)
        logger.info("Loaded model %s from %s", app.state.model_info, MODEL_URI)
    except Exception:
        logger.exception("Could not load model from %s", MODEL_URI)
    app.state.prediction_logger = get_prediction_logger(PREDICTION_LOG_PATH)
    yield


app = FastAPI(
    title="Hospital Readmission Risk API",
    description="Predicts 30-day readmission risk for diabetic inpatients at discharge.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = app.state.model is not None
    body = HealthResponse(
        status="ok" if loaded else "unavailable",
        model_loaded=loaded,
        model_name=str(app.state.model_info.get("model_name")),
        model_version=str(app.state.model_info.get("model_version")),
        model_uri=MODEL_URI,
    )
    if not loaded:
        raise HTTPException(status_code=503, detail=body.model_dump())
    return body


@app.post("/predict", response_model=PredictionResponse)
def predict(encounter: Encounter):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    start = time.perf_counter()
    raw = pd.DataFrame([encounter.model_dump(by_alias=True)])
    features = dp.engineer_features(raw)
    # JSON nulls arrive as None; SimpleImputer only recognises NaN, so normalise to match training data.
    for column in features.select_dtypes(include="object").columns:
        features[column] = features[column].where(features[column].notna(), np.nan)
    probability = float(app.state.model.predict_proba(features)[:, 1][0])
    prediction = int(probability >= DECISION_THRESHOLD)

    response = PredictionResponse(
        request_id=str(uuid.uuid4()),
        model_name=str(app.state.model_info.get("model_name")),
        model_version=str(app.state.model_info.get("model_version")),
        readmission_probability=round(probability, 6),
        prediction=prediction,
        risk_label="high risk" if prediction else "low risk",
        threshold=DECISION_THRESHOLD,
    )
    app.state.prediction_logger.info(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **response.model_dump(),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "features": encounter.model_dump(by_alias=True),
            }
        )
    )
    return response
