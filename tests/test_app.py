import importlib
import json

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
import skops.io as sio
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import data_prep as dp


@pytest.fixture(scope="module")
def model_dir(splits, tmp_path_factory):
    """A small pipeline with the production structure, saved the same way MLflow serves it."""
    pipeline = Pipeline([
        ("preprocess", dp.build_preprocessor(splits["numeric"], splits["categorical"], splits["specialty"])),
        ("model", LogisticRegression(class_weight="balanced", max_iter=500)),
    ]).fit(splits["X_train"].head(5000), splits["y_train"].head(5000))
    path = tmp_path_factory.mktemp("model") / "model"
    mlflow.sklearn.save_model(pipeline, str(path), skops_trusted_types=sio.get_untrusted_types(data=sio.dumps(pipeline)))
    (path / "model_info.json").write_text(json.dumps({"model_name": "test-model", "model_version": "7"}))
    return path


def make_client(monkeypatch, model_uri, log_path):
    monkeypatch.setenv("MODEL_URI", str(model_uri))
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_path))
    import app as app_module
    return TestClient(importlib.reload(app_module).app)


@pytest.fixture
def client(monkeypatch, model_dir, tmp_path):
    with make_client(monkeypatch, model_dir, tmp_path / "predictions.jsonl") as test_client:
        test_client.log_path = tmp_path / "predictions.jsonl"
        yield test_client


def test_health_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["model_version"] == "7"


def test_predict_returns_probability_and_logs(client, raw_encounter):
    response = client.post("/predict", json=raw_encounter)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["readmission_probability"] <= 1.0
    assert body["prediction"] == int(body["readmission_probability"] >= body["threshold"])
    assert body["risk_label"] in {"high risk", "low risk"}

    records = [json.loads(line) for line in client.log_path.read_text().splitlines()]
    assert records[-1]["request_id"] == body["request_id"]
    assert records[-1]["features"]["age"] == raw_encounter["age"]


def test_predict_matches_offline_pipeline_with_null_fields(client, model_dir, raw_encounter):
    payload = {**raw_encounter, "payer_code": None, "medical_specialty": None, "race": None}
    served = client.post("/predict", json=payload).json()["readmission_probability"]

    row = pd.DataFrame([payload]).fillna(np.nan)
    offline = mlflow.sklearn.load_model(str(model_dir)).predict_proba(dp.engineer_features(row))[:, 1][0]
    assert served == pytest.approx(offline, abs=1e-6)


def test_predict_accepts_minimal_payload_with_defaults(client):
    minimal = {
        "gender": "Female", "age": "[60-70)", "admission_type_id": 1, "discharge_disposition_id": 1,
        "admission_source_id": 7, "time_in_hospital": 3, "num_lab_procedures": 40, "num_procedures": 1,
        "num_medications": 12, "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 1,
        "number_diagnoses": 7, "change": "No", "diabetesMed": "Yes",
    }
    assert client.post("/predict", json=minimal).status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        {"age": "75"},                          # not an age band
        {"time_in_hospital": 30},               # out of range
        {"discharge_disposition_id": 11},       # expired: out of scope
        {"insulin": "Sometimes"},               # invalid medication status
        {"diag_1": "not-a-code"},               # invalid ICD-9 code
        {"unexpected_field": 1},                # extra fields forbidden
    ],
)
def test_predict_rejects_invalid_input(client, raw_encounter, change):
    assert client.post("/predict", json={**raw_encounter, **change}).status_code == 422


def test_predict_rejects_missing_required_field(client, raw_encounter):
    payload = {k: v for k, v in raw_encounter.items() if k != "gender"}
    assert client.post("/predict", json=payload).status_code == 422


def test_unavailable_model_returns_503(monkeypatch, tmp_path, raw_encounter):
    with make_client(monkeypatch, tmp_path / "missing-model", tmp_path / "log.jsonl") as test_client:
        assert test_client.get("/health").status_code == 503
        assert test_client.post("/predict", json=raw_encounter).status_code == 503
