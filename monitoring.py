"""Drift detection, performance monitoring, logging, Evidently reports, and governance monitoring.

    export_production_model -> copy the model behind the `production` alias to ./model for serving
    build_reference_sample  -> fixed sample of training features (reference_sample.csv)
    simulate_current_batch  -> a shifted "production" batch (longer stays, more medications,
                               older patients, more circulatory diagnoses)
    feature_drift           -> Evidently DataDriftPreset report (drift_report.html)
    population_stability_index / prediction_drift -> PSI on model scores
    run_monitoring          -> both checks + machine-readable drift_summary.json
    registry_history / pinned_dependencies -> governance evidence

Run end to end:  python monitoring.py
"""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

import data_prep as dp

PROJECT_ROOT = Path(__file__).resolve().parent
MLFLOW_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
REGISTERED_MODEL_NAME = "hospital-readmission-30d-classifier"
PRODUCTION_ALIAS = "production"

MODEL_EXPORT_DIR = PROJECT_ROOT / "model"
MONITORING_DATA_DIR = PROJECT_ROOT / "data" / "monitoring"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
REFERENCE_SAMPLE_PATH = MONITORING_DATA_DIR / "reference_sample.csv"
CURRENT_BATCH_PATH = MONITORING_DATA_DIR / "current_batch.csv"
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.html"
DRIFT_SUMMARY_PATH = REPORTS_DIR / "drift_summary.json"
MONITORING_LOG_PATH = LOGS_DIR / "monitoring.log"

REFERENCE_SAMPLE_SIZE = 5000
CURRENT_BATCH_SIZE = 5000
# Evidently DataDriftPreset default: the dataset is drifted when >= 50% of columns drift.
DATASET_DRIFT_SHARE = 0.5
PSI_BINS = 10


def get_logger(name="monitoring", path=MONITORING_LOG_PATH):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        path.parent.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = get_logger()


def configure_mlflow(tracking_uri=MLFLOW_TRACKING_URI):
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def production_version(client=None, name=REGISTERED_MODEL_NAME, alias=PRODUCTION_ALIAS):
    client = client or configure_mlflow()
    return client.get_model_version_by_alias(name, alias)


def load_production_model(name=REGISTERED_MODEL_NAME, alias=PRODUCTION_ALIAS):
    configure_mlflow()
    return mlflow.sklearn.load_model(f"models:/{name}@{alias}")


def export_production_model(dst=MODEL_EXPORT_DIR, name=REGISTERED_MODEL_NAME, alias=PRODUCTION_ALIAS):
    """Copy the model behind the alias to a plain folder that the API/Docker image can load
    without access to the MLflow store, plus a model_info.json with its registry identity."""
    client = configure_mlflow()
    version = client.get_model_version_by_alias(name, alias)
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{version.version}", dst_path=str(dst))
    info = {
        "model_name": name,
        "model_version": str(version.version),
        "alias": alias,
        "run_id": version.run_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    (dst / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    logger.info("Exported %s v%s (@%s) to %s", name, version.version, alias, dst)
    return info


def build_reference_sample(X_train, n=REFERENCE_SAMPLE_SIZE, path=REFERENCE_SAMPLE_PATH,
                           random_state=dp.RANDOM_STATE):
    """Fixed baseline: a sample of the training features that every future batch is compared against."""
    reference = X_train.sample(n=min(n, len(X_train)), random_state=random_state).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(path, index=False)
    logger.info("Saved reference sample %s to %s", reference.shape, path)
    return reference


def simulate_current_batch(X_pool, y_pool=None, n=CURRENT_BATCH_SIZE, strength=0.8, path=CURRENT_BATCH_PATH,
                           random_state=dp.RANDOM_STATE):
    """Stand-in for a live batch: resample unseen encounters, favouring longer stays, more medications,
    older patients and circulatory primary diagnoses. `strength` = 0 gives an unshifted sample.

    Resampling real rows (rather than editing values) keeps every encounter clinically coherent, so
    correlated features shift too, as they would in a sicker population."""
    rng = np.random.default_rng(random_state)
    pool = X_pool.reset_index(drop=True)

    def standardise(s):
        return (s - s.mean()) / s.std()

    severity = (
        standardise(pool["time_in_hospital"])
        + standardise(pool["num_medications"])
        + standardise(pool["age_midpoint"])
        + pool["diag_1_group"].eq("Circulatory").astype(float)
    )
    weights = np.exp(strength * severity)
    index = rng.choice(len(pool), size=n, replace=True, p=weights / weights.sum())
    current = pool.iloc[index].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    current.to_csv(path, index=False)
    logger.info("Simulated current batch %s (strength=%s) saved to %s", current.shape, strength, path)
    if y_pool is None:
        return current
    return current, y_pool.reset_index(drop=True).iloc[index].reset_index(drop=True)


def feature_drift(reference, current, html_path=DRIFT_REPORT_PATH, drift_share=DATASET_DRIFT_SHARE):
    """Evidently DataDriftPreset: per-column drift tests plus the share of drifted columns."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    snapshot = Report([DataDriftPreset(drift_share=drift_share)]).run(
        current_data=current, reference_data=reference
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))
    metrics = snapshot.dict()["metrics"]

    summary = next(m for m in metrics if m["metric_name"].startswith("DriftedColumnsCount"))
    rows = []
    for m in metrics:
        if not m["metric_name"].startswith("ValueDrift"):
            continue
        config, value = m["config"], float(m["value"])
        is_p_value = "p_value" in config["method"]
        rows.append(
            {
                "column": config["column"],
                "method": config["method"],
                "threshold": config["threshold"],
                "score": value,
                "drifted": value < config["threshold"] if is_p_value else value >= config["threshold"],
            }
        )
    columns = pd.DataFrame(rows).sort_values(["drifted", "score"], ascending=[False, False]).reset_index(drop=True)
    share = float(summary["value"]["share"])
    result = {
        "n_columns": len(columns),
        "n_drifted_columns": int(summary["value"]["count"]),
        "drift_share": share,
        "dataset_drift": share >= drift_share,
        "dataset_drift_rule": f"drifted-column share >= {drift_share}",
        "drifted_columns": columns.loc[columns["drifted"], "column"].tolist(),
        "report_path": str(html_path),
    }
    logger.info("Feature drift: %d/%d columns drifted (share %.3f), dataset drift=%s",
                result["n_drifted_columns"], result["n_columns"], share, result["dataset_drift"])
    return result, columns


def population_stability_index(reference_scores, current_scores, bins=PSI_BINS, eps=1e-4):
    """PSI = sum((cur% - ref%) * ln(cur% / ref%)) over quantile bins of the reference scores.
    < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant shift."""
    reference_scores, current_scores = np.asarray(reference_scores), np.asarray(current_scores)
    edges = np.unique(np.quantile(reference_scores, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference_scores, edges)[0] / len(reference_scores)
    cur_pct = np.histogram(current_scores, edges)[0] / len(current_scores)
    ref_pct, cur_pct = np.clip(ref_pct, eps, None), np.clip(cur_pct, eps, None)
    contributions = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    table = pd.DataFrame(
        {
            "bin_lower": edges[:-1],
            "bin_upper": edges[1:],
            "reference_pct": ref_pct,
            "current_pct": cur_pct,
            "psi_contribution": contributions,
        }
    )
    return float(contributions.sum()), table


def interpret_psi(value):
    if value < 0.1:
        return "stable"
    if value <= 0.2:
        return "moderate shift"
    return "significant shift"


def prediction_drift(model, reference, current, threshold=0.5):
    reference_scores = model.predict_proba(reference)[:, 1]
    current_scores = model.predict_proba(current)[:, 1]
    psi, table = population_stability_index(reference_scores, current_scores)
    result = {
        "prediction_psi": psi,
        "psi_interpretation": interpret_psi(psi),
        "reference_mean_score": float(reference_scores.mean()),
        "current_mean_score": float(current_scores.mean()),
        "reference_flag_rate": float((reference_scores >= threshold).mean()),
        "current_flag_rate": float((current_scores >= threshold).mean()),
    }
    logger.info("Prediction drift: PSI %.3f (%s)", psi, result["psi_interpretation"])
    return result, table, reference_scores, current_scores


def write_drift_summary(features, predictions, reference_rows, current_rows, model_version="unknown",
                        summary_path=DRIFT_SUMMARY_PATH):
    """drift_summary.json: the single input the retraining trigger reads."""
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": str(model_version),
        "reference_rows": reference_rows,
        "current_rows": current_rows,
        "feature_drift": features,
        "prediction_drift": predictions,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Drift summary written to %s", summary_path)
    return summary


def run_monitoring(model, reference, current, model_version="unknown", summary_path=DRIFT_SUMMARY_PATH):
    """Feature + prediction drift for one batch; writes drift_summary.json."""
    features, column_table = feature_drift(reference, current)
    predictions, psi_table, reference_scores, current_scores = prediction_drift(model, reference, current)
    summary = write_drift_summary(features, predictions, len(reference), len(current), model_version, summary_path)
    return {
        "summary": summary,
        "column_drift": column_table,
        "psi_table": psi_table,
        "reference_scores": reference_scores,
        "current_scores": current_scores,
    }


def registry_history(client=None, name=REGISTERED_MODEL_NAME):
    """Every registered version with its aliases, tags and source run."""
    client = client or configure_mlflow()
    rows = []
    for summary in client.search_model_versions(f"name = '{name}'"):
        v = client.get_model_version(name, summary.version)  # search results omit aliases
        rows.append(
            {
                "version": int(v.version),
                "aliases": ", ".join(v.aliases) or "-",
                "model_family": v.tags.get("model_family"),
                "trained_by": v.tags.get("trained_by", "stage3_notebook"),
                "val_roc_auc": v.tags.get("val_roc_auc"),
                "test_roc_auc": v.tags.get("test_roc_auc"),
                "status": v.status,
                "run_id": v.run_id,
                "created": pd.to_datetime(v.creation_timestamp, unit="ms").strftime("%Y-%m-%d %H:%M:%S"),
                "description": v.description,
            }
        )
    return pd.DataFrame(rows).sort_values("version", ascending=False).set_index("version")


def pinned_dependencies(requirements_path=PROJECT_ROOT / "requirements.txt"):
    """Compare the pinned versions in requirements.txt with what is installed."""
    rows = []
    for line in Path(requirements_path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        name, _, pinned = line.partition("==")
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        rows.append({"package": name, "pinned": pinned or "(unpinned)", "installed": installed,
                     "matches": bool(pinned) and pinned == installed})
    return pd.DataFrame(rows).set_index("package")


if __name__ == "__main__":
    splits = dp.prepare_splits()
    reference = build_reference_sample(splits["X_train"])
    current = simulate_current_batch(splits["X_test"])
    version = production_version()
    results = run_monitoring(load_production_model(), reference, current, model_version=version.version)
    print(json.dumps(results["summary"], indent=2))
