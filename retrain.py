"""Automated retraining workflow, model update, and registry integration.

1. Read the latest drift summary (monitoring.run_monitoring -> reports/drift_summary.json).
2. Multi-signal trigger: retrain if prediction PSI > 0.20 OR drifted-feature share > 0.30
   OR Evidently flags dataset drift (never on PSI alone).
3. If triggered: clone the production pipeline (same preprocessing + hyperparameters), refit it on
   all labelled development data (train + validation), recompute scale_pos_weight, and log the run.
4. Register the candidate as a new version. Promote it to the `production` alias only if its
   held-out test ROC-AUC is no worse than the current champion's (minus a small tolerance).
5. Append a decision record (signals, metrics, versions, decision) to logs/retraining_decisions.jsonl.

Run:  python retrain.py            (respects the trigger)
      python retrain.py --force    (retrain even if no signal fires)
"""

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow
import mlflow.sklearn
import skops.io as sio
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import data_prep as dp
import monitoring as mon

EXPERIMENT_NAME = "hospital-readmission-30d"
TRIGGER_THRESHOLDS = {"prediction_psi": 0.20, "drift_share": 0.30}
PROMOTION_TOLERANCE = 0.005
DECISION_THRESHOLD = 0.5
DECISION_LOG_PATH = mon.LOGS_DIR / "retraining_decisions.jsonl"

logger = mon.get_logger("retraining")


def evaluate_triggers(drift_summary):
    """The three retraining signals and whether any of them fired."""
    psi = drift_summary["prediction_drift"]["prediction_psi"]
    share = drift_summary["feature_drift"]["drift_share"]
    dataset_drift = bool(drift_summary["feature_drift"]["dataset_drift"])
    signals = {
        "prediction_psi": {"value": round(psi, 4), "threshold": f"> {TRIGGER_THRESHOLDS['prediction_psi']}",
                           "fired": psi > TRIGGER_THRESHOLDS["prediction_psi"]},
        "drift_share": {"value": round(share, 4), "threshold": f"> {TRIGGER_THRESHOLDS['drift_share']}",
                        "fired": share > TRIGGER_THRESHOLDS["drift_share"]},
        "dataset_drift": {"value": dataset_drift, "threshold": "== True", "fired": dataset_drift},
    }
    fired = [name for name, s in signals.items() if s["fired"]]
    return {"signals": signals, "fired_signals": fired, "should_retrain": bool(fired)}


def classification_metrics(model, X, y, threshold=DECISION_THRESHOLD):
    scores = model.predict_proba(X)[:, 1]
    predictions = (scores >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, scores),
        "pr_auc": average_precision_score(y, scores),
        "recall": recall_score(y, predictions, zero_division=0),
        "precision": precision_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "accuracy": accuracy_score(y, predictions),
    }


def write_decision(record, path=DECISION_LOG_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    logger.info("Decision record %s: %s", record["decision_id"], record["decision"])
    return record


def read_decisions(path=DECISION_LOG_PATH):
    if not Path(path).is_file():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line])


def run_retraining(drift_summary, force=False, export_on_promotion=True):
    client = mon.configure_mlflow()
    trigger = evaluate_triggers(drift_summary)
    champion_version = mon.production_version(client)
    record = {
        "decision_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drift_summary_generated_at": drift_summary.get("generated_at"),
        "trigger": trigger,
        "forced": force,
        "champion_version": str(champion_version.version),
    }

    if not (trigger["should_retrain"] or force):
        record.update(decision="no_retrain", reason="No retraining signal fired.")
        return write_decision(record)

    logger.info("Retraining triggered by %s (forced=%s)", trigger["fired_signals"], force)
    splits = dp.prepare_splits()
    X_dev = pd.concat([splits["X_train"], splits["X_val"]], ignore_index=True)
    y_dev = pd.concat([splits["y_train"], splits["y_val"]], ignore_index=True)
    X_test, y_test = splits["X_test"], splits["y_test"]

    champion = mon.load_production_model()
    candidate = clone(champion)
    params = candidate.named_steps["model"].get_params()
    if "scale_pos_weight" in params:
        candidate.set_params(model__scale_pos_weight=round(float((y_dev == 0).sum() / (y_dev == 1).sum()), 4))
    candidate.fit(X_dev, y_dev)

    champion_metrics = classification_metrics(champion, X_test, y_test)
    candidate_metrics = classification_metrics(candidate, X_test, y_test)
    promote = candidate_metrics["roc_auc"] >= champion_metrics["roc_auc"] - PROMOTION_TOLERANCE
    family = champion_version.tags.get("model_family") or type(candidate.named_steps["model"]).__name__.lower()

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"retrain_{family}") as run:
        mlflow.set_tags({
            "stage": "stage4_retraining",
            "model_family": family,
            "trigger_signals": ",".join(trigger["fired_signals"]) or "forced",
            "champion_version": str(champion_version.version),
            "decision_id": record["decision_id"],
        })
        mlflow.log_params({f"model__{k}": v for k, v in candidate.named_steps["model"].get_params().items()
                           if v is not None and not callable(v)})
        mlflow.log_params({"n_train": len(X_dev), "training_data": "train+validation",
                           "decision_threshold": DECISION_THRESHOLD,
                           "promotion_rule": f"test_roc_auc >= champion - {PROMOTION_TOLERANCE}"})
        mlflow.log_metrics({f"test_{k}": v for k, v in candidate_metrics.items()})
        mlflow.log_metrics({f"champion_test_{k}": v for k, v in champion_metrics.items()})
        mlflow.log_metrics({f"trigger_{k}": float(s["value"]) for k, s in trigger["signals"].items()})
        mlflow.log_dict(drift_summary, "monitoring/drift_summary.json")
        signature_input = X_dev.head(200)
        model_info = mlflow.sklearn.log_model(
            candidate,
            name="model",
            signature=infer_signature(signature_input, candidate.predict_proba(signature_input)[:, 1]),
            input_example=X_dev.head(3),
            skops_trusted_types=sio.get_untrusted_types(data=sio.dumps(candidate)),
        )

    new_version = mlflow.register_model(model_info.model_uri, mon.REGISTERED_MODEL_NAME)
    tags = {
        "model_family": family,
        "trained_by": "retrain.py",
        "test_roc_auc": f"{candidate_metrics['roc_auc']:.4f}",
        "test_recall": f"{candidate_metrics['recall']:.4f}",
        "trigger_signals": ",".join(trigger["fired_signals"]) or "forced",
        "decision_id": record["decision_id"],
    }
    for key, value in tags.items():
        client.set_model_version_tag(mon.REGISTERED_MODEL_NAME, new_version.version, key, value)

    if promote:
        client.set_registered_model_alias(mon.REGISTERED_MODEL_NAME, mon.PRODUCTION_ALIAS, new_version.version)
        decision = "promoted"
        reason = (f"Candidate test ROC-AUC {candidate_metrics['roc_auc']:.4f} >= champion "
                  f"{champion_metrics['roc_auc']:.4f} - {PROMOTION_TOLERANCE}; alias moved to v{new_version.version}.")
    else:
        decision = "registered_not_promoted"
        reason = (f"Candidate test ROC-AUC {candidate_metrics['roc_auc']:.4f} < champion "
                  f"{champion_metrics['roc_auc']:.4f} - {PROMOTION_TOLERANCE}; champion v{champion_version.version} kept.")
    client.update_model_version(mon.REGISTERED_MODEL_NAME, new_version.version,
                                description=f"Retrained by retrain.py ({decision}). {reason}")

    record.update(
        decision=decision,
        reason=reason,
        candidate_version=str(new_version.version),
        candidate_run_id=run.info.run_id,
        production_version_after=str(mon.production_version(client).version),
        champion_test_metrics={k: round(v, 4) for k, v in champion_metrics.items()},
        candidate_test_metrics={k: round(v, 4) for k, v in candidate_metrics.items()},
        training_rows=len(X_dev),
    )
    if promote and export_on_promotion:
        record["exported_model"] = mon.export_production_model()
    return write_decision(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, default=mon.DRIFT_SUMMARY_PATH)
    parser.add_argument("--force", action="store_true", help="Retrain even if no signal fires")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    print(json.dumps(run_retraining(summary, force=args.force), indent=2, default=str))
