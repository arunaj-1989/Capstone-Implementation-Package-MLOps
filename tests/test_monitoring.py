import numpy as np

import monitoring as mon
import retrain


def test_psi_is_near_zero_for_identical_distributions():
    scores = np.random.default_rng(0).beta(2, 3, 5000)
    psi, table = mon.population_stability_index(scores, scores)
    assert psi < 1e-9
    assert len(table) == mon.PSI_BINS


def test_psi_detects_a_large_shift():
    rng = np.random.default_rng(0)
    psi, _ = mon.population_stability_index(rng.beta(2, 5, 5000), rng.beta(5, 2, 5000))
    assert psi > 0.2
    assert mon.interpret_psi(psi) == "significant shift"


def summary(psi, share, dataset_drift=False):
    return {"prediction_drift": {"prediction_psi": psi},
            "feature_drift": {"drift_share": share, "dataset_drift": dataset_drift}}


def test_no_trigger_when_all_signals_are_below_threshold():
    result = retrain.evaluate_triggers(summary(0.15, 0.25))
    assert result["should_retrain"] is False
    assert result["fired_signals"] == []


def test_each_signal_fires_on_its_own():
    assert retrain.evaluate_triggers(summary(0.25, 0.1))["fired_signals"] == ["prediction_psi"]
    assert retrain.evaluate_triggers(summary(0.05, 0.35))["fired_signals"] == ["drift_share"]
    assert retrain.evaluate_triggers(summary(0.05, 0.1, True))["fired_signals"] == ["dataset_drift"]
