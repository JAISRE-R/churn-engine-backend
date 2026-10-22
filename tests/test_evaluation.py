"""
churn-engine/tests/test_evaluation.py
=======================================
Unit tests for src/evaluation.py.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    compute_classification_metrics,
    evaluate_thresholds,
    log_experiment,
    save_decision_policy,
    save_sample_predictions,
)
from src.utils import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def dummy_labels_and_probs():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.35, 0.6, 0.4, 0.7, 0.85, 0.9])
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


class TestEvaluation:

    def test_compute_classification_metrics_includes_mcc(self, dummy_labels_and_probs):
        y_true, y_pred, y_prob = dummy_labels_and_probs
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)

        required = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "brier_score"]
        for key in required:
            assert key in metrics, f"Missing metric key: {key}"

        assert isinstance(metrics["mcc"], float)

    def test_evaluate_thresholds(self, dummy_labels_and_probs):
        y_true, _, y_prob = dummy_labels_and_probs
        thresh_df, best_thresh, best_policy = evaluate_thresholds(y_true, y_prob)

        assert len(thresh_df) == 6
        assert "fpr" in thresh_df.columns
        assert "fnr" in thresh_df.columns
        assert "selected_threshold" in best_policy
        assert "business_justification" in best_policy

    def test_save_decision_policy(self, tmp_path):
        policy = {
            "selected_threshold": 0.55,
            "selection_criteria": "Maximize F1",
            "fpr": 0.05,
            "fnr": 0.10,
            "business_justification": "Optimal trade-off",
        }
        target = tmp_path / "decision_policy.json"
        save_decision_policy(policy, target)

        assert target.exists()
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["selected_threshold"] == 0.55

    def test_save_sample_predictions(self, tmp_path, dummy_labels_and_probs):
        y_true, y_pred, y_prob = dummy_labels_and_probs
        cust_ids = [f"C-100{i}" for i in range(len(y_true))]
        target = tmp_path / "sample_predictions.csv"

        df_preds = save_sample_predictions(cust_ids, y_true, y_prob, y_pred, target)
        assert target.exists()
        assert len(df_preds) == len(y_true)
        assert "actual_label" in df_preds.columns
        assert "predicted_probability" in df_preds.columns
