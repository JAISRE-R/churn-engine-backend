"""
churn-engine/tests/test_train_model.py
========================================
Unit tests for src/train_model.py.

NOTE: All file-writing tests use pytest's tmp_path fixture so that no
      production artefacts under models/ or reports/ are overwritten.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.train_model import ChurnModelPipeline, DataSplitter, ModelSelectionEngine, TrainingPipeline
from src.utils import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def isolated_config(config, tmp_path):
    """Config variant that redirects all file output to tmp_path."""
    cfg = {k: v for k, v in config.items()}
    cfg["paths"] = {k: str(tmp_path / k) for k in config["paths"]}
    return cfg


@pytest.fixture
def dummy_features():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({
        "recency_days":         np.random.uniform(1, 90, n),
        "frequency":            np.random.randint(1, 20, n).astype(float),
        "monetary_total":       np.random.uniform(10, 500, n),
        "avg_order_value":      np.random.uniform(10, 100, n),
        "support_ticket_count": np.random.randint(0, 5, n).astype(float),
    })
    y = pd.Series(np.random.choice([0, 1], size=n, p=[0.6, 0.4]), name="churn")
    return X, y


# ---------------------------------------------------------------------------
# DataSplitter
# ---------------------------------------------------------------------------

class TestDataSplitter:

    def test_split_ratios_and_stratification(self, config, dummy_features):
        X, y = dummy_features
        splitter = DataSplitter(config)
        X_tr, X_v, X_te, y_tr, y_v, y_te = splitter.split(X, y)

        assert len(X_tr) == 70
        assert len(X_v) == 15
        assert len(X_te) == 15
        assert set(y_tr.unique()) == {0, 1}


# ---------------------------------------------------------------------------
# ModelSelectionEngine (does not write to disk)
# ---------------------------------------------------------------------------

class TestModelSelectionEngine:

    def test_benchmark_and_select(self, isolated_config, dummy_features):
        X, y = dummy_features
        splitter = DataSplitter(isolated_config)
        X_tr, X_v, X_te, y_tr, y_v, y_te = splitter.split(X, y)

        engine = ModelSelectionEngine(isolated_config)
        pipeline, winner_name, df_bench = engine.benchmark_and_select(X_tr, y_tr, X_v, y_v)

        assert isinstance(pipeline, ChurnModelPipeline)
        assert winner_name in [
            "Logistic Regression (Baseline)", "Random Forest", "XGBoost (Tuned)"
        ]
        assert len(df_bench) == 3
        assert "roc_auc" in df_bench.columns

    def test_pipeline_inference_api(self, isolated_config, dummy_features):
        X, y = dummy_features
        splitter = DataSplitter(isolated_config)
        X_tr, X_v, X_te, y_tr, y_v, y_te = splitter.split(X, y)

        engine = ModelSelectionEngine(isolated_config)
        pipeline, _, _ = engine.benchmark_and_select(X_tr, y_tr, X_v, y_v)

        probs = pipeline.predict_proba(X_te)
        assert len(probs) == len(X_te)
        assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0

        preds = pipeline.predict(X_te, threshold=0.5)
        assert len(preds) == len(X_te)
        assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# TrainingPipeline (writes to isolated tmp_path)
# ---------------------------------------------------------------------------

class TestTrainingPipeline:

    def test_pipeline_run(self, isolated_config, dummy_features):
        X, y = dummy_features
        res = TrainingPipeline(isolated_config).run(X, y)

        assert "winning_model" in res
        assert "optimal_threshold" in res
        assert "metrics" in res
        assert "mcc" in res["metrics"]
        assert "roc_auc" in res["metrics"]
