"""
churn-engine/src/train_model.py
=================================
Multi-Model Benchmark, Selection Engine, and Model Training Pipeline.

Responsibilities
----------------
- Perform stratified Train / Validation / Test split (70% / 15% / 15%).
- Benchmark three candidate models:
    1. Logistic Regression (Baseline)
    2. Random Forest Classifier
    3. XGBoost Classifier (with lightweight RandomizedSearchCV tuning)
- Select the best model automatically based on ROC-AUC (secondary: F1-score).
- Fit `StandardScaler` on X_train ONLY (preventing data leakage) and package inside `ChurnModelPipeline`.
- Expose a clean, self-contained inference API (`predict` and `predict_proba`) for Phase 4 consumption.
- Save serialised model artefact to `models/churn_model.joblib`.
- Save `models/model_metadata.json` and `models/decision_policy.json`.
- Log experiment entry to `reports/experiment_history.csv` and export `reports/sample_predictions.csv`.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.evaluation import (
    evaluate_thresholds,
    generate_evaluation_report,
    log_experiment,
    save_decision_policy,
    save_sample_predictions,
)
from src.utils import ensure_dir, get_project_root, save_model

logger = logging.getLogger(__name__)


# ===========================================================================
# Churn Model Pipeline Container (Inference Wrapper)
# ===========================================================================

class ChurnModelPipeline:
    """Self-contained container wrapping fitted StandardScaler and ML Estimator.

    Exposes clean `predict_proba()` and `predict()` methods that downstream
    components (e.g. Phase 4 Predictor / Decision Gate) can invoke directly.

    Parameters
    ----------
    scaler : StandardScaler
        Fitted scaler trained on X_train.
    model : Any
        Fitted classifier estimator (XGBoost / Random Forest / Logistic Regression).
    feature_names : list[str]
        List of feature column names expected by the model.
    optimal_threshold : float
        Optimal decision threshold selected during evaluation.
    """

    def __init__(
        self,
        scaler: StandardScaler,
        model: Any,
        feature_names: list[str],
        optimal_threshold: float = 0.50,
    ) -> None:
        self.scaler = scaler
        self.model = model
        self.feature_names = list(feature_names)
        self.optimal_threshold = float(optimal_threshold)

    def _prepare_features(self, X: pd.DataFrame | np.ndarray | dict[str, Any]) -> np.ndarray:
        """Align columns and apply scaler to raw feature input."""
        if isinstance(X, dict):
            X_df = pd.DataFrame([X])
        elif isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X.copy()

        # Reindex to ensure feature alignment
        X_aligned = X_df.reindex(columns=self.feature_names, fill_value=0.0)
        return self.scaler.transform(X_aligned)

    def predict_proba(self, X: pd.DataFrame | np.ndarray | dict[str, Any]) -> np.ndarray:
        """Return 1D array of churn probabilities in [0.0, 1.0]."""
        X_scaled = self._prepare_features(X)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)
            return probs[:, 1] if probs.ndim == 2 else probs
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X_scaled)
            return 1.0 / (1.0 + np.exp(-scores))
        else:
            raise AttributeError("Model does not support probability estimation.")

    def predict(
        self,
        X: pd.DataFrame | np.ndarray | dict[str, Any],
        threshold: float | None = None,
    ) -> np.ndarray:
        """Return 1D array of binary churn predictions (0 or 1)."""
        cutoff = self.optimal_threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        return (probs >= cutoff).astype(int)


# ===========================================================================
# Data Splitter
# ===========================================================================

class DataSplitter:
    """Split customer features matrix into train, validation, and test subsets.

    Parameters
    ----------
    config : dict[str, Any]
        Full project configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        data_cfg = config.get("data", {})
        self.train_split: float = data_cfg.get("train_split", 0.70)
        self.val_split: float = data_cfg.get("val_split", 0.15)
        self.test_split: float = data_cfg.get("test_split", 0.15)
        self.random_seed: int = data_cfg.get("random_seed", 42)

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Perform stratified 70/15/15 split."""
        val_test_ratio = self.val_split + self.test_split
        test_size_relative = self.test_split / val_test_ratio

        # First split: Train vs (Val + Test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=val_test_ratio,
            random_state=self.random_seed,
            stratify=y,
        )

        # Second split: Val vs Test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_size_relative,
            random_state=self.random_seed,
            stratify=y_temp,
        )

        logger.info(
            "Stratified Split Complete | Train: %d, Val: %d, Test: %d | Features: %d",
            len(X_train), len(X_val), len(X_test), X_train.shape[1]
        )
        return X_train, X_val, X_test, y_train, y_val, y_test


# ===========================================================================
# Model Selection Engine
# ===========================================================================

class ModelSelectionEngine:
    """Benchmark candidate models, perform hyperparameter tuning, and select winner.

    Parameters
    ----------
    config : dict[str, Any]
        Full project configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.root: Path = get_project_root()
        self.models_dir: Path = self.root / config["paths"]["models"]
        ensure_dir(self.models_dir)

    def benchmark_and_select(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[ChurnModelPipeline, str, pd.DataFrame]:
        """Train and evaluate 3 candidate models, returning the best pipeline.

        Returns
        -------
        tuple[ChurnModelPipeline, str, pd.DataFrame]
            (Winning ChurnModelPipeline, winning_model_name, benchmark_results_df)
        """
        logger.info("Starting Multi-Model Benchmark (Logistic Regression, Random Forest, XGBoost)...")
        feature_names = list(X_train.columns)

        # Fit StandardScaler on X_train only (No data leakage!)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_v_s = scaler.transform(X_val)

        results = []

        # -------------------------------------------------------------------
        # 1. Logistic Regression (Baseline)
        # -------------------------------------------------------------------
        t0 = time.time()
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_tr_s, y_train)
        t_lr = time.time() - t0
        lr_probs = lr.predict_proba(X_v_s)[:, 1]
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        lr_roc = float(roc_auc_score(y_val, lr_probs))
        lr_preds = (lr_probs >= 0.5).astype(int)

        results.append({
            "model_name": "Logistic Regression (Baseline)",
            "estimator": lr,
            "roc_auc": round(lr_roc, 4),
            "f1": round(float(f1_score(y_val, lr_preds, zero_division=0)), 4),
            "precision": round(float(precision_score(y_val, lr_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_val, lr_preds, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y_val, lr_preds)), 4),
            "training_time_sec": round(t_lr, 3),
            "best_params": {},
        })

        # -------------------------------------------------------------------
        # 2. Random Forest Classifier
        # -------------------------------------------------------------------
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        rf.fit(X_tr_s, y_train)
        t_rf = time.time() - t0
        rf_probs = rf.predict_proba(X_v_s)[:, 1]
        rf_roc = float(roc_auc_score(y_val, rf_probs))
        rf_preds = (rf_probs >= 0.5).astype(int)

        results.append({
            "model_name": "Random Forest",
            "estimator": rf,
            "roc_auc": round(rf_roc, 4),
            "f1": round(float(f1_score(y_val, rf_preds, zero_division=0)), 4),
            "precision": round(float(precision_score(y_val, rf_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_val, rf_preds, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y_val, rf_preds)), 4),
            "training_time_sec": round(t_rf, 3),
            "best_params": {"n_estimators": 100, "max_depth": 6},
        })

        # -------------------------------------------------------------------
        # 3. XGBoost Classifier (with Lightweight RandomizedSearchCV)
        # -------------------------------------------------------------------
        t0 = time.time()
        param_dist = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 150],
            "subsample": [0.7, 0.8, 1.0],
        }
        xgb_base = XGBClassifier(eval_metric="logloss", random_state=42)
        rsearch = RandomizedSearchCV(
            xgb_base,
            param_distributions=param_dist,
            n_iter=5,
            scoring="roc_auc",
            cv=3,
            random_state=42,
            n_jobs=1,
        )
        rsearch.fit(X_tr_s, y_train)
        best_xgb = rsearch.best_estimator_
        t_xgb = time.time() - t0

        xgb_probs = best_xgb.predict_proba(X_v_s)[:, 1]
        xgb_roc = float(roc_auc_score(y_val, xgb_probs))
        xgb_preds = (xgb_probs >= 0.5).astype(int)

        results.append({
            "model_name": "XGBoost (Tuned)",
            "estimator": best_xgb,
            "roc_auc": round(xgb_roc, 4),
            "f1": round(float(f1_score(y_val, xgb_preds, zero_division=0)), 4),
            "precision": round(float(precision_score(y_val, xgb_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_val, xgb_preds, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y_val, xgb_preds)), 4),
            "training_time_sec": round(t_xgb, 3),
            "best_params": rsearch.best_params_,
        })

        # Convert to benchmark Summary DataFrame
        df_bench = pd.DataFrame(results).drop(columns=["estimator"])

        # Select Winner (Primary: ROC-AUC, Secondary: F1)
        winner_dict = sorted(results, key=lambda x: (x["roc_auc"], x["f1"]), reverse=True)[0]
        winner_name = winner_dict["model_name"]
        winner_estimator = winner_dict["estimator"]

        logger.info("Model Selection Winner: %s | Val ROC-AUC: %.4f", winner_name, winner_dict["roc_auc"])

        # Package into ChurnModelPipeline
        winning_pipeline = ChurnModelPipeline(
            scaler=scaler,
            model=winner_estimator,
            feature_names=feature_names,
            optimal_threshold=0.50,
        )

        return winning_pipeline, winner_name, df_bench

    def save(
        self,
        model_pipeline: ChurnModelPipeline,
        winner_name: str,
        metrics: dict[str, float],
        best_params: dict[str, Any],
        dataset_shape: tuple[int, int],
    ) -> None:
        """Serialise model artefact to models/churn_model.joblib & metadata to models/model_metadata.json."""
        model_path = self.models_dir / "churn_model.joblib"
        joblib.dump(model_pipeline, model_path)
        logger.info("Saved serialized model pipeline → %s", model_path)

        metadata = {
            "model_name": winner_name,
            "training_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_rows": dataset_shape[0],
            "num_features": dataset_shape[1],
            "selected_threshold": model_pipeline.optimal_threshold,
            "hyperparameters": best_params,
            "test_metrics": metrics,
            "feature_names": model_pipeline.feature_names,
            "model_version": "1.0.0-phase3",
        }

        meta_path = self.models_dir / "model_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved model metadata → %s", meta_path)


# ===========================================================================
# Training Pipeline Orchestrator
# ===========================================================================

class TrainingPipeline:
    """Orchestrates end-to-end model training, evaluation, and artefact generation."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.root: Path = get_project_root()
        self.splitter = DataSplitter(config)
        self.selection_engine = ModelSelectionEngine(config)

    def run(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Execute Phase 3 pipeline."""
        t_start = time.time()
        logger.info("Executing Phase 3 Model Training Pipeline ...")

        # 1. Stratified Split
        X_tr, X_v, X_te, y_tr, y_v, y_te = self.splitter.split(X, y)

        # 2. Benchmark & Model Selection
        pipeline, winner_name, df_bench = self.selection_engine.benchmark_and_select(X_tr, y_tr, X_v, y_v)

        # 3. Test Set Prediction & Threshold Optimization
        test_probs = pipeline.predict_proba(X_te)
        thresh_df, best_thresh, best_policy = evaluate_thresholds(y_te, test_probs)
        pipeline.optimal_threshold = best_thresh

        # Test set final predictions
        test_preds = pipeline.predict(X_te)

        # 4. Save Artefacts
        # Save models/decision_policy.json
        policy_path = self.root / self.config["paths"]["models"] / "decision_policy.json"
        save_decision_policy(best_policy, policy_path)

        # Save reports/sample_predictions.csv
        sample_path = self.root / self.config["paths"]["reports"] / "sample_predictions.csv"
        save_sample_predictions(X_te.index, y_te, test_probs, test_preds, sample_path)

        # Log to reports/experiment_history.csv
        total_time = time.time() - t_start
        exp_path = self.root / self.config["paths"]["reports"] / "experiment_history.csv"
        from src.evaluation import compute_classification_metrics
        test_metrics = compute_classification_metrics(y_te, test_preds, test_probs)
        log_experiment(winner_name, test_metrics, best_thresh, total_time, exp_path)

        # Save models/churn_model.joblib & models/model_metadata.json
        self.selection_engine.save(
            model_pipeline=pipeline,
            winner_name=winner_name,
            metrics=test_metrics,
            best_params={},
            dataset_shape=X.shape,
        )

        # Generate logs/evaluation_report.json
        rep_dir = self.root / self.config["paths"]["logs"]
        report = generate_evaluation_report(
            y_true=y_te,
            y_pred=test_preds,
            y_prob=test_probs,
            model=pipeline.model,
            feature_names=pipeline.feature_names,
            threshold_policy=best_policy,
            benchmark_df=df_bench,
            save_dir=rep_dir,
        )

        logger.info(
            "Phase 3 Pipeline Complete! Winner: %s | Optimal Cutoff: %.2f | Test ROC-AUC: %.4f | F1: %.4f | MCC: %.4f",
            winner_name, best_thresh, test_metrics["roc_auc"], test_metrics["f1"], test_metrics["mcc"]
        )

        return {
            "winning_model": winner_name,
            "optimal_threshold": best_thresh,
            "metrics": test_metrics,
            "benchmark_summary": df_bench,
            "threshold_analysis": thresh_df,
            "decision_policy": best_policy,
        }
