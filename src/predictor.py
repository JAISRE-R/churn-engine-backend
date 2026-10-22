"""
churn-engine/src/predictor.py
===============================
Real-Time Churn Score Inference Engine.

Responsibilities
----------------
- Load a serialised XGBoost/RandomForest ChurnModelPipeline from disk.
- Load the intervention threshold from models/decision_policy.json.
- Score individual customer features in real-time.
- Map probability scores to descriptive risk tiers and intervention decisions.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils import get_project_root

logger = logging.getLogger(__name__)


# ===========================================================================
# Risk Tier Enum
# ===========================================================================

class RiskTier(str, Enum):
    """Churn risk classification tiers based on predicted probability.

    Tiers
    -----
    LOW      : probability < medium_risk_lower threshold
    MEDIUM   : medium_risk_lower <= probability < churn_threshold
    HIGH     : churn_threshold <= probability < high_risk_threshold
    CRITICAL : probability >= high_risk_threshold
    """
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


# ===========================================================================
# Churn Predictor
# ===========================================================================

class ChurnPredictor:
    """Load a trained model and score customer records for churn probability."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.root = get_project_root()

        # Config fallback values
        self.gate_cfg: dict[str, Any] = config.get("decision_gate", {})
        model_cfg: dict[str, Any] = config.get("model", {})

        self.model_path: Path = (
            self.root / config["paths"]["models"] / model_cfg.get("model_filename", "churn_model.joblib")
        )
        self.policy_path: Path = (
            self.root / config["paths"]["models"] / "decision_policy.json"
        )

        self.model: Any = None
        self._churn_threshold: float = float(self.gate_cfg.get("churn_threshold", 0.40))
        self._high_risk_threshold: float = float(self.gate_cfg.get("high_risk_threshold", 0.80))
        self._medium_risk_lower: float = float(self.gate_cfg.get("medium_risk_lower", 0.20))

        # Dynamically load the model and threshold at startup
        try:
            self.load_model()
            self.load_decision_policy()
        except Exception as e:
            logger.warning("Predictor initialised but could not load model/policy: %s", e)

    def load_model(self) -> None:
        """Load the serialised model pipeline into memory.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        try:
            self.model = joblib.load(self.model_path)
            logger.info("Loaded serialized model pipeline from: %s", self.model_path)

            if not hasattr(self.model, "predict_proba"):
                raise AttributeError("Loaded model pipeline does not expose predict_proba().")

            # Check if model container has a custom feature_names list
            if hasattr(self.model, "feature_names"):
                logger.info("Model expects %d features: %s", len(self.model.feature_names), self.model.feature_names)
        except Exception as e:
            logger.error("Error loading model pipeline: %s", e)
            raise

    def load_decision_policy(self) -> None:
        """Load the intervention threshold and risk configurations from policy file."""
        if not self.policy_path.exists():
            logger.warning("Decision policy file not found at: %s. Using config defaults.", self.policy_path)
            return

        try:
            with open(self.policy_path, "r", encoding="utf-8") as f:
                policy = json.load(f)

            if "selected_threshold" in policy:
                self._churn_threshold = float(policy["selected_threshold"])
                logger.info("Intervention threshold successfully updated from policy: %.2f", self._churn_threshold)

                # Ensure Risk Tiers align logically if the threshold shifted
                if self._medium_risk_lower >= self._churn_threshold:
                    self._medium_risk_lower = max(0.05, round(self._churn_threshold * 0.5, 2))
                if self._high_risk_threshold <= self._churn_threshold:
                    self._high_risk_threshold = min(0.95, round(self._churn_threshold + 0.30, 2))
        except Exception as e:
            logger.error("Error reading decision policy file: %s", e)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return churn probability scores for a feature DataFrame.

        Parameters
        ----------
        features : pd.DataFrame
            Feature matrix matching the training columns.

        Returns
        -------
        np.ndarray
            1-D array of churn probabilities in [0.0, 1.0].
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        try:
            t0 = time.time()
            probs = self.model.predict_proba(features)
            latency = (time.time() - t0) * 1000.0
            logger.debug("Inference call completed in %.2f ms", latency)
            return probs
        except Exception as e:
            logger.error("Failed to perform prediction: %s", e)
            raise

    def predict_single(self, feature_vector: dict[str, Any]) -> dict[str, Any]:
        """Score a single customer from a raw feature dictionary."""
        customer_id = feature_vector.get("customer_id", "unknown")
        # Convert dictionary to 1-row DataFrame, dropping customer_id if present
        clean_features = {k: v for k, v in feature_vector.items() if k != "customer_id"}
        df_feat = pd.DataFrame([clean_features])

        probs = self.predict_proba(df_feat)
        prob = float(probs[0])

        risk = self.classify_risk(prob)
        override = prob >= self._churn_threshold

        return {
            "customer_id": customer_id,
            "churn_probability": prob,
            "risk_tier": risk,
            "override_flag": override,
        }

    def predict(self, customer_id: str, customer_features: pd.DataFrame | dict[str, Any]) -> dict[str, Any]:
        """Core real-time interface for Phase 4 stream and future decision gates.

        Parameters
        ----------
        customer_id : str
            Unique customer ID to score.
        customer_features : pd.DataFrame | dict[str, Any]
            Feature vector.

        Returns
        -------
        dict[str, Any]
            Inference results container.
        """
        if isinstance(customer_features, dict):
            clean_feats = {k: v for k, v in customer_features.items() if k != "customer_id"}
            df_feat = pd.DataFrame([clean_feats])
        else:
            df_feat = customer_features.copy()
            if "customer_id" in df_feat.columns:
                df_feat = df_feat.drop(columns=["customer_id"])

        if self.model is None:
            # Safe prediction fallback if model cannot load
            logger.error("Cannot score customer %s: Model is None", customer_id)
            return {
                "customer_id": customer_id,
                "churn_probability": 0.0,
                "risk_level": "LOW",
                "threshold": self._churn_threshold,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "intervention_status": "NORMAL",
            }

        try:
            prob = float(self.predict_proba(df_feat)[0])
            risk_tier = self.classify_risk(prob)
            decision = "INTERVENTION_REQUIRED" if prob >= self._churn_threshold else "NORMAL"

            return {
                "customer_id": customer_id,
                "churn_probability": round(prob, 4),
                "risk_level": risk_tier.value,
                "threshold": self._churn_threshold,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "intervention_status": decision,
            }
        except Exception as e:
            logger.error("Prediction failed for customer %s: %s", customer_id, e)
            return {
                "customer_id": customer_id,
                "churn_probability": 0.0,
                "risk_level": "LOW",
                "threshold": self._churn_threshold,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "intervention_status": "NORMAL",
            }

    def classify_risk(self, probability: float) -> RiskTier:
        """Map predicted probability to RiskTier category."""
        if probability < self._medium_risk_lower:
            return RiskTier.LOW
        elif probability < self._churn_threshold:
            return RiskTier.MEDIUM
        elif probability < self._high_risk_threshold:
            return RiskTier.HIGH
        else:
            return RiskTier.CRITICAL

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score multiple customers and append prediction outputs."""
        if len(df) == 0:
            df_out = df.copy()
            df_out["churn_probability"] = []
            df_out["risk_tier"] = []
            df_out["override_flag"] = []
            return df_out

        # Align features
        feats_input = df.copy()
        if "customer_id" in feats_input.columns:
            feats_input.set_index("customer_id", inplace=True)

        probs = self.predict_proba(feats_input)

        out = df.copy()
        out["churn_probability"] = probs
        out["risk_tier"] = [self.classify_risk(p).value for p in probs]
        out["override_flag"] = probs >= self._churn_threshold
        return out
