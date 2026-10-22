"""
churn-engine/src/decision_gate.py
===================================
Decision Gate Component.

Responsibilities
----------------
- Load the authoritative intervention threshold from models/decision_policy.json.
- Convert real-time churn probability scores into explicit business decisions.
- Classify decisions into NORMAL or INTERVENTION_REQUIRED.
- Detect threshold crossing events (i.e. transitions from below to at/above threshold).
- Maintain a thread-safe in-memory decision history log for the current session.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils import get_project_root

logger = logging.getLogger(__name__)


class DecisionGate:
    """Rules engine for churn risk assessment and recommendation override gates."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.root = get_project_root()
        self.policy_path = self.root / config["paths"]["models"] / "decision_policy.json"
        self._lock = threading.Lock()

        # Load policy configurations
        self.threshold = self.load_threshold()
        
        # In-memory history trackers
        self.decision_history: deque[dict[str, Any]] = deque(maxlen=1000)
        self.crossing_events: list[dict[str, Any]] = []

    def load_threshold(self) -> float:
        """Load the authoritative threshold from decision_policy.json.

        Falls back to config default if not found or malformed.
        """
        if self.policy_path.exists():
            try:
                with open(self.policy_path, "r", encoding="utf-8") as f:
                    policy = json.load(f)
                val = float(policy.get("selected_threshold", 0.40))
                logger.info("DecisionGate loaded authoritative threshold: %.2f from %s", val, self.policy_path.name)
                return val
            except Exception as e:
                logger.error("Failed to load policy threshold: %s. Using default fallback.", e)
        
        fallback = float(self.config.get("decision_gate", {}).get("churn_threshold", 0.40))
        logger.warning("DecisionGate falling back to config threshold: %.2f", fallback)
        return fallback

    def decide(
        self,
        customer_id: str,
        churn_probability: float,
        previous_probability: float | None = None,
        risk_level: str = "LOW"
    ) -> dict[str, Any]:
        """Convert a churn probability into a baseline vs override decision.

        Parameters
        ----------
        customer_id : str
        churn_probability : float
        previous_probability : float | None
        risk_level : str

        Returns
        -------
        dict[str, Any]
            Decision metrics dictionary.
        """
        with self._lock:
            threshold = self.threshold

        # Detect transition: was below threshold, is now at/above
        threshold_crossed = False
        if previous_probability is not None:
            was_below = previous_probability < threshold
            is_above = churn_probability >= threshold
            threshold_crossed = was_below and is_above

        decision = "INTERVENTION_REQUIRED" if churn_probability >= threshold else "NORMAL"
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        result = {
            "timestamp": timestamp,
            "customer_id": customer_id,
            "churn_probability": round(churn_probability, 4),
            "previous_probability": round(previous_probability, 4) if previous_probability is not None else None,
            "threshold": threshold,
            "decision": decision,
            "threshold_crossed": threshold_crossed,
            "risk_level": risk_level,
        }

        with self._lock:
            self.decision_history.append(result)
            if threshold_crossed:
                self.crossing_events.append(result)

        return result

    def get_history(self) -> list[dict[str, Any]]:
        """Return decision logs sorted in FIFO order."""
        with self._lock:
            return list(self.decision_history)

    def get_crossing_events(self) -> list[dict[str, Any]]:
        """Return all crossing events detected."""
        with self._lock:
            return list(self.crossing_events)

    def clear(self) -> None:
        """Reset historical logs."""
        with self._lock:
            self.decision_history.clear()
            self.crossing_events.clear()
            # Reload policy to pick up changes
            self.threshold = self.load_threshold()
