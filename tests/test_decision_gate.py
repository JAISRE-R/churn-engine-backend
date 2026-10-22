"""
churn-engine/tests/test_decision_gate.py
==========================================
Unit tests for DecisionGate component.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from src.decision_gate import DecisionGate
from src.utils import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def isolated_config(config, tmp_path):
    """Config with temp models path to isolate policy testing."""
    cfg = {k: v for k, v in config.items()}
    cfg["paths"] = {k: str(tmp_path / k) for k in config["paths"]}
    # Ensure the models directory exists in tmp_path
    (tmp_path / config["paths"]["models"]).mkdir(parents=True, exist_ok=True)
    return cfg


def test_decision_gate_threshold_logic(isolated_config):
    # Setup test policy file in isolated tmp path
    models_dir = Path(isolated_config["paths"]["models"])
    policy_file = models_dir / "decision_policy.json"
    
    policy_data = {
        "selected_threshold": 0.40,
        "selection_criteria": "Maximize F1",
        "business_justification": "Optimal boundary"
    }
    with open(policy_file, "w", encoding="utf-8") as f:
        json.dump(policy_data, f)

    dg = DecisionGate(isolated_config)
    assert dg.threshold == 0.40

    # Probability below threshold
    res1 = dg.decide("C-1001", churn_probability=0.35, previous_probability=0.30, risk_level="LOW")
    assert res1["decision"] == "NORMAL"
    assert res1["threshold_crossed"] is False

    # Probability equal to threshold
    res2 = dg.decide("C-1001", churn_probability=0.40, previous_probability=0.35, risk_level="HIGH")
    assert res2["decision"] == "INTERVENTION_REQUIRED"
    assert res2["threshold_crossed"] is True

    # Probability above threshold
    res3 = dg.decide("C-1001", churn_probability=0.55, previous_probability=0.50, risk_level="CRITICAL")
    assert res3["decision"] == "INTERVENTION_REQUIRED"
    assert res3["threshold_crossed"] is False


def test_decision_gate_threshold_crossing_cases(isolated_config):
    dg = DecisionGate(isolated_config)
    dg.threshold = 0.40

    # Case A: Below to Above -> True
    resA = dg.decide("C-1", churn_probability=0.45, previous_probability=0.35)
    assert resA["threshold_crossed"] is True

    # Case B: Already Above -> False
    resB = dg.decide("C-1", churn_probability=0.48, previous_probability=0.45)
    assert resB["threshold_crossed"] is False

    # Case C: Below to Below -> False
    resC = dg.decide("C-1", churn_probability=0.38, previous_probability=0.35)
    assert resC["threshold_crossed"] is False

    # Case D: Above to Below -> False
    resD = dg.decide("C-1", churn_probability=0.35, previous_probability=0.45)
    assert resD["threshold_crossed"] is False


def test_decision_gate_history_and_clear(isolated_config):
    dg = DecisionGate(isolated_config)
    dg.threshold = 0.40

    # Make three decisions
    dg.decide("C-1", churn_probability=0.30)
    dg.decide("C-2", churn_probability=0.50, previous_probability=0.35)
    dg.decide("C-3", churn_probability=0.55, previous_probability=0.50)

    # Validate history
    history = dg.get_history()
    crossings = dg.get_crossing_events()

    assert len(history) == 3
    assert len(crossings) == 1
    assert crossings[0]["customer_id"] == "C-2"

    # Clear history
    dg.clear()
    assert len(dg.get_history()) == 0
    assert len(dg.get_crossing_events()) == 0


def test_decision_gate_policy_loading_fallback(isolated_config):
    # File policy doesn't exist, should fall back to config
    dg = DecisionGate(isolated_config)
    assert dg.threshold == float(isolated_config["decision_gate"]["churn_threshold"])
