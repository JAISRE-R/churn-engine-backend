"""
churn-engine/tests/test_integration_phase5.py
===============================================
Integration tests for Churn Predictor, Decision Gate, and Recommendation Engine.
"""

from __future__ import annotations

import pytest
from src.predictor import ChurnPredictor
from src.decision_gate import DecisionGate
from src.recommender import BaselineRecommender
from src.utils import load_config


@pytest.fixture
def config():
    return load_config()


def test_integration_predictor_decision_recommender(config):
    # Setup components
    pred = ChurnPredictor(config)
    gate = DecisionGate(config)
    recommender = BaselineRecommender(config)

    # 1. Simulate customer scoring with high risk
    cid = "C-1001"
    # Create fake feature vector that should lead to a prediction
    # Mocking single prediction output
    pred_res_high = {
        "customer_id": cid,
        "churn_probability": 0.85,
        "risk_level": "CRITICAL",
        "threshold": gate.threshold,
        "timestamp": "2026-01-01 12:00:00",
        "intervention_status": "INTERVENTION_REQUIRED"
    }

    # 2. Pass prediction output to Decision Gate
    decision_high = gate.decide(
        customer_id=cid,
        churn_probability=pred_res_high["churn_probability"],
        previous_probability=0.30,
        risk_level=pred_res_high["risk_level"]
    )

    assert decision_high["decision"] == "INTERVENTION_REQUIRED"
    assert decision_high["threshold_crossed"] is True

    # 3. Request recommendations
    # Mock history showing cart abandonment
    mock_events = [
        {"product_id": "P-101", "event_type": "product_click", "event_timestamp": "2026-01-01 10:00:00"},
        {"product_id": "P-101", "event_type": "add_to_cart", "event_timestamp": "2026-01-01 10:01:00"},
    ]
    recs_high = recommender.recommend(cid, mock_events)

    # Verify strategy is BASELINE (it remains unchanged by churn status in Phase 5)
    assert recs_high["strategy"] == "BASELINE"
    assert len(recs_high["recommendations"]) == 5

    # 4. Repeat with a low risk customer
    pred_res_low = {
        "customer_id": cid,
        "churn_probability": 0.15,
        "risk_level": "LOW",
        "threshold": gate.threshold,
        "timestamp": "2026-01-01 12:05:00",
        "intervention_status": "NORMAL"
    }

    decision_low = gate.decide(
        customer_id=cid,
        churn_probability=pred_res_low["churn_probability"],
        previous_probability=0.10,
        risk_level=pred_res_low["risk_level"]
    )
    assert decision_low["decision"] == "NORMAL"
    assert decision_low["threshold_crossed"] is False

    recs_low = recommender.recommend(cid, mock_events)
    # The output strategy and recommendations must be identical regardless of churn
    assert recs_low["strategy"] == "BASELINE"
    assert recs_low["recommendations"] == recs_high["recommendations"]
