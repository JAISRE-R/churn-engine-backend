"""
churn-engine/tests/test_integration_phase6.py
===============================================
Integration tests for Phase 6 (Intervention Routing).
"""

from __future__ import annotations

import pytest
from src.predictor import ChurnPredictor
from src.decision_gate import DecisionGate
from src.recommender import BaselineRecommender
from src.override_engine import OverrideEngine
from src.utils import load_config


@pytest.fixture
def config():
    return load_config()


def test_integration_phase6_flow(config):
    # Setup components
    pred = ChurnPredictor(config)
    gate = DecisionGate(config)
    recommender = BaselineRecommender(config)
    override = OverrideEngine(config)

    # 1. Low risk customer flow
    cid_low = "C-1002"
    pred_low = {
        "customer_id": cid_low,
        "churn_probability": 0.12,
        "risk_level": "LOW",
        "threshold": gate.threshold,
        "timestamp": "2026-01-01 10:00:00",
        "intervention_status": "NORMAL"
    }

    dec_low = gate.decide(
        customer_id=cid_low,
        churn_probability=pred_low["churn_probability"],
        previous_probability=None,
        risk_level=pred_low["risk_level"]
    )
    assert dec_low["decision"] == "NORMAL"

    mock_events = [
        {"product_id": "P-101", "event_type": "product_click", "event_timestamp": "2026-01-01 09:30:00"}
    ]
    base_recs_low = recommender.recommend(cid_low, mock_events)
    
    # Process
    final_low = override.process(base_recs_low, dec_low, mock_events)
    
    # Strategy remains BASELINE, recommendations pass through untouched
    assert final_low["strategy"] == "BASELINE"
    assert final_low["recommendations"] == base_recs_low["recommendations"]
    assert final_low["metadata"]["override_triggered"] is False

    # 2. Critical risk customer flow
    cid_high = "C-1003"
    pred_high = {
        "customer_id": cid_high,
        "churn_probability": 0.88,
        "risk_level": "CRITICAL",
        "threshold": gate.threshold,
        "timestamp": "2026-01-01 10:10:00",
        "intervention_status": "INTERVENTION_REQUIRED"
    }

    dec_high = gate.decide(
        customer_id=cid_high,
        churn_probability=pred_high["churn_probability"],
        previous_probability=None,
        risk_level=pred_high["risk_level"]
    )
    assert dec_high["decision"] == "INTERVENTION_REQUIRED"

    base_recs_high = recommender.recommend(cid_high, mock_events)

    # Process
    final_high = override.process(base_recs_high, dec_high, mock_events)

    # Intercepted! Strategy becomes RETENTION_OVERRIDE
    assert final_high["strategy"] == "RETENTION_OVERRIDE"
    assert final_high["metadata"]["override_triggered"] is True
    assert final_high["metadata"]["discount_pct"] == 25
    
    # Recommendations replaced with coupon and bundle
    pids = [r["product_id"] for r in final_high["recommendations"]]
    assert "COUPON-CRIT-25" in pids
    assert "P-101-BNDL" in pids  # bundle associated with P-101 viewed
