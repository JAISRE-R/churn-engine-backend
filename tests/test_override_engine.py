"""
churn-engine/tests/test_override_engine.py
============================================
Unit tests for the OverrideEngine component.
"""

from __future__ import annotations

import pytest
from src.override_engine import OverrideEngine
from src.recommender import BaselineRecommender
from src.utils import load_config


@pytest.fixture
def config():
    return load_config()


def test_override_engine_normal_pass_through(config):
    engine = OverrideEngine(config)
    
    baseline = {
        "customer_id": "C-1",
        "recommendations": [
            {"product_id": "P-101", "category": "Electronics", "reason": "Trending Product", "score": 0.85}
        ],
        "strategy": "BASELINE",
        "timestamp": "2026-01-01 10:00:00"
    }

    decision = {
        "customer_id": "C-1",
        "churn_probability": 0.15,
        "threshold": 0.40,
        "decision": "NORMAL",
        "risk_level": "LOW"
    }

    # Pass-through condition: decision == NORMAL
    res = engine.process(baseline, decision, None)
    
    assert res["strategy"] == "BASELINE"
    assert res["recommendations"] == baseline["recommendations"]
    assert res["metadata"]["override_triggered"] is False
    assert len(engine.get_audit_log()) == 0


def test_override_engine_intervention_trigger(config):
    engine = OverrideEngine(config)
    
    baseline = {
        "customer_id": "C-1",
        "recommendations": [
            {"product_id": "P-101", "category": "Electronics", "reason": "Trending Product", "score": 0.85}
        ],
        "strategy": "BASELINE",
        "timestamp": "2026-01-01 10:00:00"
    }

    decision = {
        "customer_id": "C-1",
        "churn_probability": 0.85,
        "threshold": 0.40,
        "decision": "INTERVENTION_REQUIRED",
        "risk_level": "CRITICAL"
    }

    # Intervention triggered: decision == INTERVENTION_REQUIRED
    res = engine.process(baseline, decision, None)
    
    assert res["strategy"] == "RETENTION_OVERRIDE"
    assert res["metadata"]["override_triggered"] is True
    assert res["metadata"]["discount_pct"] == 25  # default critical_risk_discount = 25%
    
    # Audit log generated
    assert len(engine.get_audit_log()) == 1
    assert engine.get_audit_log()[0]["customer_id"] == "C-1"


def test_override_engine_risk_tier_strategies(config):
    engine = OverrideEngine(config)
    
    baseline = {
        "customer_id": "C-1",
        "recommendations": [
            {"product_id": "P-101", "category": "Electronics", "reason": "Trending Product", "score": 0.85}
        ],
        "strategy": "BASELINE",
        "timestamp": "2026-01-01 10:00:00"
    }

    # Test HIGH risk strategy (15% discount)
    dec_high = {
        "customer_id": "C-1",
        "churn_probability": 0.65,
        "threshold": 0.40,
        "decision": "INTERVENTION_REQUIRED",
        "risk_level": "HIGH"
    }
    res_high = engine.process(baseline, dec_high, None)
    assert res_high["metadata"]["discount_pct"] == 15
    assert all(r.get("discount", 0) == 0 for r in res_high["recommendations"])

    # Test CRITICAL risk strategy (25% discount, bundle deal, etc.)
    dec_crit = {
        "customer_id": "C-1",
        "churn_probability": 0.85,
        "threshold": 0.40,
        "decision": "INTERVENTION_REQUIRED",
        "risk_level": "CRITICAL"
    }
    res_crit = engine.process(baseline, dec_crit, None)
    assert res_crit["metadata"]["discount_pct"] == 25
    assert any("BNDL" in r["product_id"] for r in res_crit["recommendations"])
    assert all(r.get("discount", 0) == 0 for r in res_crit["recommendations"])


def test_override_engine_targets_offers_to_eligible_products(config):
    engine = OverrideEngine(config)
    baseline = {
        "customer_id": "C-1",
        "recommendations": [
            {"product_id": "P-1002", "category": "Laptops", "reason": "Baseline", "score": 0.8}
        ],
        "strategy": "BASELINE",
        "timestamp": "2026-01-01 10:00:00",
    }
    decision = {
        "customer_id": "C-1",
        "churn_probability": 0.65,
        "threshold": 0.40,
        "decision": "INTERVENTION_REQUIRED",
        "risk_level": "HIGH",
    }
    events = [
        {"product_id": "P-1001", "event_type": "add_to_cart"},
        {"product_id": "P-1003", "event_type": "product_click"},
    ]

    result = engine.process(baseline, decision, events)
    offers = {
        item["product_id"]: item["discount"]
        for item in result["recommendations"]
        if item.get("discount", 0) > 0
    }

    assert offers == {"P-1001": 15, "P-1003": 15}
    assert next(item for item in result["recommendations"] if item["product_id"] == "P-1002").get(
        "discount", 0
    ) == 0


def test_intervention_without_eligible_products_has_no_offer(config):
    engine = OverrideEngine(config)
    baseline = {
        "customer_id": "C-1",
        "recommendations": [
            {"product_id": "P-1002", "category": "Laptops", "reason": "Baseline", "score": 0.8}
        ],
        "strategy": "BASELINE",
        "timestamp": "2026-01-01 10:00:00",
    }
    decision = {
        "customer_id": "C-1",
        "churn_probability": 0.65,
        "threshold": 0.40,
        "decision": "INTERVENTION_REQUIRED",
        "risk_level": "HIGH",
    }

    result = engine.process(baseline, decision, [])

    assert result["strategy"] == "RETENTION_OVERRIDE"
    assert all(item.get("discount", 0) == 0 for item in result["recommendations"])


def test_override_engine_rejects_unauthorized_customer_offers(config):
    engine = OverrideEngine(config)
    baseline = {
        "customer_id": "C-1001",
        "recommendations": [
            {"product_id": "P-1001", "category": "Laptops", "reason": "Baseline", "score": 0.8}
        ],
        "strategy": "BASELINE",
        "timestamp": "2026-01-01 10:00:00",
    }
    decision = {
        "customer_id": "C-1001",
        "churn_probability": 0.8,
        "threshold": 0.5,
        "decision": "INTERVENTION_REQUIRED",
        "risk_level": "HIGH",
    }
    events = [{"product_id": "P-1001", "event_type": "add_to_cart"}]

    result = engine.process(
        baseline,
        decision,
        events,
        offers_allowed=False,
    )

    assert result["strategy"] == "BASELINE"
    assert result["metadata"]["override_triggered"] is False
    assert "not authorized" in result["metadata"]["override_reason"]
    assert all(item.get("discount", 0) == 0 for item in result["recommendations"])


def test_local_demo_mapping_is_enforced_by_override_engine(config, monkeypatch):
    from src.demo_mode import DEMO_AT_RISK_CUSTOMERS, DEMO_NOT_AT_RISK_CUSTOMERS

    monkeypatch.setenv("CHURN_LOCAL_DEMO_MODE", "true")
    monkeypatch.setenv("CHURN_ENVIRONMENT", "local")
    for marker in ("RENDER", "FLY_APP_NAME", "DYNO", "K_SERVICE", "AWS_LAMBDA_FUNCTION_NAME", "WEBSITE_INSTANCE_ID"):
        monkeypatch.delenv(marker, raising=False)

    engine = OverrideEngine(config)
    decision = {
        "churn_probability": 0.8,
        "threshold": 0.5,
        "decision": "INTERVENTION_REQUIRED",
        "risk_level": "HIGH",
    }
    events = [{"product_id": "P-1001", "event_type": "add_to_cart"}]

    for customer_id in DEMO_AT_RISK_CUSTOMERS | DEMO_NOT_AT_RISK_CUSTOMERS:
        baseline = {
            "customer_id": customer_id,
            "recommendations": [
                {"product_id": "P-1002", "category": "Laptops", "reason": "Baseline", "score": 0.8}
            ],
            "strategy": "BASELINE",
            "timestamp": "2026-01-01 10:00:00",
        }
        result = engine.process(baseline, decision, events)
        offers = [
            item for item in result["recommendations"]
            if item.get("discount", 0) > 0
        ]
        if customer_id in DEMO_AT_RISK_CUSTOMERS:
            assert [item["product_id"] for item in offers] == ["P-1001"]
        else:
            assert offers == []


def test_override_engine_behavioral_affinity(config):
    engine = OverrideEngine(config)
    
    baseline = {
        "customer_id": "C-1",
        "recommendations": [],
        "strategy": "BASELINE",
        "timestamp": "2026-01-01 10:00:00"
    }

    dec_crit = {
        "customer_id": "C-1",
        "churn_probability": 0.85,
        "threshold": 0.40,
        "decision": "INTERVENTION_REQUIRED",
        "risk_level": "CRITICAL"
    }

    # Ingest viewed items & cart abandonments
    events = [
        {"product_id": "P-202", "event_type": "product_click", "event_timestamp": "2026-01-01 09:00:00"},
        {"product_id": "P-202", "event_type": "add_to_cart", "event_timestamp": "2026-01-01 09:01:00"},
    ]

    res = engine.process(baseline, dec_crit, events)
    pids = [r["product_id"] for r in res["recommendations"]]
    
    # 1. P-202-BNDL should be present (bundle deal for P-202)
    assert "P-202-BNDL" in pids
    # 2. P-202 should be present (abandoned cart item saved)
    assert "P-202" in pids


def test_override_engine_contract_structure(config):
    engine = OverrideEngine(config)
    baseline = {
        "customer_id": "C-1",
        "recommendations": [
            {"product_id": "P-101", "category": "Electronics", "reason": "Trending Product", "score": 0.85}
        ],
        "strategy": "BASELINE",
        "timestamp": "2026-01-01 10:00:00"
    }
    decision = {
        "customer_id": "C-1",
        "churn_probability": 0.85,
        "threshold": 0.40,
        "decision": "INTERVENTION_REQUIRED",
        "risk_level": "CRITICAL"
    }

    res = engine.process(baseline, decision, None)
    
    # Top-level keys check
    for key in ["customer_id", "recommendations", "strategy", "timestamp", "metadata"]:
        assert key in res

    # Metadata sub-keys check
    meta = res["metadata"]
    for key in ["override_triggered", "risk_level", "churn_probability", "threshold", "override_reason", "discount_pct"]:
        assert key in meta

    # Recommendations sub-keys check
    for rec in res["recommendations"]:
        assert "product_id" in rec
        assert "category" in rec
        assert "reason" in rec
        assert "score" in rec
