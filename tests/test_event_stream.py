"""
churn-engine/tests/test_event_stream.py
=========================================
Unit tests for src/event_stream.py and src/predictor.py.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.utils import load_config
from src.event_stream import CustomerEvent, EventBuffer, EventSimulator, CustomerStateManager, StreamManager
from src.predictor import ChurnPredictor, RiskTier


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def isolated_config(config, tmp_path):
    """Isolated config variant for testing."""
    cfg = {k: v for k, v in config.items()}
    cfg["paths"] = {k: str(tmp_path / k) for k in config["paths"]}
    # copy raw_events.csv if it exists in base config to temp raw path
    orig_raw_dir = Path(config["paths"]["data_raw"])
    dest_raw_dir = tmp_path / config["paths"]["data_raw"]
    dest_raw_dir.mkdir(parents=True, exist_ok=True)
    if (orig_raw_dir / "raw_events.csv").exists():
        import shutil
        shutil.copy(orig_raw_dir / "raw_events.csv", dest_raw_dir / "raw_events.csv")
    return cfg


# ===========================================================================
# EventBuffer Tests
# ===========================================================================

def test_event_buffer_push_drain():
    buffer = EventBuffer(max_size=5)
    ev = CustomerEvent(
        event_id="evt-1", customer_id="C-1", event_timestamp=datetime.utcnow(),
        event_type="page_view", session_id="S-1"
    )
    buffer.push(ev)
    assert len(buffer) == 1

    drained = buffer.drain(1)
    assert len(drained) == 1
    assert drained[0].event_id == "evt-1"
    assert len(buffer) == 0


def test_event_buffer_to_dataframe():
    buffer = EventBuffer(max_size=5)
    for i in range(3):
        ev = CustomerEvent(
            event_id=f"evt-{i}", customer_id=f"C-{i}", event_timestamp=datetime.utcnow(),
            event_type="page_view", session_id="S-1"
        )
        buffer.push(ev)

    df = buffer.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df["event_id"]) == ["evt-0", "evt-1", "evt-2"]


# ===========================================================================
# EventSimulator Tests
# ===========================================================================

def test_event_simulator_deterministic(isolated_config):
    sim1 = EventSimulator(isolated_config)
    sim2 = EventSimulator(isolated_config)

    ev1 = sim1.generate_event()
    ev2 = sim2.generate_event()

    # With deterministic seed=42 from config, first events should match
    assert ev1.customer_id == ev2.customer_id
    assert ev1.event_type == ev2.event_type


def test_event_simulator_schema_and_types(isolated_config):
    sim = EventSimulator(isolated_config)
    ev = sim.generate_event()

    assert isinstance(ev, CustomerEvent)
    assert isinstance(ev.event_id, str)
    assert ev.customer_id.startswith("C-")
    assert isinstance(ev.event_timestamp, datetime)
    assert isinstance(ev.session_id, str)
    assert ev.event_type in [
        "page_view", "product_click", "add_to_cart",
        "remove_from_cart", "purchase", "search",
        "session_end", "support_contact", "wishlist_click"
    ]


# ===========================================================================
# CustomerStateManager Tests
# ===========================================================================

def test_customer_state_manager_features(isolated_config):
    manager = CustomerStateManager(isolated_config)
    # Get features for unknown customer
    feats_unknown = manager.get_features("C-unknown")
    assert isinstance(feats_unknown, pd.DataFrame)
    assert list(feats_unknown.columns) == CustomerStateManager.FEATURE_ORDER
    assert feats_unknown.loc["C-unknown", "days_since_last_purchase"] == 999.0

    # Add a mock session of events for a new customer
    cid = "C-test-99"
    now = datetime.utcnow()
    events = [
        CustomerEvent("e-1", cid, now - pd.Timedelta(days=5), "page_view", "s-1"),
        CustomerEvent("e-2", cid, now - pd.Timedelta(days=5), "add_to_cart", "s-1"),
        CustomerEvent("e-3", cid, now - pd.Timedelta(days=5), "purchase", "s-1", order_value=120.0, coupon_used=True),
        CustomerEvent("e-4", cid, now - pd.Timedelta(days=1), "support_contact", "s-2"),
    ]

    for ev in events:
        manager.update_state(ev)

    feats = manager.get_features(cid, reference_date=now)
    assert feats.loc[cid, "frequency"] == 4
    assert feats.loc[cid, "monetary_total"] == 120.0
    assert feats.loc[cid, "purchase_count"] == 1
    assert feats.loc[cid, "support_ticket_count"] == 1
    assert feats.loc[cid, "coupon_usage_ratio"] == 1.0
    assert feats.loc[cid, "days_since_last_purchase"] == 5.0
    assert feats.loc[cid, "support_7d"] == 1


# ===========================================================================
# StreamManager Tests
# ===========================================================================

def test_stream_manager_lifecycle(isolated_config):
    sm = StreamManager(isolated_config)
    assert sm.get_stats()["status"] == "stopped"

    sm.start()
    assert sm.get_stats()["status"] == "running"
    time.sleep(0.5)

    stats = sm.get_stats()
    assert stats["total_events"] > 0
    assert len(sm.get_recent_events(5)) > 0

    sm.stop()
    assert sm.get_stats()["status"] == "stopped"


# ===========================================================================
# ChurnPredictor Tests
# ===========================================================================

def test_predictor_fallback_and_classification(isolated_config):
    # Predictor initialization (even without production model file) should fall back gracefully
    pred = ChurnPredictor(isolated_config)

    # Missing model should not crash predict()
    res = pred.predict("C-1001", {"frequency": 1.0, "recency_days": 10.0})
    assert res["customer_id"] == "C-1001"
    assert res["churn_probability"] == 0.0
    assert res["risk_level"] == "LOW"
    assert res["intervention_status"] == "NORMAL"

    # Risk classification mapping
    pred._churn_threshold = 0.40
    pred._medium_risk_lower = 0.20
    pred._high_risk_threshold = 0.80

    assert pred.classify_risk(0.10) == RiskTier.LOW
    assert pred.classify_risk(0.30) == RiskTier.MEDIUM
    assert pred.classify_risk(0.50) == RiskTier.HIGH
    assert pred.classify_risk(0.85) == RiskTier.CRITICAL


def test_predictor_with_real_model(config):
    # This test runs against the real production models/churn_model.joblib
    pred = ChurnPredictor(config)
    assert pred.model is not None, "Real model could not be loaded!"
    
    import json
    with open("models/decision_policy.json") as f:
        policy = json.load(f)
    expected_threshold = policy["selected_threshold"]
    assert pred._churn_threshold == expected_threshold

    # Get sample features for customer C-1001
    manager = CustomerStateManager(config)
    feats = manager.get_features("C-1001")

    # Score
    res = pred.predict("C-1001", feats)
    assert res["customer_id"] == "C-1001"
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert res["intervention_status"] in ["NORMAL", "INTERVENTION_REQUIRED"]
    assert res["threshold"] == expected_threshold

