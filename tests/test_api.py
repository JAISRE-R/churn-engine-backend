"""
churn-engine/tests/test_api.py
==============================
Unit and integration tests for FastAPI thin bridge (api.py).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import app, create_app, VALID_CATALOG_SET, resolve_region, adapt_recommendations
from src.utils import load_config
from src.event_stream import CustomerStateManager
from src.predictor import ChurnPredictor
from src.decision_gate import DecisionGate
from src.recommender import BaselineRecommender
from src.override_engine import OverrideEngine


@pytest.fixture
def client():
    """TestClient instance with the main application."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test 1: GET /health returns 200 and valid status."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "churn-engine"
    assert data["model_loaded"] is True
    assert "threshold" in data
    assert data["customers_loaded"] > 0


def test_post_event_valid(client):
    """Test 2: POST /api/events processes a shopping event."""
    payload = {
        "customer_id": "C-1000",
        "session_id": "S-test-01",
        "event_type": "product_view",
        "product_id": "P-1003",
        "category": "Laptops",
        "quantity": 1,
        "order_value": 749.0,
        "query": None,
        "timestamp": "2026-09-19T04:00:00.000Z",
        "metadata": {},
    }
    res = client.post("/api/events", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["customer_id"] == "C-1000"
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "decision" in data
    assert "strategy" in data
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)


def test_get_experience_valid(client):
    """Test 3: GET /api/experience/{customer_id} returns experience without appending event."""
    res = client.get("/api/experience/C-1000")
    assert res.status_code == 200
    data = res.json()
    assert data["customer_id"] == "C-1000"
    assert data["decision"] in ["NORMAL", "INTERVENTION_REQUIRED"]
    assert data["strategy"] in ["BASELINE", "RETENTION_OVERRIDE"]


def test_invalid_request_handling(client):
    """Test 4: Invalid requests return 422 Unprocessable Entity."""
    # Missing required fields customer_id and session_id
    res = client.post("/api/events", json={"product_id": "P-1001"})
    assert res.status_code == 422

    # Empty customer_id for GET
    res_get = client.get("/api/experience/%20")
    assert res_get.status_code in [400, 404, 422]


def test_unknown_customer_handling(client):
    """Test 5: Unknown customer succeeds gracefully using baseline defaults."""
    res = client.get("/api/experience/C-999999")
    assert res.status_code == 200
    data = res.json()
    assert data["customer_id"] == "C-999999"
    assert data["decision"] in ["NORMAL", "INTERVENTION_REQUIRED"]
    assert data["strategy"] in ["BASELINE", "RETENTION_OVERRIDE"]
    assert len(data["recommendations"]) > 0


def test_event_normalization(client):
    """Test 6: Frontend event types normalize to backend schema correctly."""
    payload = {
        "customer_id": "C-1001",
        "session_id": "S-norm-01",
        "event_type": "product_view",
        "product_id": "P-1002",
        "category": "Laptops",
        "timestamp": "2026-09-19T04:05:00.000Z",
    }
    res = client.post("/api/events", json=payload)
    assert res.status_code == 200

    # Verify event stored with normalized type in state manager
    sm = app.state.state_manager
    history = sm.customer_history.get("C-1001", [])
    last_event = history[-1]
    assert last_event["event_type"] == "product_click"
    assert last_event["product_id"] == "P-1002"


def test_customer_state_update(client):
    """Test 7: POST /api/events appends to customer history in CustomerStateManager."""
    sm = app.state.state_manager
    initial_count = len(sm.customer_history.get("C-1042", []))

    payload = {
        "customer_id": "C-1042",
        "session_id": "S-upd-01",
        "event_type": "add_to_cart",
        "product_id": "P-1005",
        "order_value": 299.0,
    }
    res = client.post("/api/events", json=payload)
    assert res.status_code == 200

    new_count = len(sm.customer_history.get("C-1042", []))
    assert new_count == initial_count + 1


def test_response_schema_completeness(client):
    """Test 8: Response JSON adheres strictly to the CustomerExperience schema."""
    res = client.get("/api/experience/C-1327")
    assert res.status_code == 200
    data = res.json()

    required_fields = [
        "customer_id", "region", "churn_probability", "risk_level",
        "decision", "strategy", "discount", "headline", "message",
        "recommendation_type", "recommendations"
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"

    for rec in data["recommendations"]:
        assert "product_id" in rec
        assert "reason" in rec
        assert "discount" in rec


def test_valid_frontend_product_ids(client):
    """Test 9: All returned recommendation product IDs belong to frontend catalog (P-1001..P-1036)."""
    res = client.get("/api/experience/C-1000")
    assert res.status_code == 200
    data = res.json()
    for rec in data["recommendations"]:
        assert rec["product_id"] in VALID_CATALOG_SET, (
            f"Product ID {rec['product_id']} not in frontend catalog!"
        )


def test_region_mapping():
    """Test 10: Region mapping resolves countries and IDs deterministically."""
    assert resolve_region("C-1000", "US") == "North America"
    assert resolve_region("C-1000", "CA") == "North America"
    assert resolve_region("C-1000", "GB") == "Europe"
    assert resolve_region("C-1000", "AU") == "Asia-Pacific"
    assert resolve_region("C-1000", "JP") == "Asia-Pacific"
    assert resolve_region("C-1327", None) in [
        "Asia-Pacific", "North America", "Europe", "Latin America", "Middle East"
    ]


def test_baseline_and_intervention_flow(client):
    """Test 11 & 12: Baseline pass-through vs Intervention override handling."""
    # Test baseline
    res_base = client.get("/api/experience/C-1000")
    assert res_base.status_code == 200
    data_base = res_base.json()
    if data_base["decision"] == "NORMAL":
        assert data_base["strategy"] == "BASELINE"
        assert data_base["discount"] == 0
        assert data_base["recommendation_type"] == "personalized"

    # Test direct pipeline with an at-risk customer trigger
    # Create isolated customer with high churn features (frequent support tickets, high inactivity)
    sm = app.state.state_manager
    cid_risk = "C-HIGH-RISK-TEST"
    for i in range(10):
        from src.event_stream import CustomerEvent
        from datetime import datetime, timedelta
        sm.update_state(CustomerEvent(
            event_id=f"evt-risk-{i}",
            customer_id=cid_risk,
            event_timestamp=datetime.utcnow() - timedelta(days=90),
            event_type="support_contact",
            session_id=f"S-risk-{i}",
            country="US",
            customer_age_days=300,
            churn=1,
        ))

    res_risk = client.get(f"/api/experience/{cid_risk}")
    assert res_risk.status_code == 200
    data_risk = res_risk.json()
    assert data_risk["customer_id"] == cid_risk
    # Recommendations should still be valid catalog products
    for rec in data_risk["recommendations"]:
        assert rec["product_id"] in VALID_CATALOG_SET


def test_live_event_buffer_receives_events(client):
    """Test 13: POST /api/events appends live records into LiveEventBuffer."""
    payload = {
        "customer_id": "C-1002",
        "session_id": "S-live-buf-01",
        "event_type": "product_view",
        "product_id": "P-1004",
        "category": "Audio",
        "order_value": 99.0,
    }
    res_post = client.post("/api/events", json=payload)
    assert res_post.status_code == 200

    res_admin = client.get("/api/admin/events/recent?limit=10")
    assert res_admin.status_code == 200
    data = res_admin.json()
    assert data["status"] == "ok"
    assert data["total_events"] > 0
    assert len(data["events"]) > 0

    latest = data["events"][0]
    assert latest["customer_id"] == "C-1002"
    assert latest["event_type"] == "product_view"
    assert latest["product_id"] == "P-1004"
    assert "churn_probability" in latest
    assert "risk_level" in latest
    assert "decision" in latest
    assert "strategy" in latest
    assert "discount" in latest
    assert "recommendations" in latest
    assert "region" in latest


def test_admin_events_limit_parameter(client):
    """Test 14: GET /api/admin/events/recent respects limit query parameter."""
    for i in range(5):
        client.post("/api/events", json={
            "customer_id": f"C-100{i}",
            "session_id": f"S-lim-{i}",
            "event_type": "product_view",
            "product_id": f"P-100{i+1}",
        })

    res_lim = client.get("/api/admin/events/recent?limit=3")
    assert res_lim.status_code == 200
    data = res_lim.json()
    assert len(data["events"]) == 3
    assert data["count"] == 3


def test_live_event_buffer_thread_safety():
    """Test 15: LiveEventBuffer handles concurrent appends safely."""
    import threading
    from api import LiveEventBuffer

    buf = LiveEventBuffer(maxlen=500)
    threads = []

    def worker(idx):
        for j in range(50):
            buf.append({"id": f"evt-{idx}-{j}", "val": j})

    for t in range(5):
        th = threading.Thread(target=worker, args=(t,))
        threads.append(th)
        th.start()

    for th in threads:
        th.join()

    assert buf.get_total_count() == 250
    assert len(buf.get_recent(limit=500)) == 250


def test_simulation_mode_and_live_store_mode_isolation():
    """Test 16: Simulation Mode generates synthetic events; Live Store Mode stops simulator and isolates events."""
    import time
    from src.event_stream import StreamManager

    config = load_config()
    sm = StreamManager(config)

    # In simulation mode: StreamManager runs and generates synthetic events
    sm.start()
    assert sm._running is True
    time.sleep(0.3)
    sim_stats = sm.get_stats()
    assert sim_stats["status"] == "running"
    initial_sim_count = sim_stats["total_events"]
    assert initial_sim_count > 0

    # Switching to Live Store Mode stops the synthetic generator
    sm.stop()
    assert sm._running is False
    assert sm.get_stats()["status"] == "stopped"

    # Verify no new synthetic events are generated when stopped
    time.sleep(0.3)
    assert sm.get_stats()["total_events"] == initial_sim_count


