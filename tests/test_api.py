"""
churn-engine/tests/test_api.py
==============================
Unit and integration tests for FastAPI thin bridge (api.py).
"""

from __future__ import annotations

import pytest
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient

from api import app, create_app, VALID_CATALOG_SET, resolve_region, adapt_recommendations
from src.checkout_ledger import CheckoutLedger
from src.utils import load_config
from src.event_stream import CustomerStateManager
from src.predictor import ChurnPredictor
from src.decision_gate import DecisionGate
from src.recommender import BaselineRecommender
from src.override_engine import OverrideEngine


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient instance with the main application."""
    monkeypatch.setattr(
        app.state,
        "checkout_ledger",
        CheckoutLedger(tmp_path / "checkout_ledger.sqlite3"),
    )
    with TestClient(app) as test_client:
        yield test_client


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


def test_demo_customer_options_are_server_owned_and_do_not_enable_demo_mode(
    client, monkeypatch
):
    from src.demo_mode import DEMO_AT_RISK_CUSTOMERS, DEMO_NOT_AT_RISK_CUSTOMERS

    monkeypatch.setattr(app.state, "local_demo_enabled", False)
    response = client.get("/api/demo/customers")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["demo_mode"] is False
    assert {item["customer_id"] for item in data["customers"]} == (
        DEMO_AT_RISK_CUSTOMERS | DEMO_NOT_AT_RISK_CUSTOMERS
    )
    assert all(item["label"].startswith("Demo customer ") for item in data["customers"])
    assert all("classification" not in item for item in data["customers"])

    monkeypatch.setattr(app.state, "local_demo_enabled", True)
    local_demo = client.get("/api/demo/customers").json()
    assert local_demo["available"] is True
    assert local_demo["demo_mode"] is True


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


def test_adapt_recommendations_preserves_only_authorized_product_offers():
    recs = [
        {"product_id": "P-1001", "discount": 15},
        {"product_id": "P-1002"},
        {"product_id": "COUPON-CRIT-25", "discount": 25, "category": "Laptops"},
    ]

    authorized = adapt_recommendations(recs, offers_authorized=True)
    assert [item["discount"] for item in authorized] == [15, 0, 0]

    baseline = adapt_recommendations(recs, offers_authorized=False)
    assert all(item["discount"] == 0 for item in baseline)


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


def _stub_checkout_pipeline(monkeypatch, *, decision, risk_level, events):
    class StateManager:
        customer_history = {"C-CHECKOUT": events}

        def get_features(self, customer_id, reference_date=None):
            return {}

        def update_state(self, event):
            self.customer_history.setdefault(event.customer_id, []).append({
                "product_id": event.product_id,
                "event_type": event.event_type,
                "event_timestamp": event.event_timestamp,
            })

    class Predictor:
        def predict(self, customer_id, features):
            return {
                "churn_probability": 0.8 if decision == "INTERVENTION_REQUIRED" else 0.1,
                "risk_level": risk_level,
            }

    class Gate:
        def get_history(self):
            return []

        def decide(self, **kwargs):
            return {
                "customer_id": kwargs["customer_id"],
                "churn_probability": kwargs["churn_probability"],
                "threshold": 0.4,
                "decision": decision,
                "risk_level": risk_level,
            }

    class Recommender:
        def recommend(self, customer_id, customer_events):
            return {
                "customer_id": customer_id,
                "recommendations": [
                    {
                        "product_id": "P-1002",
                        "category": "Laptops",
                        "reason": "Baseline",
                        "score": 0.8,
                    }
                ],
                "strategy": "BASELINE",
                "timestamp": "2026-01-01 10:00:00",
            }

    monkeypatch.setattr(app.state, "state_manager", StateManager())
    monkeypatch.setattr(app.state, "predictor", Predictor())
    monkeypatch.setattr(app.state, "decision_gate", Gate())
    monkeypatch.setattr(app.state, "recommender", Recommender())
    monkeypatch.setattr(app.state, "override_engine", OverrideEngine(load_config()))


def test_checkout_quote_applies_offer_only_to_eligible_product(client, monkeypatch):
    _stub_checkout_pipeline(
        monkeypatch,
        decision="INTERVENTION_REQUIRED",
        risk_level="HIGH",
        events=[{"product_id": "P-1001", "event_type": "add_to_cart"}],
    )

    res = client.post(
        "/api/checkout/quote",
        json={
            "customer_id": "C-CHECKOUT",
            "lines": [
                {"product_id": "P-1001", "quantity": 1},
                {"product_id": "P-1002", "quantity": 1},
            ],
        },
    )

    assert res.status_code == 200
    data = res.json()
    assert data["subtotal"] == 3648
    assert data["discount_value"] == 217.35
    assert data["lines"][0]["discount_percent"] == 15
    assert data["lines"][0]["line_total"] == 1231.65
    assert data["lines"][1]["discount_percent"] == 0
    assert data["lines"][1]["line_total"] == 2199
    assert data["tax"] == 274.45
    assert data["total"] == 3705.1

    experience = client.get("/api/experience/C-CHECKOUT").json()
    assert experience["discount"] == 0
    assert {
        item["product_id"]: item["discount"]
        for item in experience["recommendations"]
    }["P-1001"] == 15
    assert next(
        item["discount"]
        for item in experience["recommendations"]
        if item["product_id"] == "P-1002"
    ) == 0


def test_checkout_quote_has_no_discount_without_intervention(client, monkeypatch):
    _stub_checkout_pipeline(
        monkeypatch,
        decision="NORMAL",
        risk_level="LOW",
        events=[{"product_id": "P-1001", "event_type": "add_to_cart"}],
    )

    res = client.post(
        "/api/checkout/quote",
        json={
            "customer_id": "C-CHECKOUT",
            "lines": [{"product_id": "P-1001", "quantity": 1}],
        },
    )

    assert res.status_code == 200
    data = res.json()
    assert data["discount_value"] == 0
    assert data["lines"][0]["discount_percent"] == 0
    assert data["lines"][0]["line_total"] == 1449


def test_checkout_quote_with_no_eligible_product_has_no_cart_wide_offer(client, monkeypatch):
    _stub_checkout_pipeline(
        monkeypatch,
        decision="INTERVENTION_REQUIRED",
        risk_level="HIGH",
        events=[],
    )

    res = client.post(
        "/api/checkout/quote",
        json={
            "customer_id": "C-CHECKOUT",
            "lines": [{"product_id": "P-1001", "quantity": 1}],
        },
    )

    assert res.status_code == 200
    data = res.json()
    assert data["discount_value"] == 0
    assert data["lines"][0]["discount_percent"] == 0


def test_checkout_quote_rejects_unknown_catalog_product(client):
    res = client.post(
        "/api/checkout/quote",
        json={
            "customer_id": "C-CHECKOUT",
            "lines": [{"product_id": "UNKNOWN", "quantity": 1}],
        },
    )

    assert res.status_code == 422


def test_checkout_quote_rejects_client_claimed_offer_or_price(client):
    res = client.post(
        "/api/checkout/quote",
        json={
            "customer_id": "C-CHECKOUT",
            "churn_probability": 1,
            "discount": 100,
            "eligible": True,
            "lines": [
                {
                    "product_id": "P-1001",
                    "quantity": 1,
                    "price": 0,
                    "discount_percent": 100,
                }
            ],
        },
    )

    assert res.status_code == 422


def test_local_demo_groups_gate_quotes_and_direct_event_offers(client, monkeypatch):
    from src.demo_mode import DEMO_AT_RISK_CUSTOMERS, DEMO_NOT_AT_RISK_CUSTOMERS

    monkeypatch.setattr(app.state, "local_demo_enabled", True)

    for customer_id in DEMO_AT_RISK_CUSTOMERS | DEMO_NOT_AT_RISK_CUSTOMERS:
        _stub_checkout_pipeline(
            monkeypatch,
            decision="INTERVENTION_REQUIRED",
            risk_level="HIGH",
            events=[{"product_id": "P-1001", "event_type": "add_to_cart"}],
        )
        app.state.state_manager.customer_history = {
            customer_id: [{"product_id": "P-1001", "event_type": "add_to_cart"}]
        }

        experience = client.get(f"/api/experience/{customer_id}").json()
        quote = client.post(
            "/api/checkout/quote",
            json={
                "customer_id": customer_id,
                "lines": [{"product_id": "P-1001", "quantity": 1}],
            },
        ).json()

        if customer_id in DEMO_AT_RISK_CUSTOMERS:
            assert experience["demo_classification"] == "DEMO_AT_RISK"
            assert experience["decision"] == "INTERVENTION_REQUIRED"
            assert quote["lines"][0]["discount_percent"] == 15
        else:
            assert experience["demo_classification"] == "DEMO_NOT_AT_RISK"
            assert experience["decision"] == "INTERVENTION_REQUIRED"
            assert all(item["discount"] == 0 for item in experience["recommendations"])
            assert quote["lines"][0]["discount_percent"] == 0

            event_response = client.post(
                "/api/events",
                json={
                    "customer_id": customer_id,
                    "session_id": "S-demo-denial",
                    "event_type": "add_to_cart",
                    "product_id": "P-1001",
                },
            )
            assert event_response.status_code == 200
            assert all(item["discount"] == 0 for item in event_response.json()["recommendations"])

            finalized = client.post(
                "/api/checkout/finalize",
                json={
                    "order_id": f"ORD-DENY-{customer_id}",
                    "customer_id": customer_id,
                    "lines": [{"product_id": "P-1001", "quantity": 1}],
                },
            )
            assert finalized.status_code == 200
            assert finalized.json()["retention_offer_status"] == "NOT_REDEEMED"
            assert finalized.json()["discount_value"] == 0


def test_checkout_finalization_is_idempotent_and_consumes_offer_once(client, monkeypatch):
    monkeypatch.setattr(app.state, "local_demo_enabled", True)
    _stub_checkout_pipeline(
        monkeypatch,
        decision="INTERVENTION_REQUIRED",
        risk_level="HIGH",
        events=[{"product_id": "P-1001", "event_type": "add_to_cart"}],
    )
    app.state.state_manager.customer_history = {
        "C-1000": [{"product_id": "P-1001", "event_type": "add_to_cart"}]
    }
    payload = {
        "order_id": "ORD-IDEMPOTENT-DEMO",
        "customer_id": "C-1000",
        "lines": [{"product_id": "P-1001", "quantity": 1}],
    }

    preview = client.post("/api/checkout/quote", json={
        "customer_id": payload["customer_id"],
        "lines": payload["lines"],
    }).json()
    assert preview["discount_value"] == 217.35

    finalized = client.post("/api/checkout/finalize", json=payload)
    repeated = client.post("/api/checkout/finalize", json=payload)
    assert finalized.status_code == repeated.status_code == 200
    assert finalized.json() == repeated.json()
    assert finalized.json()["retention_offer_status"] == "REDEEMED"
    assert finalized.json()["discount_value"] == 217.35

    next_order = client.post(
        "/api/checkout/finalize",
        json={
            **payload,
            "order_id": "ORD-SECOND-DEMO",
        },
    )
    assert next_order.status_code == 200
    assert next_order.json()["discount_value"] == 0
    assert next_order.json()["retention_offer_status"] == "NOT_REDEEMED"

    conflict = client.post(
        "/api/checkout/finalize",
        json={
            **payload,
            "customer_id": "C-1002",
        },
    )
    assert conflict.status_code == 409


def test_concurrent_checkouts_cannot_redeem_the_same_product_offer(client, monkeypatch):
    monkeypatch.setattr(app.state, "local_demo_enabled", True)
    _stub_checkout_pipeline(
        monkeypatch,
        decision="INTERVENTION_REQUIRED",
        risk_level="HIGH",
        events=[{"product_id": "P-1001", "event_type": "add_to_cart"}],
    )
    app.state.state_manager.customer_history = {
        "C-1000": [{"product_id": "P-1001", "event_type": "add_to_cart"}]
    }

    def finalize(order_id):
        with TestClient(app) as concurrent_client:
            return concurrent_client.post(
                "/api/checkout/finalize",
                json={
                    "order_id": order_id,
                    "customer_id": "C-1000",
                    "lines": [{"product_id": "P-1001", "quantity": 1}],
                },
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(finalize, ["ORD-CONCURRENT-1", "ORD-CONCURRENT-2"]))

    assert all(response.status_code == 200 for response in responses)
    assert sorted(response.json()["discount_value"] for response in responses) == [
        0,
        217.35,
    ]


def test_concurrent_retries_of_one_order_finalize_only_once(client, monkeypatch):
    monkeypatch.setattr(app.state, "local_demo_enabled", True)
    _stub_checkout_pipeline(
        monkeypatch,
        decision="INTERVENTION_REQUIRED",
        risk_level="HIGH",
        events=[{"product_id": "P-1001", "event_type": "add_to_cart"}],
    )
    app.state.state_manager.customer_history = {
        "C-1000": [{"product_id": "P-1001", "event_type": "add_to_cart"}]
    }

    def finalize(_):
        with TestClient(app) as concurrent_client:
            return concurrent_client.post(
                "/api/checkout/finalize",
                json={
                    "order_id": "ORD-CONCURRENT-RETRY",
                    "customer_id": "C-1000",
                    "lines": [{"product_id": "P-1001", "quantity": 1}],
                },
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(finalize, range(2)))

    assert all(response.status_code == 200 for response in responses)
    assert responses[0].json() == responses[1].json()
    with app.state.checkout_ledger.transaction() as connection:
        order_count = connection.execute(
            "SELECT COUNT(*) FROM finalized_checkouts WHERE order_id = ?",
            ("ORD-CONCURRENT-RETRY",),
        ).fetchone()[0]
        redemption_count = connection.execute(
            "SELECT COUNT(*) FROM redeemed_offers WHERE order_id = ?",
            ("ORD-CONCURRENT-RETRY",),
        ).fetchone()[0]
    assert order_count == 1
    assert redemption_count == 1


def test_checkout_redemption_and_idempotency_survive_app_restart(
    client, monkeypatch
):
    monkeypatch.setattr(app.state, "local_demo_enabled", True)
    _stub_checkout_pipeline(
        monkeypatch,
        decision="INTERVENTION_REQUIRED",
        risk_level="HIGH",
        events=[{"product_id": "P-1001", "event_type": "add_to_cart"}],
    )
    app.state.state_manager.customer_history = {
        "C-1000": [{"product_id": "P-1001", "event_type": "add_to_cart"}]
    }
    ledger_path = app.state.checkout_ledger.path
    payload = {
        "order_id": "ORD-PERSISTED",
        "customer_id": "C-1000",
        "lines": [{"product_id": "P-1001", "quantity": 1}],
    }
    first = client.post("/api/checkout/finalize", json=payload)
    assert first.status_code == 200
    assert first.json()["discount_value"] == 217.35

    restarted_app = create_app(
        config=app.state.config,
        state_manager=app.state.state_manager,
        predictor=app.state.predictor,
        decision_gate=app.state.decision_gate,
        recommender=app.state.recommender,
        override_engine=app.state.override_engine,
        checkout_ledger=CheckoutLedger(ledger_path),
    )
    restarted_app.state.local_demo_enabled = True
    with TestClient(restarted_app) as restarted_client:
        retry = restarted_client.post("/api/checkout/finalize", json=payload)
        next_order = restarted_client.post(
            "/api/checkout/finalize",
            json={**payload, "order_id": "ORD-AFTER-RESTART"},
        )

    assert retry.status_code == 200
    assert retry.json() == first.json()
    assert next_order.status_code == 200
    assert next_order.json()["discount_value"] == 0
    assert next_order.json()["retention_offer_status"] == "NOT_REDEEMED"


def test_production_requires_explicit_demo_cors_and_ledger_config(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CHURN_ENVIRONMENT", "production")
    monkeypatch.setenv("CHURN_LOCAL_DEMO_MODE", "false")
    monkeypatch.setenv("CORS_ORIGINS", "https://shop.example.com")
    monkeypatch.delenv("CHECKOUT_LEDGER_PATH", raising=False)
    dependencies = {
        "config": app.state.config,
        "state_manager": app.state.state_manager,
        "predictor": app.state.predictor,
        "decision_gate": app.state.decision_gate,
        "recommender": app.state.recommender,
        "override_engine": app.state.override_engine,
    }

    with pytest.raises(RuntimeError, match="CHECKOUT_LEDGER_PATH"):
        create_app(**dependencies)

    temporary_path = tmp_path / "render-ephemeral" / "checkout.sqlite3"
    temporary_path.parent.mkdir()
    monkeypatch.setenv("CHECKOUT_LEDGER_PATH", str(temporary_path))
    production_app = create_app(**dependencies)

    assert production_app.state.local_demo_enabled is False
    assert production_app.state.checkout_ledger.path == temporary_path
    assert temporary_path.exists()

    ledger_path = tmp_path / "persistent" / "checkout.sqlite3"
    ledger_path.parent.mkdir()
    monkeypatch.setenv("CHECKOUT_LEDGER_PATH", str(ledger_path))
    persistent_app = create_app(**dependencies)

    assert persistent_app.state.local_demo_enabled is False
    assert persistent_app.state.checkout_ledger.path == ledger_path
    assert ledger_path.exists()


def test_production_rejects_local_cors_origins(monkeypatch, tmp_path):
    monkeypatch.setenv("CHURN_ENVIRONMENT", "production")
    monkeypatch.setenv("CHURN_LOCAL_DEMO_MODE", "false")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173")
    monkeypatch.setenv(
        "CHECKOUT_LEDGER_PATH", str(tmp_path / "checkout.sqlite3")
    )
    dependencies = {
        "config": app.state.config,
        "state_manager": app.state.state_manager,
        "predictor": app.state.predictor,
        "decision_gate": app.state.decision_gate,
        "recommender": app.state.recommender,
        "override_engine": app.state.override_engine,
    }

    with pytest.raises(RuntimeError, match="public HTTPS frontend origins"):
        create_app(**dependencies)


def test_local_demo_flag_is_explicit_local_only(monkeypatch):
    from src.demo_mode import local_demo_enabled

    config = {"project": {"environment": "development"}}
    monkeypatch.setenv("CHURN_LOCAL_DEMO_MODE", "true")
    monkeypatch.setenv("CHURN_ENVIRONMENT", "local")
    assert local_demo_enabled(config) is True

    monkeypatch.setenv("CHURN_ENVIRONMENT", "production")
    assert local_demo_enabled(config) is False

    monkeypatch.setenv("CHURN_ENVIRONMENT", "local")
    monkeypatch.setenv("RENDER", "true")
    assert local_demo_enabled(config) is False

    monkeypatch.delenv("RENDER")
    assert local_demo_enabled({"project": {"environment": "production"}}) is False


def test_model_and_authoritative_threshold_remain_unchanged(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_type"] == "RandomForestClassifier"
    assert response.json()["threshold"] == 0.5


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
