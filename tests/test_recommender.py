"""
churn-engine/tests/test_recommender.py
========================================
Unit tests for BaselineRecommender and recommendation contracts.
"""

from __future__ import annotations

import pytest
from src.recommender import BaselineRecommender, get_product_category
from src.utils import load_config


@pytest.fixture
def config():
    return load_config()


def test_get_product_category():
    assert get_product_category("P-150") == "Electronics"
    assert get_product_category("P-350") == "Fashion & Apparel"
    assert get_product_category("P-600") == "Home & Living"
    assert get_product_category("P-800") == "Beauty & Health"
    assert get_product_category("P-950") == "Sports & Outdoors"
    assert get_product_category("invalid") == "General"


def test_recommender_empty_and_unknown_history(config):
    recommender = BaselineRecommender(config)
    
    # Empty history
    res = recommender.recommend("C-unknown", [])
    assert res["customer_id"] == "C-unknown"
    assert res["strategy"] == "BASELINE"
    assert len(res["recommendations"]) == 5
    assert all(r["reason"] == "Trending Product" for r in res["recommendations"])

    # None history
    res2 = recommender.recommend("C-unknown", None)
    assert len(res2["recommendations"]) == 5


def test_recommender_with_events_history(config):
    recommender = BaselineRecommender(config)
    
    # 3 mock events
    events = [
        {"product_id": "P-105", "event_type": "product_click", "event_timestamp": "2026-01-01 10:00:00"},
        {"product_id": "P-450", "event_type": "add_to_cart", "event_timestamp": "2026-01-01 10:05:00"},
        {"product_id": "P-602", "event_type": "wishlist_click", "event_timestamp": "2026-01-01 10:10:00"},
    ]

    res = recommender.recommend("C-1001", events)
    assert res["customer_id"] == "C-1001"
    assert res["strategy"] == "BASELINE"
    assert len(res["recommendations"]) == 5

    # Check that interacted products are prioritized with higher scores
    pids = [r["product_id"] for r in res["recommendations"]]
    assert "P-602" in pids  # Newest event product
    assert "P-450" in pids
    assert "P-105" in pids
    
    # Check score ordering (descending)
    scores = [r["score"] for r in res["recommendations"]]
    assert scores == sorted(scores, reverse=True)


def test_recommender_contract(config):
    recommender = BaselineRecommender(config)
    res = recommender.recommend("C-99")
    
    required_keys = ["customer_id", "recommendations", "strategy", "timestamp"]
    for key in required_keys:
        assert key in res
        
    for rec in res["recommendations"]:
        assert "product_id" in rec
        assert "category" in rec
        assert "reason" in rec
        assert "score" in rec
