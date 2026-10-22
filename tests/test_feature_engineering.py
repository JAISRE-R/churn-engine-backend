"""
churn-engine/tests/test_feature_engineering.py
=================================================
Unit tests for src/feature_engineering.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering import (
    FeatureEngineeringPipeline,
    RFMBuilder,
    RollingFeatureBuilder,
    SessionFeatureBuilder,
)
from src.utils import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def clean_events_df():
    return pd.DataFrame([
        {
            "event_id": "evt-001",
            "customer_id": "C-1001",
            "event_timestamp": pd.Timestamp("2026-01-01 10:00:00"),
            "event_type": "page_view",
            "session_id": "S-001",
            "product_id": "P-100",
            "order_value": 0.0,
            "coupon_used": False,
            "device_type": "desktop",
            "country": "US",
            "churn": 0,
        },
        {
            "event_id": "evt-002",
            "customer_id": "C-1001",
            "event_timestamp": pd.Timestamp("2026-01-01 10:05:00"),
            "event_type": "add_to_cart",
            "session_id": "S-001",
            "product_id": "P-100",
            "order_value": 0.0,
            "coupon_used": False,
            "device_type": "desktop",
            "country": "US",
            "churn": 0,
        },
        {
            "event_id": "evt-003",
            "customer_id": "C-1001",
            "event_timestamp": pd.Timestamp("2026-01-01 10:10:00"),
            "event_type": "purchase",
            "session_id": "S-001",
            "product_id": "P-100",
            "order_value": 50.0,
            "coupon_used": True,
            "device_type": "desktop",
            "country": "US",
            "churn": 0,
        },
        {
            "event_id": "evt-004",
            "customer_id": "C-1002",
            "event_timestamp": pd.Timestamp("2026-01-02 12:00:00"),
            "event_type": "support_contact",
            "session_id": "S-002",
            "product_id": "NONE",
            "order_value": 0.0,
            "coupon_used": False,
            "device_type": "mobile",
            "country": "CA",
            "churn": 1,
        },
    ])


class TestFeatureEngineering:

    def test_rfm_builder(self, config, clean_events_df):
        rfm = RFMBuilder(config).compute(clean_events_df)
        assert "C-1001" in rfm.index
        assert "recency_days" in rfm.columns
        assert rfm.loc["C-1001", "frequency"] == 3
        assert rfm.loc["C-1001", "monetary_total"] == 50.0

    def test_rolling_builder(self, config, clean_events_df):
        rolling = RollingFeatureBuilder(config).build(clean_events_df)
        assert "events_7d" in rolling.columns
        assert "purchases_30d" in rolling.columns
        assert rolling.loc["C-1001", "events_7d"] == 3

    def test_session_builder(self, config, clean_events_df):
        sess = SessionFeatureBuilder(config).build(clean_events_df)
        required_cols = [
            "avg_order_value", "avg_purchase_interval_days",
            "days_since_last_purchase", "session_frequency",
            "support_ticket_count", "coupon_usage_ratio",
            "wishlist_click_count", "cart_abandonment_rate",
            "avg_pages_per_session", "avg_session_duration_sec",
            "device_switch_count"
        ]
        for col in required_cols:
            assert col in sess.columns, f"Missing feature column: {col}"

        assert sess.loc["C-1001", "avg_order_value"] == 50.0
        assert sess.loc["C-1001", "coupon_usage_ratio"] == 1.0
        assert sess.loc["C-1002", "support_ticket_count"] == 1

    def test_pipeline_build(self, config, clean_events_df, tmp_path):
        cfg = dict(config)
        cfg["paths"] = dict(config["paths"])
        cfg["paths"]["data_features"] = str(tmp_path)
        pipeline = FeatureEngineeringPipeline(cfg)
        features = pipeline.build(clean_events_df)
        assert len(features) == 2
        assert "churn" in features.columns
        assert features.loc["C-1001", "churn"] == 0
        assert features.loc["C-1002", "churn"] == 1
