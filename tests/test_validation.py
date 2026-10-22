"""
churn-engine/tests/test_validation.py
=======================================
Unit tests for src/validation.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.utils import load_config
from src.validation import DataValidator


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def isolated_config(config, tmp_path):
    cfg = {k: v for k, v in config.items()}
    cfg["paths"] = {k: str(tmp_path / k) for k in config["paths"]}
    return cfg


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "event_id": "evt-001",
            "customer_id": "C-1001",
            "event_timestamp": "2026-01-01 10:00:00",
            "event_type": "page_view",
            "session_id": "S-001",
            "order_value": 0.0,
            "country": "US",
            "churn": 0,
        },
        {
            "event_id": "evt-002",
            "customer_id": "C-1001",
            "event_timestamp": "2026-01-01 10:05:00",
            "event_type": "purchase",
            "session_id": "S-001",
            "order_value": 45.50,
            "country": "US",
            "churn": 0,
        },
        {
            "event_id": "evt-003",
            "customer_id": "C-1002",
            "event_timestamp": "2026-01-02 12:00:00",
            "event_type": "support_contact",
            "session_id": "S-002",
            "order_value": 0.0,
            "country": "CA",
            "churn": 1,
        },
    ])


class TestDataValidator:

    def test_validation_passes_clean_data(self, isolated_config, sample_df):
        validator = DataValidator(isolated_config)
        validated_df, report = validator.validate(sample_df)
        assert len(validated_df) == 3
        assert report["status"] == "PASSED"
        assert report["checks"]["missing_customer_ids"]["status"] == "PASSED"

    def test_validation_detects_missing_customer_ids(self, isolated_config, sample_df):
        invalid_df = sample_df.copy()
        invalid_df.loc[0, "customer_id"] = None
        validator = DataValidator(isolated_config)
        validated_df, report = validator.validate(invalid_df)

        assert report["checks"]["missing_customer_ids"]["status"] == "FAILED"
        assert report["checks"]["missing_customer_ids"]["failed_count"] == 1
        assert len(validated_df) == 2

    def test_validation_detects_negative_order_value(self, isolated_config, sample_df):
        invalid_df = sample_df.copy()
        invalid_df.loc[1, "order_value"] = -10.0
        validator = DataValidator(isolated_config)
        validated_df, report = validator.validate(invalid_df)

        assert report["checks"]["negative_order_values"]["status"] == "FAILED"
        assert pd.isna(validated_df.loc[1, "order_value"])

    def test_validation_detects_invalid_event_types(self, isolated_config, sample_df):
        invalid_df = sample_df.copy()
        invalid_df.loc[2, "event_type"] = "unknown_hacked_event"
        validator = DataValidator(isolated_config)
        validated_df, report = validator.validate(invalid_df)

        assert report["checks"]["invalid_event_types"]["status"] == "FAILED"
        assert len(validated_df) == 2
