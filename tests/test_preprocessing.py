"""
churn-engine/tests/test_preprocessing.py
===========================================
Unit tests for src/preprocessing.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import DataCleaner, DataLoader, PreprocessingPipeline
from src.utils import load_config


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def raw_sample_df():
    return pd.DataFrame([
        {
            "EVENT_ID": "evt-001",
            "Customer ID": "C-1001",
            "Event Timestamp": "2026-01-01 10:00:00",
            "event_type": "page_view",
            "session_id": "S-001",
            "order_value": None,
            "device_type": "desktop",
            "country": "US",
            "churn": 0,
        },
        {
            "EVENT_ID": "evt-001",  # Duplicate row
            "Customer ID": "C-1001",
            "Event Timestamp": "2026-01-01 10:00:00",
            "event_type": "page_view",
            "session_id": "S-001",
            "order_value": None,
            "device_type": "desktop",
            "country": "US",
            "churn": 0,
        },
    ])


class TestDataCleaner:

    def test_normalise_column_names(self, config, raw_sample_df):
        cleaner = DataCleaner(config)
        norm_df = cleaner.normalise_column_names(raw_sample_df)
        assert "event_id" in norm_df.columns
        assert "customer_id" in norm_df.columns
        assert "event_timestamp" in norm_df.columns

    def test_drop_duplicates(self, config, raw_sample_df):
        cleaner = DataCleaner(config)
        norm_df = cleaner.normalise_column_names(raw_sample_df)
        dedup_df = cleaner.drop_duplicates(norm_df, subset=["event_id"])
        assert len(dedup_df) == 1

    def test_handle_missing_values(self, config, raw_sample_df):
        cleaner = DataCleaner(config)
        norm_df = cleaner.normalise_column_names(raw_sample_df)
        clean_df = cleaner.handle_missing_values(norm_df)
        assert clean_df["order_value"].iloc[0] == 0.0

    def test_cast_dtypes(self, config, raw_sample_df):
        cleaner = DataCleaner(config)
        norm_df = cleaner.normalise_column_names(raw_sample_df)
        casted_df = cleaner.cast_dtypes(norm_df)
        assert pd.api.types.is_datetime64_any_dtype(casted_df["event_timestamp"])
        assert isinstance(casted_df["event_type"].dtype, pd.CategoricalDtype)
