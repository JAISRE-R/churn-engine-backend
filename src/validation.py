"""
churn-engine/src/validation.py
================================
Pre-Preprocessing Data Validation Stage.

Responsibilities
----------------
- Accept raw data DataFrame from DataLoader.
- Execute 7 essential quality checks before cleaning:
    1. Missing / null customer IDs.
    2. Future timestamps (timestamp > current UTC time).
    3. Negative order values (order_value < 0).
    4. Invalid / unrecognized event types.
    5. Duplicate session IDs shared across multiple customers.
    6. Invalid country codes (outside ISO list).
    7. Invalid churn labels (outside {0, 1}).
- Output a structured Data Validation Report dictionary and save to `logs/data_validation_report.json`.
- Output validated dataset to `data/intermediate/validated_events.parquet`.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import ensure_dir, get_project_root, load_config

logger = logging.getLogger(__name__)

VALID_EVENT_TYPES = {
    "page_view", "product_click", "add_to_cart",
    "remove_from_cart", "purchase", "search",
    "session_end", "support_contact", "wishlist_click"
}

VALID_COUNTRIES = {"US", "CA", "GB", "DE", "FR", "AU", "JP"}


class DataValidator:
    """Validate raw customer event logs against key quality rules.

    Parameters
    ----------
    config : dict[str, Any]
        Full project configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.root: Path = get_project_root()
        self.intermediate_dir: Path = self.root / config["paths"]["data_intermediate"]
        self.log_dir: Path = self.root / config["paths"]["logs"]
        ensure_dir(self.intermediate_dir)
        ensure_dir(self.log_dir)

    def validate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Run all 7 quality validation checks on the raw DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw event DataFrame.

        Returns
        -------
        tuple[pd.DataFrame, dict[str, Any]]
            Validated DataFrame (with invalid records flagged/removed) and
            the validation summary report dict.
        """
        logger.info("Starting Data Validation stage on %d records...", len(df))
        report: dict[str, Any] = {
            "validation_timestamp": datetime.utcnow().isoformat(),
            "total_records_ingested": len(df),
            "checks": {},
            "status": "PASSED",
        }

        working_df = df.copy()

        # 1. Missing customer IDs
        missing_cust_mask = working_df["customer_id"].isna() | (working_df["customer_id"].astype(str).str.strip() == "")
        report["checks"]["missing_customer_ids"] = {
            "failed_count": int(missing_cust_mask.sum()),
            "status": "FAILED" if missing_cust_mask.sum() > 0 else "PASSED",
        }
        if missing_cust_mask.sum() > 0:
            working_df = working_df[~missing_cust_mask]

        # 2. Future timestamps
        working_df["_parsed_ts"] = pd.to_datetime(working_df["event_timestamp"], errors="coerce")
        now_utc = pd.Timestamp.now()
        future_mask = working_df["_parsed_ts"] > now_utc
        report["checks"]["future_timestamps"] = {
            "failed_count": int(future_mask.sum()),
            "status": "FAILED" if future_mask.sum() > 0 else "PASSED",
        }
        if future_mask.sum() > 0:
            working_df = working_df[~future_mask]

        # 3. Negative order values
        neg_val_mask = (working_df["event_type"] == "purchase") & (working_df["order_value"] < 0)
        report["checks"]["negative_order_values"] = {
            "failed_count": int(neg_val_mask.sum()),
            "status": "FAILED" if neg_val_mask.sum() > 0 else "PASSED",
        }
        if neg_val_mask.sum() > 0:
            working_df.loc[neg_val_mask, "order_value"] = np.nan

        # 4. Invalid event types
        invalid_evt_mask = ~working_df["event_type"].isin(VALID_EVENT_TYPES)
        report["checks"]["invalid_event_types"] = {
            "failed_count": int(invalid_evt_mask.sum()),
            "status": "FAILED" if invalid_evt_mask.sum() > 0 else "PASSED",
        }
        if invalid_evt_mask.sum() > 0:
            working_df = working_df[~invalid_evt_mask]

        # 5. Duplicate session IDs across different customers
        sess_cust = working_df.groupby("session_id")["customer_id"].nunique()
        multi_cust_sessions = set(sess_cust[sess_cust > 1].index)
        dup_sess_mask = working_df["session_id"].isin(multi_cust_sessions)
        report["checks"]["duplicate_session_ids"] = {
            "failed_count": int(dup_sess_mask.sum()),
            "status": "FAILED" if dup_sess_mask.sum() > 0 else "PASSED",
        }

        # 6. Invalid countries
        invalid_ctry_mask = ~working_df["country"].isin(VALID_COUNTRIES)
        report["checks"]["invalid_countries"] = {
            "failed_count": int(invalid_ctry_mask.sum()),
            "status": "FAILED" if invalid_ctry_mask.sum() > 0 else "PASSED",
        }

        # 7. Invalid churn labels
        invalid_churn_mask = ~working_df["churn"].isin([0, 1])
        report["checks"]["invalid_churn_labels"] = {
            "failed_count": int(invalid_churn_mask.sum()),
            "status": "FAILED" if invalid_churn_mask.sum() > 0 else "PASSED",
        }

        # Overall Status
        any_failures = any(c["status"] == "FAILED" for c in report["checks"].values())
        report["status"] = "WARNINGS_FOUND" if any_failures else "PASSED"
        report["valid_records_retained"] = len(working_df)

        # Cleanup internal column
        working_df.drop(columns=["_parsed_ts"], inplace=True, errors="ignore")

        # Save outputs
        self._save_report(report)
        self._save_validated_data(working_df)

        logger.info("Validation complete | Status: %s | Retained: %d/%d", report["status"], len(working_df), len(df))
        return working_df, report

    def _save_report(self, report: dict[str, Any]) -> None:
        report_path = self.log_dir / "data_validation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved data validation report → %s", report_path)

    def _save_validated_data(self, df: pd.DataFrame) -> Path:
        out_path = self.intermediate_dir / "validated_events.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Saved validated dataset → %s", out_path)
        return out_path
