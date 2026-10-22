"""
churn-engine/src/feature_engineering.py
=========================================
Feature Engineering Pipeline (Business Domain Features).

Responsibilities
----------------
- Construct RFM (Recency, Frequency, Monetary) customer features.
- Calculate 11+ explicit customer-level behavioural features:
    1. avg_order_value
    2. avg_purchase_interval_days
    3. days_since_last_purchase
    4. session_frequency
    5. support_ticket_count
    6. coupon_usage_ratio
    7. wishlist_click_count
    8. cart_abandonment_rate
    9. avg_pages_per_session
    10. avg_session_duration_sec
    11. device_switch_count
- Generate rolling-window aggregates (7d, 14d, 30d).
- Save unscaled, unencoded customer features to `data/features/customer_features.parquet`.

Architectural Note
------------------
Scaling (StandardScaler) and Categorical Encoding (OneHotEncoder / LabelEncoder)
are explicitly deferred to the ML Training Pipeline (Phase 3) to prevent data leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import ensure_dir, get_project_root

logger = logging.getLogger(__name__)


# ===========================================================================
# RFM Feature Builder
# ===========================================================================

class RFMBuilder:
    """Compute Recency, Frequency, and Monetary (RFM) features per customer.

    Parameters
    ----------
    config : dict[str, Any]
        Full project configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def compute(self, df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
        """Compute customer-level RFM features.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned event DataFrame.
        reference_date : pd.Timestamp | None
            Snapshot anchor date. Defaults to max timestamp in df.

        Returns
        -------
        pd.DataFrame
            DataFrame with index `customer_id` and RFM columns.
        """
        if reference_date is None:
            reference_date = df["event_timestamp"].max()

        rfm = df.groupby("customer_id").agg(
            last_event_time=("event_timestamp", "max"),
            first_event_time=("event_timestamp", "min"),
            frequency=("event_id", "count"),
            monetary_total=("order_value", "sum"),
        )

        rfm["recency_days"] = (reference_date - rfm["last_event_time"]).dt.total_seconds() / (24 * 3600)
        rfm["recency_days"] = rfm["recency_days"].clip(lower=0.0)

        # Log transform for skewed monetary distribution
        rfm["monetary_log"] = np.log1p(rfm["monetary_total"])

        return rfm.drop(columns=["last_event_time", "first_event_time"])


# ===========================================================================
# Rolling Window Feature Builder
# ===========================================================================

class RollingFeatureBuilder:
    """Generate rolling-window behavioural aggregates over 7d, 14d, 30d.

    Parameters
    ----------
    config : dict[str, Any]
        Full project configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        fe_cfg = config.get("feature_engineering", {})
        self.windows: list[int] = fe_cfg.get("rolling_windows", [7, 14, 30])

    def build(self, df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
        """Compute rolling-window event counts per customer.

        Parameters
        ----------
        df : pd.DataFrame
        reference_date : pd.Timestamp | None

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by `customer_id` with rolling count columns.
        """
        if reference_date is None:
            reference_date = df["event_timestamp"].max()

        results = pd.DataFrame(index=df["customer_id"].unique())
        results.index.name = "customer_id"

        for w in self.windows:
            window_start = reference_date - pd.Timedelta(days=w)
            sub_df = df[(df["event_timestamp"] >= window_start) & (df["event_timestamp"] <= reference_date)]

            # Total events in window
            counts = sub_df.groupby("customer_id")["event_id"].count()
            results[f"events_{w}d"] = counts.reindex(results.index, fill_value=0)

            # Purchases in window
            purchases = sub_df[sub_df["event_type"] == "purchase"].groupby("customer_id")["event_id"].count()
            results[f"purchases_{w}d"] = purchases.reindex(results.index, fill_value=0)

            # Support contacts in window
            support = sub_df[sub_df["event_type"] == "support_contact"].groupby("customer_id")["event_id"].count()
            results[f"support_{w}d"] = support.reindex(results.index, fill_value=0)

        return results


# ===========================================================================
# Session & Behavioural Feature Builder
# ===========================================================================

class SessionFeatureBuilder:
    """Derive session and explicit domain features per customer.

    Computes:
    - avg_order_value
    - avg_purchase_interval_days
    - days_since_last_purchase
    - session_frequency
    - support_ticket_count
    - coupon_usage_ratio
    - wishlist_click_count
    - cart_abandonment_rate
    - avg_pages_per_session
    - avg_session_duration_sec
    - device_switch_count
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def build(self, df: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> pd.DataFrame:
        """Compute customer behavioural features.

        Parameters
        ----------
        df : pd.DataFrame
        reference_date : pd.Timestamp | None

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by `customer_id`.
        """
        if reference_date is None:
            reference_date = df["event_timestamp"].max()

        cust_ids = df["customer_id"].unique()
        res = pd.DataFrame(index=cust_ids)
        res.index.name = "customer_id"

        # 1. Support ticket count
        supp = df[df["event_type"] == "support_contact"].groupby("customer_id")["event_id"].count()
        res["support_ticket_count"] = supp.reindex(res.index, fill_value=0)

        # 2. Wishlist click count
        wish = df[df["event_type"] == "wishlist_click"].groupby("customer_id")["event_id"].count()
        res["wishlist_click_count"] = wish.reindex(res.index, fill_value=0)

        # 3. Device switch count
        dev_sw = df.groupby("customer_id")["device_type"].nunique()
        res["device_switch_count"] = dev_sw.reindex(res.index, fill_value=1)

        # 4. Purchases and Order Values
        purch_df = df[df["event_type"] == "purchase"]
        order_stats = purch_df.groupby("customer_id").agg(
            purchase_count=("event_id", "count"),
            avg_order_value=("order_value", "mean"),
            last_purchase_time=("event_timestamp", "max"),
            coupon_purchases=("coupon_used", lambda x: int(x.sum())),
        )

        res = res.join(order_stats, how="left")
        res["purchase_count"] = res["purchase_count"].fillna(0).astype(int)
        res["avg_order_value"] = res["avg_order_value"].fillna(0.0)

        # Days since last purchase
        res["days_since_last_purchase"] = (reference_date - res["last_purchase_time"]).dt.total_seconds() / (24 * 3600)
        res["days_since_last_purchase"] = res["days_since_last_purchase"].fillna(999.0).clip(lower=0.0)
        res.drop(columns=["last_purchase_time"], inplace=True)

        # Coupon usage ratio
        res["coupon_usage_ratio"] = np.where(
            res["purchase_count"] > 0,
            res["coupon_purchases"] / res["purchase_count"],
            0.0
        )
        res.drop(columns=["coupon_purchases"], inplace=True)

        # 5. Purchase Interval
        def calc_avg_interval(cust_df: pd.DataFrame) -> float:
            p_times = cust_df[cust_df["event_type"] == "purchase"]["event_timestamp"].sort_values()
            if len(p_times) < 2:
                return 0.0
            intervals = p_times.diff().dt.total_seconds().dropna() / (24 * 3600)
            return float(intervals.mean())

        interval_dict = {cid: calc_avg_interval(cdf) for cid, cdf in df.groupby("customer_id")}
        res["avg_purchase_interval_days"] = pd.Series(interval_dict).reindex(res.index, fill_value=0.0)

        # 6. Cart Abandonment Rate: (add_to_cart - purchase) / add_to_cart
        cart_adds = df[df["event_type"] == "add_to_cart"].groupby("customer_id")["event_id"].count()
        res["add_to_cart_count"] = cart_adds.reindex(res.index, fill_value=0)

        res["cart_abandonment_rate"] = np.where(
            res["add_to_cart_count"] > 0,
            (res["add_to_cart_count"] - res["purchase_count"]).clip(lower=0) / res["add_to_cart_count"],
            0.0
        )
        res.drop(columns=["add_to_cart_count"], inplace=True)

        # 7. Session Stats
        sess_summary = df.groupby(["customer_id", "session_id"]).agg(
            sess_start=("event_timestamp", "min"),
            sess_end=("event_timestamp", "max"),
            page_views=("event_type", lambda x: int((x == "page_view").sum())),
        ).reset_index()

        sess_summary["duration_sec"] = (sess_summary["sess_end"] - sess_summary["sess_start"]).dt.total_seconds()

        cust_sess_stats = sess_summary.groupby("customer_id").agg(
            total_sessions=("session_id", "count"),
            avg_pages_per_session=("page_views", "mean"),
            avg_session_duration_sec=("duration_sec", "mean"),
            first_sess=("sess_start", "min"),
            last_sess=("sess_end", "max"),
        )

        res = res.join(cust_sess_stats, how="left")

        res["total_sessions"] = res["total_sessions"].fillna(1).astype(int)
        res["avg_pages_per_session"] = res["avg_pages_per_session"].fillna(1.0)
        res["avg_session_duration_sec"] = res["avg_session_duration_sec"].fillna(0.0)

        # Session Frequency (sessions per active day)
        active_span_days = ((res["last_sess"] - res["first_sess"]).dt.total_seconds() / (24 * 3600)).clip(lower=1.0)
        res["session_frequency"] = res["total_sessions"] / active_span_days

        res.drop(columns=["first_sess", "last_sess", "total_sessions"], inplace=True)

        return res


# ===========================================================================
# Feature Engineering Pipeline
# ===========================================================================

class FeatureEngineeringPipeline:
    """Orchestrate customer feature generation.

    Parameters
    ----------
    config : dict[str, Any]
        Full project configuration dictionary.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.root: Path = get_project_root()
        self.rfm_builder = RFMBuilder(config)
        self.rolling_builder = RollingFeatureBuilder(config)
        self.session_builder = SessionFeatureBuilder(config)
        self.features_dir: Path = self.root / config["paths"]["data_features"]
        ensure_dir(self.features_dir)
        logger.info("FeatureEngineeringPipeline ready.")

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the feature engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned event DataFrame from PreprocessingPipeline.

        Returns
        -------
        pd.DataFrame
            Customer feature DataFrame saved to `data/features/customer_features.parquet`.
        """
        logger.info("Building customer features for %d events...", len(df))
        ref_date = df["event_timestamp"].max()

        rfm_df = self.rfm_builder.compute(df, reference_date=ref_date)
        rolling_df = self.rolling_builder.build(df, reference_date=ref_date)
        session_df = self.session_builder.build(df, reference_date=ref_date)

        # Target label per customer
        target_df = df.groupby("customer_id")["churn"].first().to_frame()

        # Merge all feature tables
        features = rfm_df.join(rolling_df, how="inner").join(session_df, how="inner").join(target_df, how="inner")

        out_path = self.features_dir / "customer_features.parquet"
        features.to_parquet(out_path)

        logger.info(
            "Saved customer features → %s | Shape: %d customers × %d features",
            out_path, *features.shape
        )
        return features
