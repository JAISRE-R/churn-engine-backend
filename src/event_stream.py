"""
churn-engine/src/event_stream.py
==================================
Live Event Stream Ingestion Layer and Customer State Manager.

Responsibilities
----------------
- Simulate real-time customer behavioural events using realistic personas.
- Maintain an in-memory thread-safe event buffer with a configurable max size.
- Update customer state and calculate features matching training schema.
- Coordinate simulator, buffer, and state manager via StreamManager.
"""

from __future__ import annotations

import logging
import random
import time
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generator, Iterator

import numpy as np
import pandas as pd

from src.utils import get_project_root

logger = logging.getLogger(__name__)

# ===========================================================================
# Persona Configurations (Mirrors generate_data.py)
# ===========================================================================
PERSONAS: dict[str, dict[str, Any]] = {
    "loyal": {
        "weight": 0.25,
        "avg_sessions": (15, 35),
        "order_prob": 0.45,
        "avg_order_val": (60.0, 180.0),
        "coupon_prob": 0.20,
        "support_prob": 0.05,
        "wishlist_prob": 0.35,
        "inactivity_days": (1, 15),
        "base_churn_prob": 0.05,
    },
    "high_value": {
        "weight": 0.15,
        "avg_sessions": (10, 25),
        "order_prob": 0.50,
        "avg_order_val": (200.0, 650.0),
        "coupon_prob": 0.10,
        "support_prob": 0.08,
        "wishlist_prob": 0.40,
        "inactivity_days": (2, 20),
        "base_churn_prob": 0.05,
    },
    "window_shopper": {
        "weight": 0.20,
        "avg_sessions": (8, 20),
        "order_prob": 0.08,
        "avg_order_val": (25.0, 75.0),
        "coupon_prob": 0.15,
        "support_prob": 0.05,
        "wishlist_prob": 0.50,
        "inactivity_days": (10, 45),
        "base_churn_prob": 0.45,
    },
    "discount_hunter": {
        "weight": 0.15,
        "avg_sessions": (6, 18),
        "order_prob": 0.30,
        "avg_order_val": (20.0, 60.0),
        "coupon_prob": 0.85,
        "support_prob": 0.10,
        "wishlist_prob": 0.20,
        "inactivity_days": (15, 50),
        "base_churn_prob": 0.40,
    },
    "at_risk": {
        "weight": 0.15,
        "avg_sessions": (3, 10),
        "order_prob": 0.10,
        "avg_order_val": (30.0, 100.0),
        "coupon_prob": 0.30,
        "support_prob": 0.45,
        "wishlist_prob": 0.10,
        "inactivity_days": (60, 120),
        "base_churn_prob": 0.85,
    },
    "returning": {
        "weight": 0.10,
        "avg_sessions": (5, 14),
        "order_prob": 0.25,
        "avg_order_val": (40.0, 120.0),
        "coupon_prob": 0.25,
        "support_prob": 0.10,
        "wishlist_prob": 0.25,
        "inactivity_days": (5, 30),
        "base_churn_prob": 0.15,
    },
}

COUNTRIES = ["US", "CA", "GB", "DE", "FR", "AU", "JP"]
DEVICES = ["desktop", "mobile", "tablet"]

# ===========================================================================
# Event Data Model
# ===========================================================================

@dataclass
class CustomerEvent:
    """Represents a single customer behavioural event.

    Matches the raw dataset schema perfectly.
    """
    event_id: str
    customer_id: str
    event_timestamp: datetime
    event_type: str
    session_id: str
    product_id: str | None = None
    order_value: float | None = None
    coupon_used: bool = False
    page_url: str = ""
    device_type: str = ""
    country: str = ""
    customer_age_days: int = 0
    churn: int = 0

    @property
    def timestamp(self) -> datetime:
        """Alias for event_timestamp for backward compatibility."""
        return self.event_timestamp


# ===========================================================================
# Event Buffer
# ===========================================================================

class EventBuffer:
    """Thread-safe circular buffer for holding recent customer events."""

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._buffer: deque[CustomerEvent] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        logger.debug("EventBuffer initialised | max_size=%d", max_size)

    def push(self, event: CustomerEvent) -> None:
        """Add an event thread-safely."""
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self.max_size * 0.95:
                logger.warning("EventBuffer approaching capacity! size=%d", len(self._buffer))

    def drain(self, n: int | None = None) -> list[CustomerEvent]:
        """Remove and return up to ``n`` events from the buffer."""
        with self._lock:
            drained = []
            limit = n if n is not None else len(self._buffer)
            for _ in range(min(limit, len(self._buffer))):
                drained.append(self._buffer.popleft())
            return drained

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the current buffer contents to a pandas DataFrame."""
        with self._lock:
            records = []
            for ev in self._buffer:
                records.append({
                    "event_id": ev.event_id,
                    "customer_id": ev.customer_id,
                    "event_timestamp": ev.event_timestamp,
                    "event_type": ev.event_type,
                    "session_id": ev.session_id,
                    "product_id": ev.product_id,
                    "order_value": ev.order_value,
                    "coupon_used": ev.coupon_used,
                    "page_url": ev.page_url,
                    "device_type": ev.device_type,
                    "country": ev.country,
                    "customer_age_days": ev.customer_age_days,
                    "churn": ev.churn,
                })
            return pd.DataFrame(records)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


# ===========================================================================
# Event Simulator
# ===========================================================================

class EventSimulator:
    """Generate synthetic customer events matching the personas schema."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        stream_cfg = config.get("event_stream", {})
        self.rate_hz: float = float(stream_cfg.get("event_frequency", 2.0))
        self.simulation_speed: float = float(stream_cfg.get("simulation_speed", 1.0))
        self.seed: int = int(stream_cfg.get("random_seed", 42))

        # Set up deterministic state for simulation
        self._rng = random.Random(self.seed)
        self._event_counter = 0

        # Load customer pool
        self._init_customer_pool()

        # Track active customer sessions
        self.active_sessions: dict[str, dict[str, Any]] = {}

    def _init_customer_pool(self) -> None:
        """Initialise pool of customers from raw_events.csv if available."""
        self.customer_pool: list[dict[str, Any]] = []
        root = get_project_root()
        raw_path = root / self.config["paths"]["data_raw"] / self.config["data"].get("raw_filename", "raw_events.csv")

        if raw_path.exists():
            try:
                df = pd.read_csv(raw_path)
                # Group by customer_id to extract realistic personas, country, age
                grouped = df.groupby("customer_id").agg(
                    country=("country", "first"),
                    device_type=("device_type", "first"),
                    customer_age_days=("customer_age_days", "first"),
                    churn=("churn", "first")
                ).reset_index()

                # Recompute deterministic personas based on seed 42
                num_custs = len(grouped)
                p_names = list(PERSONAS.keys())
                p_weights = [PERSONAS[p]["weight"] for p in p_names]

                # Reuse same seed rules
                temp_rng = random.Random(42)
                personas = temp_rng.choices(p_names, weights=p_weights, k=num_custs)

                for idx, row in grouped.iterrows():
                    self.customer_pool.append({
                        "customer_id": row["customer_id"],
                        "country": row["country"],
                        "device_type": row["device_type"],
                        "customer_age_days": int(row["customer_age_days"]),
                        "churn": int(row["churn"]),
                        "persona": personas[idx],
                    })
                logger.info("Loaded customer pool of %d profiles from %s", len(self.customer_pool), raw_path)
                return
            except Exception as e:
                logger.error("Failed to load customer pool from %s: %s. Generating fallback...", raw_path, e)

        # Fallback customer pool generation
        logger.info("Generating fallback customer pool of 100 profiles...")
        p_names = list(PERSONAS.keys())
        p_weights = [PERSONAS[p]["weight"] for p in p_names]
        temp_rng = random.Random(self.seed)
        personas = temp_rng.choices(p_names, weights=p_weights, k=100)

        for i in range(100):
            self.customer_pool.append({
                "customer_id": f"C-{1000 + i:04d}",
                "country": temp_rng.choice(COUNTRIES),
                "device_type": temp_rng.choice(DEVICES),
                "customer_age_days": temp_rng.randint(30, 730),
                "churn": temp_rng.choice([0, 1]),
                "persona": personas[i],
            })

    def generate_event(self) -> CustomerEvent:
        """Generate a single customer event."""
        self._event_counter += 1
        # Pick a customer
        cust = self._rng.choice(self.customer_pool)
        cid = cust["customer_id"]
        persona = cust["persona"]
        p_cfg = PERSONAS[persona]

        # Check session status
        session_info = self.active_sessions.get(cid)
        if session_info is None:
            # Start new session
            session_id = f"S-{uuid.uuid4().hex[:8]}"
            event_type = "page_view"
            event_count = 1
            max_events = self._rng.randint(3, 12)
            self.active_sessions[cid] = {
                "session_id": session_id,
                "event_count": event_count,
                "max_events": max_events,
            }
        else:
            session_id = session_info["session_id"]
            session_info["event_count"] += 1
            event_count = session_info["event_count"]
            max_events = session_info["max_events"]

            if event_count >= max_events:
                event_type = "session_end"
                del self.active_sessions[cid]
            else:
                # Random type based on persona probabilities
                r = self._rng.random()
                if r < 0.35:
                    event_type = "product_click"
                elif r < 0.55:
                    event_type = "add_to_cart"
                elif r < 0.65:
                    event_type = "search"
                elif r < 0.65 + p_cfg["wishlist_prob"] * 0.2:
                    event_type = "wishlist_click"
                elif r < 0.65 + p_cfg["wishlist_prob"] * 0.2 + p_cfg["support_prob"] * 0.2:
                    event_type = "support_contact"
                elif self._rng.random() < p_cfg["order_prob"]:
                    event_type = "purchase"
                else:
                    event_type = "remove_from_cart"

        # Special event fields
        order_value = None
        coupon_used = False
        if event_type == "purchase":
            order_value = round(self._rng.uniform(*p_cfg["avg_order_val"]), 2)
            coupon_used = self._rng.random() < p_cfg["coupon_prob"]

        product_id = None
        if event_type in ["product_click", "add_to_cart", "purchase", "wishlist_click"]:
            product_id = f"P-{self._rng.randint(100, 999)}"

        return CustomerEvent(
            event_id=f"evt-sim-{self._event_counter:06d}",
            customer_id=cid,
            event_timestamp=datetime.utcnow(),
            event_type=event_type,
            session_id=session_id,
            product_id=product_id,
            order_value=order_value,
            coupon_used=coupon_used,
            page_url=f"https://shop.example.com/{event_type}",
            device_type=cust["device_type"],
            country=cust["country"],
            customer_age_days=cust["customer_age_days"],
            churn=cust["churn"],
        )

    def stream(self) -> Generator[CustomerEvent, None, None]:
        """Continuously yield simulated events."""
        while True:
            # Yield event
            yield self.generate_event()
            # Calculate sleep window
            rate = self.rate_hz * self.simulation_speed
            sleep_sec = 1.0 / rate if rate > 0 else 0.5
            time.sleep(sleep_sec)

    def burst(self, n: int = 10) -> list[CustomerEvent]:
        """Return a burst of n simulated events instantly."""
        return [self.generate_event() for _ in range(n)]


# ===========================================================================
# Customer State Manager
# ===========================================================================

class CustomerStateManager:
    """Consumes incoming events and calculates customer-level features."""

    FEATURE_ORDER = [
        "frequency", "monetary_total", "recency_days", "monetary_log",
        "events_7d", "purchases_7d", "support_7d",
        "events_14d", "purchases_14d", "support_14d",
        "events_30d", "purchases_30d", "support_30d",
        "support_ticket_count", "wishlist_click_count", "device_switch_count",
        "purchase_count", "avg_order_value", "days_since_last_purchase",
        "coupon_usage_ratio", "avg_purchase_interval_days", "cart_abandonment_rate",
        "avg_pages_per_session", "avg_session_duration_sec", "session_frequency"
    ]

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.customer_history: dict[str, list[dict[str, Any]]] = {}
        self.loaded_event_count = 0
        self._lock = threading.Lock()
        self._init_history_from_raw()

    def _init_history_from_raw(self) -> None:
        """Pre-populate the customer history dictionary from raw_events.csv."""
        root = get_project_root()
        raw_path = root / self.config["paths"]["data_raw"] / self.config["data"].get("raw_filename", "raw_events.csv")
        if raw_path.exists():
            try:
                df = pd.read_csv(raw_path)
                self.loaded_event_count = len(df)
                df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
                # Convert timestamps to timezone-naive
                df["event_timestamp"] = df["event_timestamp"].dt.tz_localize(None)

                # Convert to records grouped by customer
                for cid, group in df.groupby("customer_id"):
                    self.customer_history[cid] = group.to_dict("records")
                logger.info("StateManager pre-populated with history for %d customers", len(self.customer_history))
            except Exception as e:
                logger.error("Failed to pre-populate history from %s: %s", raw_path, e)

    def update_state(self, event: CustomerEvent) -> None:
        """Add event to customer's history."""
        with self._lock:
            cid = event.customer_id
            record = {
                "event_id": event.event_id,
                "customer_id": cid,
                "event_timestamp": pd.to_datetime(event.event_timestamp).replace(tzinfo=None),
                "event_type": event.event_type,
                "session_id": event.session_id,
                "product_id": event.product_id,
                "order_value": event.order_value,
                "coupon_used": event.coupon_used,
                "page_url": event.page_url,
                "device_type": event.device_type,
                "country": event.country,
                "customer_age_days": event.customer_age_days,
                "churn": event.churn,
            }
            self.customer_history.setdefault(cid, []).append(record)
            # Keep history capped to last 1000 events to manage memory
            if len(self.customer_history[cid]) > 1000:
                self.customer_history[cid] = self.customer_history[cid][-1000:]

    def get_features(self, customer_id: str, reference_date: datetime | None = None) -> pd.DataFrame:
        """Calculate and return features aligned to the training schema."""
        with self._lock:
            history = self.customer_history.get(customer_id, [])

        if not history:
            # Return safe baseline values for new/insufficient history customers
            defaults = {f: 0.0 for f in self.FEATURE_ORDER}
            defaults["days_since_last_purchase"] = 999.0
            defaults["device_switch_count"] = 1.0
            defaults["avg_pages_per_session"] = 1.0
            df_feat = pd.DataFrame([defaults], index=[customer_id])
            df_feat.index.name = "customer_id"
            return df_feat[self.FEATURE_ORDER]

        df = pd.DataFrame(history)
        # Ensure correct types and naive timestamps
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"]).dt.tz_localize(None)

        if reference_date is None:
            ref_date = df["event_timestamp"].max()
        else:
            ref_date = pd.to_datetime(reference_date).replace(tzinfo=None)

        # Feature calculation dictionary
        feats = {}

        # 1. RFM
        feats["frequency"] = len(df)
        total_monetary = float(df[df["event_type"] == "purchase"]["order_value"].fillna(0.0).sum())
        feats["monetary_total"] = total_monetary
        feats["monetary_log"] = float(np.log1p(total_monetary))

        rec_sec = (ref_date - df["event_timestamp"].max()).total_seconds()
        feats["recency_days"] = max(0.0, rec_sec / (24 * 3600))

        # 2. Rolling window counts
        for w in [7, 14, 30]:
            window_start = ref_date - pd.Timedelta(days=w)
            sub_df = df[df["event_timestamp"] >= window_start]
            feats[f"events_{w}d"] = len(sub_df)
            feats[f"purchases_{w}d"] = len(sub_df[sub_df["event_type"] == "purchase"])
            feats[f"support_{w}d"] = len(sub_df[sub_df["event_type"] == "support_contact"])

        # 3. Simple event summaries
        purch_df = df[df["event_type"] == "purchase"]
        feats["support_ticket_count"] = len(df[df["event_type"] == "support_contact"])
        feats["wishlist_click_count"] = len(df[df["event_type"] == "wishlist_click"])
        feats["device_switch_count"] = df["device_type"].nunique()
        feats["purchase_count"] = len(purch_df)

        # 4. Purchases and Order Values
        if len(purch_df) > 0:
            feats["avg_order_value"] = float(purch_df["order_value"].fillna(0.0).mean())
            last_purch = purch_df["event_timestamp"].max()
            purch_rec_sec = (ref_date - last_purch).total_seconds()
            feats["days_since_last_purchase"] = max(0.0, purch_rec_sec / (24 * 3600))

            coupon_count = int(purch_df["coupon_used"].fillna(False).sum())
            feats["coupon_usage_ratio"] = coupon_count / len(purch_df)
        else:
            feats["avg_order_value"] = 0.0
            feats["days_since_last_purchase"] = 999.0
            feats["coupon_usage_ratio"] = 0.0

        # 5. Purchase intervals
        if len(purch_df) >= 2:
            p_times = purch_df["event_timestamp"].sort_values()
            feats["avg_purchase_interval_days"] = float(p_times.diff().dt.total_seconds().dropna().mean() / (24 * 3600))
        else:
            feats["avg_purchase_interval_days"] = 0.0

        # 6. Cart Abandonment Rate
        cart_adds = len(df[df["event_type"] == "add_to_cart"])
        if cart_adds > 0:
            feats["cart_abandonment_rate"] = max(0.0, (cart_adds - len(purch_df)) / cart_adds)
        else:
            feats["cart_abandonment_rate"] = 0.0

        # 7. Session Stats
        sessions = df["session_id"].nunique()
        page_views = len(df[df["event_type"] == "page_view"])
        feats["avg_pages_per_session"] = page_views / sessions if sessions > 0 else 1.0

        # Durations per session
        durations = []
        for sid, s_group in df.groupby("session_id"):
            dur = (s_group["event_timestamp"].max() - s_group["event_timestamp"].min()).total_seconds()
            durations.append(dur)
        feats["avg_session_duration_sec"] = float(np.mean(durations)) if durations else 0.0

        # Session Frequency
        first_sess = df["event_timestamp"].min()
        last_sess = df["event_timestamp"].max()
        active_days = max(1.0, (last_sess - first_sess).total_seconds() / (24 * 3600))
        feats["session_frequency"] = sessions / active_days

        # Compile DataFrame aligned to features list
        df_feat = pd.DataFrame([feats], index=[customer_id])
        df_feat.index.name = "customer_id"
        return df_feat[self.FEATURE_ORDER]


# ===========================================================================
# Stream Manager
# ===========================================================================

class StreamManager:
    """Manages safe background threading and updates statistics."""

    def __init__(self, config: dict[str, Any], state_manager: CustomerStateManager | None = None) -> None:
        self.config = config
        stream_cfg = config.get("event_stream", {})
        self.simulator = EventSimulator(config)
        self.buffer = EventBuffer(max_size=stream_cfg.get("max_recent_events", 1000))
        self.state_manager = state_manager or CustomerStateManager(config)

        # Import Predictor, DecisionGate, Recommender, and OverrideEngine here to avoid early circular dependencies
        from src.predictor import ChurnPredictor
        from src.decision_gate import DecisionGate
        from src.recommender import BaselineRecommender
        from src.override_engine import OverrideEngine

        self.predictor = ChurnPredictor(config)
        self.decision_gate = DecisionGate(config)
        self.recommender = BaselineRecommender(config)
        self.override_engine = OverrideEngine(config)

        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Stats counters
        self._start_time: float | None = None
        self._total_events = 0
        self._active_customers: set[str] = set()
        self._latest_event: CustomerEvent | None = None

        # Real-time inference trackers
        self.latest_predictions: dict[str, dict[str, Any]] = {}
        self.prediction_history: dict[str, deque[dict[str, Any]]] = {}
        self.latest_recommendations: dict[str, dict[str, Any]] = {}
        self.latest_decisions: dict[str, dict[str, Any]] = {}
        self.risk_history_size: int = int(stream_cfg.get("risk_history_size", 100))

    def start(self) -> None:
        """Start the background simulation loop thread."""
        with self._lock:
            if self._running:
                logger.warning("StreamManager is already running.")
                return
            self._running = True
            self._start_time = time.time()
            # Clear stats for new run
            self._total_events = 0
            self._active_customers.clear()
            self._latest_event = None
            self.latest_predictions.clear()
            self.prediction_history.clear()
            self.latest_recommendations.clear()
            self.latest_decisions.clear()
            self.decision_gate.clear()
            self.override_engine.clear()

            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info("StreamManager background thread started.")

    def _run_loop(self) -> None:
        """Background thread loop consuming simulator stream."""
        try:
            for event in self.simulator.stream():
                if not self._running:
                    break

                # Push to buffer
                self.buffer.push(event)

                # Push to Customer State
                self.state_manager.update_state(event)

                cid = event.customer_id
                
                # Fetch features
                features = self.state_manager.get_features(cid, reference_date=event.event_timestamp)
                
                # 1. Churn Predictor
                pred_res = self.predictor.predict(cid, features)
                prob = pred_res["churn_probability"]
                risk_lvl = pred_res["risk_level"]

                # 2. Decision Gate (extract previous probability first)
                with self._lock:
                    hist_deq = self.prediction_history.get(cid)
                    prev_prob = hist_deq[-1]["churn_probability"] if (hist_deq and len(hist_deq) > 0) else None

                decision_res = self.decision_gate.decide(
                    customer_id=cid,
                    churn_probability=prob,
                    previous_probability=prev_prob,
                    risk_level=risk_lvl
                )

                # 3. Baseline Recommender (independent of churn status)
                cust_events = self.state_manager.customer_history.get(cid, [])
                recommends = self.recommender.recommend(cid, cust_events)

                # 4. Override Engine (apply targeted interventions)
                final_recs = self.override_engine.process(
                    baseline_recs=recommends,
                    decision_res=decision_res,
                    customer_events=cust_events
                )

                # Update stats and trackers thread-safely
                with self._lock:
                    self._total_events += 1
                    self._active_customers.add(cid)
                    self._latest_event = event

                    # Record outputs
                    self.latest_predictions[cid] = pred_res
                    self.latest_decisions[cid] = decision_res
                    self.latest_recommendations[cid] = final_recs

                    # Record history deque
                    hist = self.prediction_history.setdefault(
                        cid, deque(maxlen=self.risk_history_size)
                    )
                    hist.append({
                        "timestamp": pred_res["timestamp"],
                        "churn_probability": prob,
                        "threshold": pred_res["threshold"],
                        "risk_level": risk_lvl,
                        "intervention_status": pred_res["intervention_status"]
                    })

        except Exception as e:
            logger.error("Error in StreamManager run loop: %s", e)
            self._running = False

    def stop(self) -> None:
        """Stop the background thread."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            logger.info("Stopping StreamManager background thread...")

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
            logger.info("StreamManager background thread stopped successfully.")

    def get_recent_events(self, n: int = 50) -> pd.DataFrame:
        """Get the most recent n events as a DataFrame."""
        df = self.buffer.to_dataframe()
        if len(df) == 0:
            return pd.DataFrame(columns=[
                "event_id", "customer_id", "event_timestamp", "event_type",
                "session_id", "product_id", "order_value", "coupon_used",
                "page_url", "device_type", "country", "customer_age_days", "churn"
            ])
        return df.tail(n)

    def get_stats(self) -> dict[str, Any]:
        """Expose current stats dictionary."""
        with self._lock:
            elapsed = time.time() - self._start_time if self._start_time else 0.0
            eps = self._total_events / elapsed if elapsed > 0 else 0.0
            return {
                "total_events": self._total_events,
                "events_per_second": round(eps, 2),
                "active_customers_count": len(self._active_customers),
                "latest_event": self._latest_event,
                "elapsed_seconds": round(elapsed, 1),
                "status": "running" if self._running else "stopped",
            }


