"""
churn-engine/src/generate_data.py
===================================
Behaviour-Driven Synthetic Data Generator.

Generates realistic e-commerce customer behavioural event logs based on 6
distinct customer personas. Ground-truth churn labels are computed using
deterministic and probabilistic business rules tied to customer behaviour.

Personas
--------
1. Loyal Customer       : High frequency, low recency gap, high wishlist clicks, low support contacts.
2. High Value Customer  : High order values ($150-$500), low cart abandonments, premium devices.
3. Window Shopper       : High page views/clicks, high cart abandonments, zero/low purchases.
4. Discount Hunter      : High coupon usage ratio, low average order values.
5. At-Risk Customer     : High recency gap (>60d), high support tickets, rising cart abandonments.
6. Returning Customer   : Re-engaged after past inactivity, moderate activity.
"""

import sys
from pathlib import Path

# Ensure project root is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.utils import ensure_dir, get_project_root, load_config, setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persona Configurations
# ---------------------------------------------------------------------------
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
EVENT_TYPES = [
    "page_view", "product_click", "add_to_cart",
    "remove_from_cart", "purchase", "search",
    "session_end", "support_contact", "wishlist_click"
]


def generate_synthetic_dataset(
    num_customers: int = 500,
    start_date: datetime | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a behaviour-driven synthetic event log DataFrame.

    Parameters
    ----------
    num_customers : int
        Number of synthetic customers to generate.
    start_date : datetime | None
        Snapshot anchor date. Defaults to 90 days prior to utcnow.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Event-level raw DataFrame.
    """
    random.seed(seed)
    np.random.seed(seed)

    if start_date is None:
        anchor_end = datetime.utcnow()
        start_date = anchor_end - timedelta(days=90)
    else:
        anchor_end = start_date + timedelta(days=90)

    events: list[dict[str, Any]] = []

    # Assign personas
    persona_names = list(PERSONAS.keys())
    persona_weights = [PERSONAS[p]["weight"] for p in persona_names]
    customer_personas = random.choices(persona_names, weights=persona_weights, k=num_customers)

    for i in range(num_customers):
        customer_id = f"C-{1000 + i:04d}"
        persona = customer_personas[i]
        p_cfg = PERSONAS[persona]

        country = random.choice(COUNTRIES)
        primary_device = random.choice(DEVICES)
        customer_age_days = random.randint(30, 730)

        # Inactivity gap determines last event timestamp
        inactivity_days = random.randint(*p_cfg["inactivity_days"])
        last_event_time = anchor_end - timedelta(days=inactivity_days)

        # Generate sessions
        num_sessions = random.randint(*p_cfg["avg_sessions"])
        session_times = sorted([
            start_date + timedelta(
                seconds=random.randint(0, int((last_event_time - start_date).total_seconds()))
            )
            for _ in range(num_sessions)
        ]) if last_event_time > start_date else [last_event_time]

        support_contacts_count = 0
        total_purchases_count = 0
        cart_add_count = 0

        for sess_idx, s_time in enumerate(session_times):
            session_id = f"S-{uuid.uuid4().hex[:8]}"

            # Device switching
            device = primary_device if random.random() > 0.15 else random.choice(DEVICES)

            # Events within session
            num_events_in_sess = random.randint(3, 12)
            cur_time = s_time

            for ev_idx in range(num_events_in_sess):
                cur_time += timedelta(seconds=random.randint(5, 180))

                # Event type logic
                if ev_idx == 0:
                    event_type = "page_view"
                elif ev_idx == num_events_in_sess - 1:
                    event_type = "session_end"
                else:
                    r = random.random()
                    if r < 0.35:
                        event_type = "product_click"
                    elif r < 0.55:
                        event_type = "add_to_cart"
                        cart_add_count += 1
                    elif r < 0.65:
                        event_type = "search"
                    elif r < 0.65 + p_cfg["wishlist_prob"] * 0.2:
                        event_type = "wishlist_click"
                    elif r < 0.65 + p_cfg["wishlist_prob"] * 0.2 + p_cfg["support_prob"] * 0.2:
                        event_type = "support_contact"
                        support_contacts_count += 1
                    elif random.random() < p_cfg["order_prob"] and cart_add_count > total_purchases_count:
                        event_type = "purchase"
                        total_purchases_count += 1
                    else:
                        event_type = "remove_from_cart"

                order_value = None
                coupon_used = False
                if event_type == "purchase":
                    order_value = round(random.uniform(*p_cfg["avg_order_val"]), 2)
                    coupon_used = random.random() < p_cfg["coupon_prob"]

                events.append({
                    "event_id": f"evt-{len(events)+1:06d}",
                    "customer_id": customer_id,
                    "persona": persona,
                    "event_timestamp": cur_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "event_type": event_type,
                    "session_id": session_id,
                    "product_id": f"P-{random.randint(100, 999)}" if event_type in ["product_click", "add_to_cart", "purchase", "wishlist_click"] else None,
                    "order_value": order_value,
                    "coupon_used": coupon_used,
                    "page_url": f"https://shop.example.com/{event_type}",
                    "device_type": device,
                    "country": country,
                    "customer_age_days": customer_age_days,
                    "inactivity_days": inactivity_days,
                    "support_contacts_count": support_contacts_count,
                    "total_purchases_count": total_purchases_count,
                })

    df = pd.DataFrame(events)

    # ---------------------------------------------------------------------------
    # Business-Rule Driven Churn Labeling
    # ---------------------------------------------------------------------------
    # Calculate customer-level risk signals to assign ground-truth churn
    cust_summary = df.groupby("customer_id").agg(
        inactivity_days=("inactivity_days", "first"),
        support_contacts=("support_contacts_count", "max"),
        total_purchases=("total_purchases_count", "max"),
        persona=("persona", "first"),
    ).reset_index()

    def determine_churn(row: pd.Series) -> int:
        p = row["persona"]
        base_p = PERSONAS[p]["base_churn_prob"]
        # Rule adjustments
        if row["inactivity_days"] > 60:
            base_p += 0.35
        if row["support_contacts"] >= 3:
            base_p += 0.25
        if row["total_purchases"] == 0:
            base_p += 0.20
        churn_prob = min(0.95, max(0.02, base_p))
        return 1 if random.random() < churn_prob else 0

    cust_summary["churn"] = cust_summary.apply(determine_churn, axis=1)

    # Merge churn label back onto event log
    df = df.merge(cust_summary[["customer_id", "churn"]], on="customer_id", how="left")

    # Drop internal helper columns
    df.drop(columns=["persona", "inactivity_days", "support_contacts_count", "total_purchases_count"], inplace=True)

    logger.info(
        "Generated synthetic dataset: %d events across %d customers | Churn rate: %.2f%%",
        len(df), num_customers, (df.groupby("customer_id")["churn"].first().mean() * 100)
    )
    return df


def main() -> None:
    """Generate and save the synthetic raw event dataset."""
    config = load_config()
    setup_logging(config)

    raw_dir = get_project_root() / config["paths"]["data_raw"]
    ensure_dir(raw_dir)

    raw_filename = config["data"].get("raw_filename", "raw_events.csv")
    output_path = raw_dir / raw_filename

    df = generate_synthetic_dataset(num_customers=500, seed=42)
    df.to_csv(output_path, index=False)
    logger.info("Saved synthetic raw dataset to: %s", output_path)


if __name__ == "__main__":
    main()
