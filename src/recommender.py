"""
churn-engine/src/recommender.py
=================================
Baseline Recommendation Engine.

Responsibilities
----------------
- Implement the e-commerce store's NORMAL recommendation strategy.
- Generate product recommendations based purely on customer/product interactions.
- Ensure recommendation logic operates entirely independently of churn risk scores.
- Expose a standard RecommendationResult contract.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def get_product_category(product_id: str) -> str:
    """Deterministically map a product ID string to a category name."""
    if not product_id:
        return "General"
    try:
        # Extract digits
        digits = "".join(c for c in product_id if c.isdigit())
        if not digits:
            return "General"
        val = int(digits)
        if val <= 300:
            return "Electronics"
        elif val <= 500:
            return "Fashion & Apparel"
        elif val <= 700:
            return "Home & Living"
        elif val <= 900:
            return "Beauty & Health"
        else:
            return "Sports & Outdoors"
    except Exception:
        return "General"


class BaselineRecommender:
    """Generates normal product recommendations for customers based on behavior."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        # Safe default trending products
        self.trending_products = [
            {"product_id": "P-101", "category": "Electronics", "reason": "Trending Product", "score": 0.85},
            {"product_id": "P-305", "category": "Fashion & Apparel", "reason": "Trending Product", "score": 0.82},
            {"product_id": "P-512", "category": "Home & Living", "reason": "Trending Product", "score": 0.79},
            {"product_id": "P-722", "category": "Beauty & Health", "reason": "Trending Product", "score": 0.75},
            {"product_id": "P-910", "category": "Sports & Outdoors", "reason": "Trending Product", "score": 0.70},
        ]

    def recommend(self, customer_id: str, customer_events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Generate a ranked list of recommendations for a customer.

        Parameters
        ----------
        customer_id : str
            Target customer identifier.
        customer_events : list[dict[str, Any]] | None
            Historical event dictionary logs for this customer.

        Returns
        -------
        dict[str, Any]
            RecommendationResult container dictionary.
        """
        recs = []
        seen_products = set()

        if customer_events:
            # 1. Extract products recently interacted with
            interacted_events = [
                ev for ev in customer_events
                if ev.get("product_id") and ev.get("event_type") in [
                    "product_click", "add_to_cart", "wishlist_click", "purchase"
                ]
            ]
            
            # Sort by timestamp descending (newest first)
            interacted_events = sorted(
                interacted_events,
                key=lambda x: str(x.get("event_timestamp", "")),
                reverse=True
            )

            # Recommend up to 3 recently viewed products
            score_base = 0.95
            for ev in interacted_events:
                pid = ev["product_id"]
                if pid not in seen_products:
                    seen_products.add(pid)
                    ev_type = ev.get("event_type", "click")
                    reason = "Recently viewed"
                    if ev_type == "add_to_cart":
                        reason = "Item in cart"
                    elif ev_type == "wishlist_click":
                        reason = "In your wishlist"
                    elif ev_type == "purchase":
                        reason = "Related to past purchase"

                    recs.append({
                        "product_id": pid,
                        "category": get_product_category(pid),
                        "reason": reason,
                        "score": round(score_base, 2),
                    })
                    score_base -= 0.05
                    if len(recs) >= 3:
                        break

            # 2. Add popular category affinity products
            if len(recs) < 5:
                # Find most frequent category
                categories = [get_product_category(ev["product_id"]) for ev in interacted_events if ev.get("product_id")]
                if categories:
                    fav_cat = pd.Series(categories).mode()[0]
                    # Generate deterministic products for this favorite category
                    cat_prefix = {
                        "Electronics": 200,
                        "Fashion & Apparel": 400,
                        "Home & Living": 600,
                        "Beauty & Health": 800,
                        "Sports & Outdoors": 950,
                    }.get(fav_cat, 100)

                    score_cat = 0.80
                    for offset in [15, 45, 88]:
                        pid = f"P-{cat_prefix + offset}"
                        if pid not in seen_products:
                            seen_products.add(pid)
                            recs.append({
                                "product_id": pid,
                                "category": fav_cat,
                                "reason": f"Popular in {fav_cat}",
                                "score": round(score_cat, 2),
                            })
                            score_cat -= 0.04
                            if len(recs) >= 5:
                                break

        # 3. Fallback to general trending products
        if len(recs) < 5:
            for item in self.trending_products:
                pid = item["product_id"]
                if pid not in seen_products:
                    seen_products.add(pid)
                    recs.append(item.copy())
                    if len(recs) >= 5:
                        break

        # Contract structure
        return {
            "customer_id": customer_id,
            "recommendations": recs[:5],
            "strategy": "BASELINE",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }
