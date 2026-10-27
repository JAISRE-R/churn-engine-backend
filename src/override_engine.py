"""
churn-engine/src/override_engine.py
====================================
Recommendation Override Engine.

Responsibilities
----------------
- Intercept baseline recommendations for customer profiles requiring intervention.
- Formulate personalized retention recommendations (discounts, bundles, viewed items).
- Apply business constraints (configurable risk-tier discount levels).
- Log override events to a thread-safe memory audit trail.
- Maintain consistent output structures matching the recommendation contract.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime
from typing import Any

from src.demo_mode import DEMO_AT_RISK_CUSTOMERS, local_demo_enabled
from src.recommender import get_product_category

logger = logging.getLogger(__name__)


class OverrideEngine:
    """Middle-tier override engine for retention targeting and margin control."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        
        # Load risk-tier discount configurations
        rec_cfg = config.get("recommender", {})
        over_cfg = rec_cfg.get("override", {})
        self.medium_risk_discount = float(over_cfg.get("medium_risk_discount", 0.05))
        self.high_risk_discount = float(over_cfg.get("high_risk_discount", 0.15))
        self.critical_risk_discount = float(over_cfg.get("critical_risk_discount", 0.25))

        self._lock = threading.Lock()
        
        # Thread-safe circular audit logs
        self.audit_log: deque[dict[str, Any]] = deque(maxlen=1000)

    def process(
        self,
        baseline_recs: dict[str, Any],
        decision_res: dict[str, Any],
        customer_events: list[dict[str, Any]] | None = None,
        offers_allowed: bool = True,
        redeemed_product_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """Intercept baseline recommendations and apply overrides if necessary.

        Parameters
        ----------
        baseline_recs : dict[str, Any]
            RecommendationResult from BaselineRecommender.
        decision_res : dict[str, Any]
            Gate decision dict from DecisionGate.
        customer_events : list[dict[str, Any]] | None
            Log of customer behavioural events.
        offers_allowed : bool
            Server-computed authorization for the current customer.
        redeemed_product_ids : set[str] | None
            Product IDs whose retention offer was already redeemed.

        Returns
        -------
        dict[str, Any]
            Final RecommendationResult containing the resolved recommendation list.
        """
        # Read activation decision from the Decision Gate
        override_triggered = decision_res.get("decision") == "INTERVENTION_REQUIRED"
        offers_allowed = offers_allowed and (
            not local_demo_enabled(self.config)
            or baseline_recs["customer_id"] in DEMO_AT_RISK_CUSTOMERS
        )
        
        if not override_triggered or not offers_allowed:
            # Pass-through baseline unchanged
            return {
                "customer_id": baseline_recs["customer_id"],
                "recommendations": baseline_recs["recommendations"],
                "strategy": "BASELINE",
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": {
                    "override_triggered": False,
                    "risk_level": decision_res.get("risk_level", "LOW"),
                    "churn_probability": decision_res.get("churn_probability", 0.0),
                    "threshold": decision_res.get("threshold", 0.40),
                    "override_reason": (
                        "Retention offers are not authorized for this customer"
                        if override_triggered and not offers_allowed
                        else "Risk probability below threshold"
                    ),
                }
            }

        # Build override list
        cid = baseline_recs["customer_id"]
        risk_lvl = decision_res.get("risk_level", "HIGH")
        prob = decision_res.get("churn_probability", 0.50)
        threshold = decision_res.get("threshold", 0.40)
        
        # Determine discount tier
        discount = 0.0
        if risk_lvl == "MEDIUM":
            discount = self.medium_risk_discount
        elif risk_lvl == "HIGH":
            discount = self.high_risk_discount
        elif risk_lvl == "CRITICAL":
            discount = self.critical_risk_discount

        # Generate retention list
        recs = []
        seen = set()

        # Parse behavioral signals
        abandoned_pids = []
        viewed_pids = []

        if customer_events:
            # Cart abandonments (added to cart, never purchased)
            cart_adds = {ev["product_id"] for ev in customer_events if ev.get("event_type") == "add_to_cart" and ev.get("product_id")}
            purchases = {ev["product_id"] for ev in customer_events if ev.get("event_type") == "purchase" and ev.get("product_id")}
            abandoned_pids = [pid for pid in cart_adds if pid not in purchases]

            # Unique viewed products in reverse order
            for ev in reversed(customer_events):
                pid = ev.get("product_id")
                if pid and pid not in viewed_pids:
                    viewed_pids.append(pid)

        # Formulate strategies based on risk tier
        if risk_lvl == "CRITICAL":
            # Bundle recommendations remain recommendations, not cart-wide offers.
            base_product = viewed_pids[0] if viewed_pids else "P-101"
            bundle_pid = f"{base_product}-BNDL"
            recs.append({
                "product_id": bundle_pid,
                "category": get_product_category(base_product),
                "reason": f"Bundle Deal: {base_product} + Accessories",
                "score": 0.95,
            })
            seen.add(bundle_pid)

        # Offers belong to specific previously viewed or abandoned products.
        eligible_pids = list(dict.fromkeys(abandoned_pids + viewed_pids))
        for pid in eligible_pids:
            if pid in seen or pid in (redeemed_product_ids or set()):
                continue
            seen.add(pid)
            recs.append({
                "product_id": pid,
                "category": get_product_category(pid),
                "reason": f"Retention offer: save {discount*100:.0f}% on this item",
                "score": 0.92 if pid in abandoned_pids else 0.85,
                "discount": int(discount * 100),
            })

        # Fill remaining slots with category-specific or baseline items
        for item in baseline_recs["recommendations"]:
            if len(recs) >= 5:
                break
            pid = item["product_id"]
            if pid not in seen:
                seen.add(pid)
                # Clone and lower score slightly to reflect overridden position
                overridden_item = item.copy()
                overridden_item["score"] = round(item["score"] * 0.9, 2)
                recs.append(overridden_item)

        # Audit log creation
        override_reason = f"Churn probability {prob:.4f} >= threshold {threshold:.2f} ({risk_lvl} risk)"
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        audit_record = {
            "timestamp": timestamp,
            "customer_id": cid,
            "churn_probability": round(prob, 4),
            "threshold": threshold,
            "risk_level": risk_lvl,
            "baseline_strategy": baseline_recs["strategy"],
            "final_strategy": "RETENTION_OVERRIDE",
            "override_triggered": True,
            "override_reason": override_reason,
            "discount_value": discount,
            "recommended_products": [r["product_id"] for r in recs],
        }

        with self._lock:
            self.audit_log.append(audit_record)

        return {
            "customer_id": cid,
            "recommendations": recs,
            "strategy": "RETENTION_OVERRIDE",
            "timestamp": timestamp,
            "metadata": {
                "override_triggered": True,
                "risk_level": risk_lvl,
                "churn_probability": round(prob, 4),
                "threshold": threshold,
                "override_reason": override_reason,
                "discount_pct": int(discount * 100),
            }
        }

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return cumulative logs of override activations."""
        with self._lock:
            return list(self.audit_log)

    def clear(self) -> None:
        """Clear audit history logs."""
        with self._lock:
            self.audit_log.clear()
