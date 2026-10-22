"""
churn-engine/src/response_simulator.py
======================================
Phase 7 — Intervention Effectiveness and Customer Response Simulator.

Responsibilities
----------------
- Simulate individual and cohort-level customer responses to recommendations.
- Compare Control Group (Baseline Recommendations) vs. Treatment Group (Retention Overrides).
- Calculate simulated conversion rates, retention rates, and churn rates.
- Compute conversion uplift (absolute & relative) and retention uplift.
- Model simulated financial and business impacts:
    * Gross order revenue
    * Discount expenditures
    * Net revenue impact
    * Net margin impact (accounting for cost of goods sold and discount erosion)
    * Return on Discount Spend (ROI)
- Perform threshold sensitivity what-if analysis across candidate cutoffs [0.30 - 0.70].
- Provide customer-level explainable simulated outcomes.

Note on Authenticity
--------------------
ALL outputs produced by this module represent SIMULATED OUTCOMES based on
calibrated behavioural models and persona price sensitivities. They are designed
for academic evaluation and business trade-off analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.event_stream import PERSONAS
from src.utils import get_project_root, load_config

logger = logging.getLogger(__name__)

# ===========================================================================
# Persona Sensitivity Parameters for Simulation
# ===========================================================================
# Price/discount sensitivity multipliers calibrate how strongly each persona
# reacts to discount incentives relative to their base order probability.
PERSONA_PRICE_SENSITIVITY: dict[str, float] = {
    "discount_hunter": 1.80,  # Highly motivated by coupon discounts
    "window_shopper":  1.40,  # Needs strong discount incentive to convert
    "at_risk":         1.35,  # Churning, but discounts provide re-engagement lift
    "returning":       1.15,  # Moderate responsiveness to incentives
    "loyal":           0.90,  # High baseline conversion, lower incremental price lift
    "high_value":      0.60,  # Wealthy/premium, less sensitive to minor discounts
}


def get_customer_persona(customer_id: str, seed: int = 42) -> str:
    """Deterministically resolve customer persona from Customer ID."""
    try:
        idx = int(customer_id.split("-")[1]) - 1000
        p_names = list(PERSONAS.keys())
        p_weights = [PERSONAS[p]["weight"] for p in p_names]
        import random
        temp_rng = random.Random(seed)
        p_list = temp_rng.choices(p_names, weights=p_weights, k=max(1000, idx + 1))
        if 0 <= idx < len(p_list):
            return p_list[idx]
    except Exception:
        pass
    return "returning"


# ===========================================================================
# Data Structures
# ===========================================================================

@dataclass
class CustomerSimulationRecord:
    """Individual simulated outcome for a single customer evaluation."""
    customer_id: str
    persona: str
    churn_probability: float
    risk_tier: str
    is_flagged: bool
    threshold_applied: float
    baseline_strategy: str
    override_strategy: str
    discount_offered: float
    aov: float
    control_converted: bool
    treatment_converted: bool
    control_retained: bool
    treatment_retained: bool
    control_revenue: float
    treatment_revenue: float
    discount_cost: float
    net_margin_impact: float
    response_outcome: str


@dataclass
class SimulationSummary:
    """Aggregated cohort metrics comparing Control vs. Treatment groups."""
    total_customers: int
    flagged_customers: int
    flagged_ratio: float
    threshold: float
    gross_margin_rate: float

    # Conversion Metrics
    control_conversions: int
    treatment_conversions: int
    control_conversion_rate: float
    treatment_conversion_rate: float
    absolute_conversion_uplift: float
    relative_conversion_uplift: float

    # Retention Metrics
    control_retained: int
    treatment_retained: int
    control_retention_rate: float
    treatment_retention_rate: float
    retention_uplift: float
    control_churn_rate: float
    treatment_churn_rate: float

    # Financial Metrics ($)
    baseline_revenue: float
    intervention_revenue: float
    net_revenue_impact: float
    total_discount_cost: float
    baseline_margin: float
    intervention_margin: float
    net_margin_impact: float
    return_on_discount_spend: float

    # DataFrames for drill-down and visualization
    customer_records_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    persona_breakdown_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ===========================================================================
# Core Simulator
# ===========================================================================

class InterventionSimulator:
    """Simulates customer response to recommendations and overrides.

    Parameters
    ----------
    config : dict[str, Any] | None
        Project configuration dictionary.
    gross_margin : float
        Standard baseline gross margin percentage (default: 0.30 = 30%).
    random_seed : int
        Seed for reproducible stochastic outcomes (default: 42).
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        gross_margin: float = 0.30,
        random_seed: int = 42,
    ) -> None:
        self.config = config or load_config()
        self.gross_margin = float(gross_margin)
        self.random_seed = int(random_seed)

        # Load authoritative decision threshold from policy JSON if available
        self.default_threshold = self._load_policy_threshold()

        # Load discount settings from config
        rec_cfg = self.config.get("recommender", {})
        over_cfg = rec_cfg.get("override", {})
        self.discounts = {
            "LOW": 0.0,
            "MEDIUM": float(over_cfg.get("medium_risk_discount", 0.05)),
            "HIGH": float(over_cfg.get("high_risk_discount", 0.15)),
            "CRITICAL": float(over_cfg.get("critical_risk_discount", 0.25)),
        }

    def _load_policy_threshold(self) -> float:
        """Load selected threshold from models/decision_policy.json."""
        try:
            root = get_project_root()
            policy_path = root / "models" / "decision_policy.json"
            if policy_path.exists():
                with open(policy_path, "r", encoding="utf-8") as f:
                    policy = json.load(f)
                    return float(policy.get("selected_threshold", 0.50))
        except Exception as e:
            logger.warning("Could not read models/decision_policy.json: %s. Defaulting to 0.50", e)
        return 0.50

    def load_customer_features(self) -> pd.DataFrame:
        """Load customer features parquet or raw dataset fallback."""
        root = get_project_root()
        feat_path = root / "data" / "features" / "customer_features.parquet"
        if feat_path.exists():
            return pd.read_parquet(feat_path)

        # Fallback to customer history in StateManager
        from src.event_stream import CustomerStateManager
        sm = CustomerStateManager(self.config)
        rows = []
        for cid in list(sm.customer_history.keys()):
            df_f = sm.get_features(cid)
            rows.append(df_f.iloc[0])
        if rows:
            return pd.DataFrame(rows)

        raise FileNotFoundError("Customer features could not be loaded for simulation.")

    def run_simulation(
        self,
        features_df: pd.DataFrame | None = None,
        threshold: float | None = None,
        seed: int | None = None,
    ) -> SimulationSummary:
        """Run complete Control vs. Treatment cohort simulation.

        Parameters
        ----------
        features_df : pd.DataFrame | None
            Customer features dataframe. If None, loaded from disk.
        threshold : float | None
            Intervention threshold. Defaults to policy threshold (0.50).
        seed : int | None
            RNG seed. Defaults to instance seed (42).

        Returns
        -------
        SimulationSummary
            Complete metrics summary comparing Control and Treatment groups.
        """
        if features_df is None:
            features_df = self.load_customer_features()

        cutoff = float(threshold if threshold is not None else self.default_threshold)
        sim_seed = int(seed if seed is not None else self.random_seed)
        rng = np.random.RandomState(sim_seed)

        records: list[CustomerSimulationRecord] = []
        n_customers = len(features_df)

        for i, (cid, row) in enumerate(features_df.iterrows()):
            customer_id = str(cid)
            persona = get_customer_persona(customer_id, seed=sim_seed)
            p_info = PERSONAS.get(persona, PERSONAS["returning"])

            # 1. Churn Probability & Risk Tier
            # Use true churn or score proxy if model inference not called
            if "churn" in row:
                base_prob = p_info["base_churn_prob"]
                recency_factor = min(1.0, float(row.get("recency_days", 30.0)) / 90.0)
                abandon_factor = float(row.get("cart_abandonment_rate", 0.3))
                prob = float(np.clip(base_prob * 0.5 + recency_factor * 0.3 + abandon_factor * 0.2, 0.05, 0.98))
            else:
                prob = 0.50

            # Classify Risk Tier
            if prob < 0.30:
                risk_tier = "LOW"
            elif prob < cutoff:
                risk_tier = "MEDIUM"
            elif prob < 0.80:
                risk_tier = "HIGH"
            else:
                risk_tier = "CRITICAL"

            # 2. Decision Gate Evaluation
            is_flagged = prob >= cutoff
            discount_offered = self.discounts[risk_tier] if is_flagged else 0.0

            # 3. Order Value Calculation
            aov = float(row.get("avg_order_value", 0.0))
            if aov <= 0.0:
                aov = float((p_info["avg_order_val"][0] + p_info["avg_order_val"][1]) / 2.0)

            # 4. Control Group (Normal Baseline Path — No Discount)
            # Baseline conversion probability decreases with churn probability
            base_order_p = p_info["order_prob"]
            p_conv_ctl = float(np.clip(base_order_p * (1.0 - 0.55 * prob), 0.02, 0.90))
            p_ret_ctl = float(np.clip((1.0 - prob) * 0.85 + 0.15 * p_conv_ctl, 0.05, 0.95))

            draw_conv_ctl = rng.uniform(0.0, 1.0)
            draw_ret_ctl = rng.uniform(0.0, 1.0)
            ctl_converted = draw_conv_ctl < p_conv_ctl
            ctl_retained = draw_ret_ctl < p_ret_ctl

            # 5. Treatment Group (Override Path — Discount if Flagged)
            if not is_flagged:
                # Customer unflagged: Receives baseline recommendation
                trt_converted = ctl_converted
                trt_retained = ctl_retained
                discount_cost = 0.0
                trt_revenue = aov if trt_converted else 0.0
                trt_margin = trt_revenue * self.gross_margin
                response_outcome = "CONVERTED_BASELINE" if trt_converted else "NO_CONVERSION_BASELINE"
            else:
                # Customer flagged: Receives retention override with discount
                price_sens = PERSONA_PRICE_SENSITIVITY.get(persona, 1.0)
                # Incentive lift depends on discount size, persona sensitivity, and current risk
                conv_lift = discount_offered * price_sens * (1.0 - 0.25 * prob)
                p_conv_trt = float(np.clip(p_conv_ctl + conv_lift, 0.02, 0.95))

                ret_lift = conv_lift * 0.75 + discount_offered * 0.35
                p_ret_trt = float(np.clip(p_ret_ctl + ret_lift, 0.05, 0.98))

                # Paired simulation: use correlated draw so effect is attributable to treatment
                trt_converted = (draw_conv_ctl < p_conv_trt)
                trt_retained = (draw_ret_ctl < p_ret_trt)

                if trt_converted:
                    discount_cost = aov * discount_offered
                    trt_revenue = aov * (1.0 - discount_offered)
                    # Margin is net revenue minus cost of goods sold (COGS = AOV * (1 - gross_margin))
                    trt_margin = aov * (self.gross_margin - discount_offered)
                    if not ctl_converted:
                        response_outcome = "CONVERTED_VIA_OVERRIDE"
                    else:
                        response_outcome = "CONVERTED_WITH_DISCOUNT"
                else:
                    discount_cost = 0.0
                    trt_revenue = 0.0
                    trt_margin = 0.0
                    response_outcome = "NO_CONVERSION"

                if trt_retained and not ctl_retained:
                    response_outcome = "SAVED_FROM_CHURN"

            ctl_revenue = aov if ctl_converted else 0.0
            ctl_margin = ctl_revenue * self.gross_margin
            net_margin_impact = trt_margin - ctl_margin

            records.append(
                CustomerSimulationRecord(
                    customer_id=customer_id,
                    persona=persona.replace("_", " ").title(),
                    churn_probability=round(prob, 4),
                    risk_tier=risk_tier,
                    is_flagged=is_flagged,
                    threshold_applied=cutoff,
                    baseline_strategy="BASELINE",
                    override_strategy="RETENTION_OVERRIDE" if is_flagged else "BASELINE",
                    discount_offered=discount_offered,
                    aov=round(aov, 2),
                    control_converted=ctl_converted,
                    treatment_converted=trt_converted,
                    control_retained=ctl_retained,
                    treatment_retained=trt_retained,
                    control_revenue=round(ctl_revenue, 2),
                    treatment_revenue=round(trt_revenue, 2),
                    discount_cost=round(discount_cost, 2),
                    net_margin_impact=round(net_margin_impact, 2),
                    response_outcome=response_outcome,
                )
            )

        # Convert records to DataFrame
        rec_df = pd.DataFrame([r.__dict__ for r in records])

        # Calculate Cohort Aggregates
        ctl_conv_count = int(rec_df["control_converted"].sum())
        trt_conv_count = int(rec_df["treatment_converted"].sum())
        ctl_conv_rate = float(ctl_conv_count / n_customers) if n_customers > 0 else 0.0
        trt_conv_rate = float(trt_conv_count / n_customers) if n_customers > 0 else 0.0

        abs_conv_uplift = trt_conv_rate - ctl_conv_rate
        rel_conv_uplift = (abs_conv_uplift / ctl_conv_rate) if ctl_conv_rate > 0 else 0.0

        ctl_ret_count = int(rec_df["control_retained"].sum())
        trt_ret_count = int(rec_df["treatment_retained"].sum())
        ctl_ret_rate = float(ctl_ret_count / n_customers) if n_customers > 0 else 0.0
        trt_ret_rate = float(trt_ret_count / n_customers) if n_customers > 0 else 0.0

        ret_uplift = trt_ret_rate - ctl_ret_rate
        ctl_churn_rate = 1.0 - ctl_ret_rate
        trt_churn_rate = 1.0 - trt_ret_rate

        flagged_count = int(rec_df["is_flagged"].sum())
        flagged_ratio = float(flagged_count / n_customers) if n_customers > 0 else 0.0

        baseline_rev = float(rec_df["control_revenue"].sum())
        intervention_rev = float(rec_df["treatment_revenue"].sum())
        net_revenue_impact = intervention_rev - baseline_rev
        total_discount = float(rec_df["discount_cost"].sum())

        ctl_margin_total = baseline_rev * self.gross_margin
        trt_margin_total = float(rec_df["net_margin_impact"].sum()) + ctl_margin_total
        net_margin_impact = float(rec_df["net_margin_impact"].sum())

        roi = (net_margin_impact / total_discount) if total_discount > 0 else 0.0

        # Persona breakdown
        persona_grp = rec_df.groupby("persona").agg(
            customers=("customer_id", "count"),
            flagged=("is_flagged", "sum"),
            control_conv=("control_converted", "mean"),
            treatment_conv=("treatment_converted", "mean"),
            control_ret=("control_retained", "mean"),
            treatment_ret=("treatment_retained", "mean"),
            discount_cost=("discount_cost", "sum"),
            net_margin=("net_margin_impact", "sum"),
        ).reset_index()

        persona_grp["conv_uplift"] = persona_grp["treatment_conv"] - persona_grp["control_conv"]
        persona_grp["ret_uplift"] = persona_grp["treatment_ret"] - persona_grp["control_ret"]

        return SimulationSummary(
            total_customers=n_customers,
            flagged_customers=flagged_count,
            flagged_ratio=round(flagged_ratio, 4),
            threshold=round(cutoff, 4),
            gross_margin_rate=self.gross_margin,
            control_conversions=ctl_conv_count,
            treatment_conversions=trt_conv_count,
            control_conversion_rate=round(ctl_conv_rate, 4),
            treatment_conversion_rate=round(trt_conv_rate, 4),
            absolute_conversion_uplift=round(abs_conv_uplift, 4),
            relative_conversion_uplift=round(rel_conv_uplift, 4),
            control_retained=ctl_ret_count,
            treatment_retained=trt_ret_count,
            control_retention_rate=round(ctl_ret_rate, 4),
            treatment_retention_rate=round(trt_ret_rate, 4),
            retention_uplift=round(ret_uplift, 4),
            control_churn_rate=round(ctl_churn_rate, 4),
            treatment_churn_rate=round(trt_churn_rate, 4),
            baseline_revenue=round(baseline_rev, 2),
            intervention_revenue=round(intervention_rev, 2),
            net_revenue_impact=round(net_revenue_impact, 2),
            total_discount_cost=round(total_discount, 2),
            baseline_margin=round(ctl_margin_total, 2),
            intervention_margin=round(trt_margin_total, 2),
            net_margin_impact=round(net_margin_impact, 2),
            return_on_discount_spend=round(roi, 4),
            customer_records_df=rec_df,
            persona_breakdown_df=persona_grp,
        )

    def evaluate_threshold_sensitivity(
        self,
        features_df: pd.DataFrame | None = None,
        thresholds: list[float] | None = None,
    ) -> pd.DataFrame:
        """Run what-if analysis across multiple decision threshold candidates.

        Parameters
        ----------
        features_df : pd.DataFrame | None
            Customer features dataset.
        thresholds : list[float] | None
            Thresholds to evaluate (default: [0.30, 0.40, 0.50, 0.60, 0.70]).

        Returns
        -------
        pd.DataFrame
            Comparison table showing trade-offs for each threshold candidate.
        """
        if thresholds is None:
            thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

        if features_df is None:
            features_df = self.load_customer_features()

        results = []
        for t in thresholds:
            summary = self.run_simulation(features_df, threshold=t, seed=self.random_seed)
            df_rec = summary.customer_records_df

            unnecessary_discounts = int(
                ((df_rec["is_flagged"]) & (df_rec["control_converted"])).sum()
            )

            results.append({
                "threshold": t,
                "customers_flagged": summary.flagged_customers,
                "intervention_rate": f"{summary.flagged_ratio:.1%}",
                "treatment_conv_rate": f"{summary.treatment_conversion_rate:.2%}",
                "conv_uplift": f"{summary.absolute_conversion_uplift:+.2%}",
                "treatment_ret_rate": f"{summary.treatment_retention_rate:.2%}",
                "retention_uplift": f"{summary.retention_uplift:+.2%}",
                "unnecessary_discount_count": unnecessary_discounts,
                "discount_cost": summary.total_discount_cost,
                "net_revenue_impact": summary.net_revenue_impact,
                "net_margin_impact": summary.net_margin_impact,
                "roi": summary.return_on_discount_spend,
            })

        return pd.DataFrame(results)

    def simulate_single_event_response(
        self,
        customer_id: str,
        churn_probability: float,
        risk_level: str,
        discount_offered: float,
        aov: float = 85.0,
        persona: str | None = None,
    ) -> dict[str, Any]:
        """Simulate real-time response outcome for an individual live event.

        Useful for live dashboard timeline display and audit log enrichment.
        """
        resolved_persona = persona or get_customer_persona(customer_id, seed=self.random_seed)
        p_info = PERSONAS.get(resolved_persona, PERSONAS["returning"])
        price_sens = PERSONA_PRICE_SENSITIVITY.get(resolved_persona, 1.0)

        # Baseline conversion probability
        p_conv_ctl = float(np.clip(p_info["order_prob"] * (1.0 - 0.55 * churn_probability), 0.02, 0.90))

        if discount_offered > 0.0:
            conv_lift = discount_offered * price_sens * (1.0 - 0.25 * churn_probability)
            p_conv = float(np.clip(p_conv_ctl + conv_lift, 0.02, 0.95))
            discount_cost = aov * discount_offered
            net_revenue = aov * (1.0 - discount_offered)
            net_margin = aov * (self.gross_margin - discount_offered)
            action_desc = f"Applied {int(discount_offered*100)}% retention coupon"
        else:
            p_conv = p_conv_ctl
            discount_cost = 0.0
            net_revenue = aov
            net_margin = aov * self.gross_margin
            action_desc = "Standard baseline recommendation"

        # Deterministic draw based on customer ID hash
        draw = (hash(customer_id) % 1000) / 1000.0
        converted = draw < p_conv

        outcome_status = "CONVERTED" if converted else "NO_CONVERSION"
        if converted and discount_offered > 0.0:
            outcome_status = "CONVERTED_WITH_DISCOUNT"

        return {
            "customer_id": customer_id,
            "persona": resolved_persona.replace("_", " ").title(),
            "action_desc": action_desc,
            "simulated_converted": converted,
            "outcome_status": outcome_status,
            "simulated_revenue": round(net_revenue if converted else 0.0, 2),
            "discount_cost": round(discount_cost if converted else 0.0, 2),
            "margin_contribution": round(net_margin if converted else 0.0, 2),
        }
