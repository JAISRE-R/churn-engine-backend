"""
churn-engine/dashboard/pages/pg_impact_evaluation.py
======================================================
Impact Evaluation Page — Phase 7 Intervention Effectiveness & Business Analytics.

Demonstrates:
  1. Simulated A/B Control vs. Treatment outcomes.
  2. Conversion and retention uplift metrics.
  3. Cumulative financial and margin impact analysis.
  4. Threshold sensitivity what-if analysis ([0.30 - 0.70]).
  5. Customer-level simulated intervention drill-down.
  6. Transparent simulation assumptions & methodology.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.response_simulator import InterventionSimulator, SimulationSummary
from src.utils import load_config


@st.cache_data(show_spinner=False)
def get_cached_simulation_results(threshold: float = 0.50, seed: int = 42) -> SimulationSummary:
    """Run simulation with caching for responsive dashboard rendering."""
    sim = InterventionSimulator(random_seed=seed)
    return sim.run_simulation(threshold=threshold)


@st.cache_data(show_spinner=False)
def get_cached_sensitivity_table(seed: int = 42) -> pd.DataFrame:
    """Run sensitivity analysis across thresholds with caching."""
    sim = InterventionSimulator(random_seed=seed)
    return sim.evaluate_threshold_sensitivity()


def render() -> None:
    """Render the Impact Evaluation dashboard."""

    if "config" not in st.session_state:
        st.session_state.config = load_config()

    # --- Page Header ---
    st.markdown("""
    <div class="page-header">
        <div class="phase-badge">Phase 7 — Business &amp; Retention Analytics</div>
        <h1 class="page-title">Intervention Impact Evaluation</h1>
        <p class="page-subtitle">
            Simulated A/B performance evaluation of real-time recommendation overrides vs. baseline recommendation policy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Prominent SIMULATED RESULTS Disclaimer ---
    st.markdown("""
    <div style="
        background-color: #EAF0FA;
        border: 1px solid #D6E0F2;
        border-left: 4px solid #4263A5;
        border-radius: 8px;
        padding: 0.9rem 1.25rem;
        margin-bottom: 1.5rem;
        font-size: 0.88rem;
        color: #1F2937;
    ">
        <strong>SIMULATED RESULTS:</strong>
        All uplift figures, conversion outcomes, and financial returns presented on this page are
        <strong>reproducible simulated projections</strong> generated via calibrated persona behavioural models,
        price-sensitivity curves, and paired stochastic response models. They represent prospective business trade-off
        analyses rather than live production customer experiments.
    </div>
    """, unsafe_allow_html=True)

    policy_threshold = InterventionSimulator(
        config=st.session_state.config,
        random_seed=42,
    ).default_threshold

    # --- Simulation Controls Toolbar ---
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 3])
    with col_ctrl1:
        active_threshold = st.slider(
            "Evaluation Decision Threshold (θ)",
            min_value=0.20,
            max_value=0.80,
            value=min(0.80, max(0.20, policy_threshold)),
            step=0.05,
            help="Scenario threshold. The configured policy threshold is shown for reference.",
        )
    with col_ctrl2:
        margin_pct = st.slider(
            "Gross Margin Rate (%)",
            min_value=15,
            max_value=50,
            value=30,
            step=5,
            help="Base e-commerce gross product margin before discount deductions.",
        )
    with col_ctrl3:
        st.markdown("<div style='margin-top: 1.6rem;'></div>", unsafe_allow_html=True)
        is_default = active_threshold == policy_threshold and margin_pct == 30
        st.caption(
            f"Configured policy threshold: **{policy_threshold:.2f}** · "
            f"{'Policy scenario' if is_default else 'Custom what-if scenario'}"
        )

    # --- Run Simulation ---
    sim = InterventionSimulator(
        config=st.session_state.config,
        gross_margin=margin_pct / 100.0,
        random_seed=42
    )
    summary = sim.run_simulation(threshold=active_threshold)

    st.markdown("### B. Simulated business impact")
    st.caption(
        "All figures below are outputs of the response simulator, not observed production "
        "conversion, retention, revenue, or margin. They follow the simulator’s current assumptions."
    )
    comparison_data = [
        {
            "Metric": "Conversion rate",
            "Baseline simulation": f"{summary.control_conversion_rate:.2%}",
            "Intervention simulation": f"{summary.treatment_conversion_rate:.2%}",
            "Difference": f"{summary.absolute_conversion_uplift:+.2%}",
        },
        {
            "Metric": "Retention rate",
            "Baseline simulation": f"{summary.control_retention_rate:.2%}",
            "Intervention simulation": f"{summary.treatment_retention_rate:.2%}",
            "Difference": f"{summary.retention_uplift:+.2%}",
        },
        {
            "Metric": "Gross margin",
            "Baseline simulation": f"${summary.baseline_margin:,.2f}",
            "Intervention simulation": f"${summary.intervention_margin:,.2f}",
            "Difference": f"${summary.net_margin_impact:+,.2f}",
        },
    ]
    st.caption(
        f"Paired simulated cohort: {summary.total_customers:,} customers · "
        f"{summary.flagged_customers:,} above the scenario threshold "
        f"({summary.threshold:.2%})."
    )
    st.dataframe(
        pd.DataFrame(comparison_data),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Simulated impact by decision threshold")
    st.caption(
        "What-if analysis from the same simulator; separate from historical offline model-threshold metrics."
    )
    df_sens = sim.evaluate_threshold_sensitivity()
    if not df_sens.empty and {"threshold", "net_margin_impact"}.issubset(df_sens.columns):
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=df_sens["threshold"],
            y=df_sens["net_margin_impact"],
            mode="lines+markers",
            name="Simulated net margin impact",
            line={"color": "#4263A5", "width": 2},
        ))
        fig_sens.add_vline(
            x=policy_threshold,
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text=f"Configured policy ({policy_threshold:.2%})",
        )
        fig_sens.update_layout(
            template="plotly_white",
            height=300,
            margin={"l": 30, "r": 30, "t": 40, "b": 30},
            xaxis_title="Scenario threshold",
            yaxis_title="Simulated net margin impact ($)",
        )
        st.plotly_chart(fig_sens, use_container_width=True)
    else:
        st.info("Simulated threshold sensitivity is unavailable.")

    st.markdown("---")

    st.markdown("### Simulation methodology")
    with st.expander("View formulas and assumptions", expanded=False):
        st.markdown("""
        #### 1. Baseline (Control) Probability Model
        In the control condition, customers receive standard recommendations without incentives.
        The baseline purchase probability drops monotonically as churn risk increases:
        $$P_{conv, control} = \\text{clip}\\left(P_{order}(p) \\times (1.0 - 0.55 \\times c_i), 0.02, 0.90\\right)$$
        where $P_{order}(p)$ is the natural purchase probability for persona $p$, and $c_i$ is the customer's predicted churn probability.

        #### 2. Override (Treatment) Incentive Model
        When a customer crosses the Decision Gate threshold ($c_i \\ge \\theta$), the simulator applies its configured risk-tier incentive assumptions:
        - **MEDIUM Risk (30% - 50%):** 5% Discount Coupon
        - **HIGH Risk (50% - 80%):** 15% assumed discount
        - **CRITICAL Risk (≥ 80%):** 25% assumed discount

        The incremental conversion probability lift accounts for persona price sensitivity $S_{price}(p)$:
        $$\\Delta P_{conv} = d_i \\times S_{price}(p) \\times (1.0 - 0.25 \\times c_i)$$
        $$P_{conv, treat} = \\min(0.95, P_{conv, control} + \\Delta P_{conv})$$

        #### 3. Financial model and margin
        - Gross Revenue: $Rev = AOV_i \\times (1 - d_i)$
        - Cost of Goods Sold: $COGS = AOV_i \\times (1 - m_{gross})$
        - Treatment Margin: $Margin_{treat} = AOV_i \\times (m_{gross} - d_i)$
        - Net Margin Impact: $Margin_{treat} - Margin_{control}$

        #### 4. Model Constraints & Honest Limitations
        - Simulated outcomes assume rational economic response curves calibrated from persona profiles.
        - Real-world production deployment requires live randomized A/B trial validation to calibrate true price elasticity.
        - The current simulator uses average order value and risk-tier assumptions; it does not calculate product-specific offer eligibility.
        """)
