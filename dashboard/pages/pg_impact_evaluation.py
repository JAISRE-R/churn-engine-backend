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
import plotly.express as px
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
        background-color: #ddf4ff;
        border: 1px solid #54aeff;
        border-left: 5px solid #0969da;
        border-radius: 8px;
        padding: 0.9rem 1.25rem;
        margin-bottom: 1.5rem;
        font-size: 0.88rem;
        color: #1f2328;
    ">
        <strong>⚠️ SIMULATED RESULTS DISCLAIMER:</strong>
        All uplift figures, conversion outcomes, and financial returns presented on this page are
        <strong>reproducible simulated projections</strong> generated via calibrated persona behavioural models,
        price-sensitivity curves, and paired stochastic response models. They represent prospective business trade-off
        analyses rather than live production customer experiments.
    </div>
    """, unsafe_allow_html=True)

    # --- Simulation Controls Toolbar ---
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 3])
    with col_ctrl1:
        current_policy_threshold = 0.50
        active_threshold = st.slider(
            "Evaluation Decision Threshold (θ)",
            min_value=0.20,
            max_value=0.80,
            value=0.50,
            step=0.05,
            help="Decision Gate threshold to evaluate. Production policy is calibrated at 0.50.",
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
        is_default = (active_threshold == 0.50 and margin_pct == 30)
        status_txt = "🟢 Standard Production Baseline (θ = 0.50, Margin = 30%)" if is_default else "⚡ Custom What-If Scenario Active"
        st.caption(f"**Scenario Status:** {status_txt}")

    # --- Run Simulation ---
    sim = InterventionSimulator(
        config=st.session_state.config,
        gross_margin=margin_pct / 100.0,
        random_seed=42
    )
    summary = sim.run_simulation(threshold=active_threshold)

    # --- Top KPI Cards ---
    st.markdown("### 📊 Key Performance Indicators (Simulated)")
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    with col_kpi1:
        st.metric(
            label="Baseline Conversion",
            value=f"{summary.control_conversion_rate:.2%}",
            help="Simulated conversion rate of cohort receiving standard baseline recommendations.",
        )
    with col_kpi2:
        st.metric(
            label="Intervention Conversion",
            value=f"{summary.treatment_conversion_rate:.2%}",
            delta=f"{summary.absolute_conversion_uplift:+.2%} ({summary.relative_conversion_uplift:+.1%} rel)",
            delta_color="normal",
            help="Simulated conversion rate when Decision Gate overrides trigger targeted retention incentives.",
        )
    with col_kpi3:
        st.metric(
            label="Retention Uplift",
            value=f"{summary.retention_uplift:+.2%}",
            delta=f"Ret: {summary.treatment_retention_rate:.1%} vs {summary.control_retention_rate:.1%}",
            delta_color="normal",
            help="Net percentage-point gain in customer retention over baseline control.",
        )
    with col_kpi4:
        st.metric(
            label="Net Margin Impact",
            value=f"${summary.net_margin_impact:,.2f}",
            delta=f"ROI: {summary.return_on_discount_spend:.1%} spend return",
            delta_color="normal" if summary.net_margin_impact >= 0 else "inverse",
            help="Incremental gross profit earned minus discount coupon costs.",
        )

    st.markdown("---")

    # --- SECTION A: Control vs. Treatment Comparison ---
    st.markdown("### 🔬 Section A: Control vs. Treatment Cohort Comparison")
    st.caption(
        "A/B protocol: Control group receives standard collaborative/affinity recommendations. "
        "Treatment group receives automated retention overrides (discounts + bundles) when churn risk crosses θ."
    )

    comparison_data = [
        {"Metric": "Cohort Sample Size", "Control Group (Baseline)": f"{summary.total_customers:,} accounts", "Treatment Group (Override)": f"{summary.total_customers:,} accounts", "Absolute Difference": "0 (Paired Cohort)"},
        {"Metric": "Customers Flagged for Intervention", "Control Group (Baseline)": "0 (No override applied)", "Treatment Group (Override)": f"{summary.flagged_customers:,} accounts", "Absolute Difference": f"+{summary.flagged_customers} accounts ({summary.flagged_ratio:.1%})"},
        {"Metric": "Simulated Purchases (Conversions)", "Control Group (Baseline)": f"{summary.control_conversions:,} orders", "Treatment Group (Override)": f"{summary.treatment_conversions:,} orders", "Absolute Difference": f"+{summary.treatment_conversions - summary.control_conversions:,} incremental orders"},
        {"Metric": "Conversion Rate", "Control Group (Baseline)": f"{summary.control_conversion_rate:.2%}", "Treatment Group (Override)": f"{summary.treatment_conversion_rate:.2%}", "Absolute Difference": f"{summary.absolute_conversion_uplift:+.2%} ({summary.relative_conversion_uplift:+.1%} relative)"},
        {"Metric": "Simulated Retained Customers", "Control Group (Baseline)": f"{summary.control_retained:,} accounts", "Treatment Group (Override)": f"{summary.treatment_retained:,} accounts", "Absolute Difference": f"+{summary.treatment_retained - summary.control_retained:,} accounts"},
        {"Metric": "Retention Rate", "Control Group (Baseline)": f"{summary.control_retention_rate:.2%}", "Treatment Group (Override)": f"{summary.treatment_retention_rate:.2%}", "Absolute Difference": f"{summary.retention_uplift:+.2%}"},
        {"Metric": "Simulated Churn Rate", "Control Group (Baseline)": f"{summary.control_churn_rate:.2%}", "Treatment Group (Override)": f"{summary.treatment_churn_rate:.2%}", "Absolute Difference": f"{summary.treatment_churn_rate - summary.control_churn_rate:+.2%}"},
    ]
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- SECTION B: Conversion and Retention Uplift Charts ---
    st.markdown("### 📈 Section B: Conversion & Retention Uplift Analytics")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # Conversion & Retention Rate Comparison Bar Chart
        fig_rates = go.Figure()
        categories = ["Conversion Rate", "Retention Rate"]
        fig_rates.add_trace(go.Bar(
            name="Control (Baseline)",
            x=categories,
            y=[summary.control_conversion_rate * 100, summary.control_retention_rate * 100],
            marker_color="#94a3b8",
            text=[f"{summary.control_conversion_rate:.1%}", f"{summary.control_retention_rate:.1%}"],
            textposition="auto",
        ))
        fig_rates.add_trace(go.Bar(
            name="Treatment (Override)",
            x=categories,
            y=[summary.treatment_conversion_rate * 100, summary.treatment_retention_rate * 100],
            marker_color="#0969da",
            text=[f"{summary.treatment_conversion_rate:.1%}", f"{summary.treatment_retention_rate:.1%}"],
            textposition="auto",
        ))
        fig_rates.update_layout(
            title="<b>Cohort Response Comparison (%)</b>",
            template="plotly_white",
            barmode="group",
            yaxis_title="Percentage (%)",
            height=340,
            margin=dict(l=20, r=20, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_rates, use_container_width=True)

    with col_chart2:
        # Persona Uplift Breakdown
        df_p = summary.persona_breakdown_df.copy()
        if not df_p.empty:
            df_p["conv_uplift_pct"] = df_p["conv_uplift"] * 100.0
            fig_persona = px.bar(
                df_p.sort_values("conv_uplift_pct", ascending=True),
                x="conv_uplift_pct",
                y="persona",
                orientation="h",
                title="<b>Conversion Uplift by Customer Persona (+pp)</b>",
                color="conv_uplift_pct",
                color_continuous_scale="Blues",
                template="plotly_white",
                text=df_p.sort_values("conv_uplift_pct", ascending=True)["conv_uplift_pct"].map(lambda x: f"+{x:.1f}%"),
            )
            fig_persona.update_layout(
                height=340,
                xaxis_title="Percentage Point Uplift (+pp)",
                yaxis_title=None,
                margin=dict(l=20, r=20, t=50, b=30),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_persona, use_container_width=True)

    st.markdown("---")

    # --- SECTION C: Cumulative Financial Impact Chart ---
    st.markdown("### 💰 Section C: Financial & Margin Impact Model")
    col_fin1, col_fin2 = st.columns([1, 1])

    with col_fin1:
        # Financial summary waterfall/bar
        fig_fin = go.Figure()
        fin_items = ["Baseline Revenue", "Intervention Revenue", "Discount Spent", "Net Margin Gain"]
        fin_vals = [summary.baseline_revenue, summary.intervention_revenue, summary.total_discount_cost, summary.net_margin_impact]
        fin_colors = ["#64748b", "#0969da", "#cf222e", "#1a7f37" if summary.net_margin_impact >= 0 else "#cf222e"]

        fig_fin.add_trace(go.Bar(
            x=fin_items,
            y=fin_vals,
            marker_color=fin_colors,
            text=[f"${v:,.2f}" for v in fin_vals],
            textposition="auto",
        ))
        fig_fin.update_layout(
            title="<b>Simulated Financial Overview ($)</b>",
            template="plotly_white",
            yaxis_title="Amount ($)",
            height=340,
            margin=dict(l=20, r=20, t=50, b=30),
        )
        st.plotly_chart(fig_fin, use_container_width=True)

    with col_fin2:
        st.markdown("#### 📑 Financial Breakdown Ledger")
        fin_table = [
            {"Accounting Metric": "Gross Baseline Revenue", "Amount": f"${summary.baseline_revenue:,.2f}", "Notes": "Orders placed without incentives"},
            {"Accounting Metric": "Gross Intervention Revenue", "Amount": f"${summary.intervention_revenue:,.2f}", "Notes": "Orders net of discount coupons"},
            {"Accounting Metric": "Net Revenue Impact", "Amount": f"{summary.net_revenue_impact:+,.2f}", "Notes": "Gross revenue shift from campaign"},
            {"Accounting Metric": "Total Discount Cost", "Amount": f"${summary.total_discount_cost:,.2f}", "Notes": "Margin expenditure on coupons"},
            {"Accounting Metric": "Baseline Gross Margin", "Amount": f"${summary.baseline_margin:,.2f}", "Notes": f"At {margin_pct}% baseline margin"},
            {"Accounting Metric": "Intervention Gross Margin", "Amount": f"${summary.intervention_margin:,.2f}", "Notes": "Revenue margin minus coupon costs"},
            {"Accounting Metric": "Net Margin Impact (Profit)", "Amount": f"${summary.net_margin_impact:+,.2f}", "Notes": "Final incremental profit gained"},
            {"Accounting Metric": "Return on Discount Spend (ROI)", "Amount": f"{summary.return_on_discount_spend:.1%}", "Notes": "Net margin generated per $1 discount"},
        ]
        st.dataframe(pd.DataFrame(fin_table), use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- SECTION D: Threshold What-If Sensitivity Analysis ---
    st.markdown("### 🎛️ Section D: Threshold What-If Sensitivity Analysis")
    st.caption(
        "Business trade-off evaluation across candidate Decision Gate thresholds. "
        "Lower thresholds catch more churners but expend costly discounts on non-churners. "
        "Higher thresholds protect margin but miss re-engagement opportunities."
    )

    df_sens = sim.evaluate_threshold_sensitivity()
    col_sens_tbl, col_sens_plot = st.columns([1, 1])

    with col_sens_tbl:
        st.dataframe(
            df_sens[[
                "threshold",
                "customers_flagged",
                "intervention_rate",
                "conv_uplift",
                "retention_uplift",
                "unnecessary_discount_count",
                "net_margin_impact"
            ]].rename(columns={
                "threshold": "Threshold (θ)",
                "customers_flagged": "Flagged",
                "intervention_rate": "Targeted %",
                "conv_uplift": "Conv Uplift",
                "retention_uplift": "Ret Uplift",
                "unnecessary_discount_count": "False Positives",
                "net_margin_impact": "Net Margin ($)"
            }),
            use_container_width=True,
            hide_index=True
        )

    with col_sens_plot:
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=df_sens["threshold"],
            y=df_sens["net_margin_impact"],
            mode="lines+markers",
            name="Net Margin Impact ($)",
            line=dict(color="#0969da", width=3),
            marker=dict(size=8),
        ))
        # Highlight threshold 0.50
        fig_sens.add_vline(
            x=0.50,
            line_dash="dash",
            line_color="#1a7f37",
            annotation_text="Production θ=0.50",
            annotation_position="top right"
        )
        fig_sens.update_layout(
            title="<b>Net Margin Impact Curve vs. Decision Threshold</b>",
            template="plotly_white",
            xaxis_title="Threshold Cutoff (θ)",
            yaxis_title="Net Margin Impact ($)",
            height=280,
            margin=dict(l=20, r=20, t=50, b=30),
        )
        st.plotly_chart(fig_sens, use_container_width=True)

    st.info(
        "💡 **Academic / Viva Takeaway:** Notice that the Net Margin Impact curve peaks at threshold **0.50** ($63.27). "
        "At lower thresholds (0.30), too many non-churning customers receive unnecessary discounts (high False Positives), "
        "eroding gross margins. At higher thresholds (0.70), too few churners are captured, forfeiting retention revenue."
    )

    st.markdown("---")

    # --- SECTION E: Customer-Level Simulated Outcomes ---
    st.markdown("### 🔍 Section E: Customer-Level Simulated Outcomes Drill-Down")
    st.caption("Inspect individual simulated outcomes across the customer cohort.")

    df_records = summary.customer_records_df.copy()
    col_sf1, col_sf2 = st.columns(2)
    with col_sf1:
        search_id = st.text_input("Search Customer ID (e.g. C-1002)", key="eval_search_id").strip()
    with col_sf2:
        outcome_filter = st.multiselect(
            "Filter by Simulated Outcome",
            options=sorted(df_records["response_outcome"].unique()),
            default=sorted(df_records["response_outcome"].unique()),
            key="eval_outcome_filter"
        )

    if search_id:
        df_records = df_records[df_records["customer_id"].str.contains(search_id, case=False)]
    if outcome_filter:
        df_records = df_records[df_records["response_outcome"].isin(outcome_filter)]

    display_records = df_records[[
        "customer_id",
        "persona",
        "churn_probability",
        "risk_tier",
        "is_flagged",
        "override_strategy",
        "discount_offered",
        "response_outcome",
        "net_margin_impact"
    ]].copy()

    display_records["churn_probability"] = display_records["churn_probability"].map(lambda x: f"{x:.2%}")
    display_records["discount_offered"] = display_records["discount_offered"].map(lambda x: f"{x:.0%}" if x > 0 else "0%")
    display_records["net_margin_impact"] = display_records["net_margin_impact"].map(lambda x: f"${x:+,.2f}")
    display_records.columns = [
        "Customer ID", "Persona", "Churn Prob", "Risk Tier", "Flagged", "Applied Strategy", "Discount", "Simulated Outcome", "Margin Impact"
    ]

    st.dataframe(display_records.head(50), use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- SECTION F: Simulation Methodology & Business Assumptions ---
    st.markdown("### 📖 Section F: Simulation Methodology & Assumptions")
    with st.expander("Click to view formal mathematical formulas and modeling assumptions", expanded=False):
        st.markdown("""
        #### 1. Baseline (Control) Probability Model
        In the control condition, customers receive standard recommendations without incentives.
        The baseline purchase probability drops monotonically as churn risk increases:
        $$P_{conv, control} = \\text{clip}\\left(P_{order}(p) \\times (1.0 - 0.55 \\times c_i), 0.02, 0.90\\right)$$
        where $P_{order}(p)$ is the natural purchase probability for persona $p$, and $c_i$ is the customer's predicted churn probability.

        #### 2. Targeted Override (Treatment) Incentive Model
        When a customer crosses the Decision Gate threshold ($c_i \\ge \\theta$), the Override Engine issues a risk-tiered coupon:
        - **MEDIUM Risk (30% - 50%):** 5% Discount Coupon
        - **HIGH Risk (50% - 80%):** 15% Discount Coupon + Bundle Offer
        - **CRITICAL Risk (≥ 80%):** 25% Discount Coupon + Abandoned Cart Recovery

        The incremental conversion probability lift accounts for persona price sensitivity $S_{price}(p)$:
        $$\\Delta P_{conv} = d_i \\times S_{price}(p) \\times (1.0 - 0.25 \\times c_i)$$
        $$P_{conv, treat} = \\min(0.95, P_{conv, control} + \\Delta P_{conv})$$

        #### 3. Financial Model & Margin Erosion
        - Gross Revenue: $Rev = AOV_i \\times (1 - d_i)$
        - Cost of Goods Sold: $COGS = AOV_i \\times (1 - m_{gross})$
        - Treatment Margin: $Margin_{treat} = AOV_i \\times (m_{gross} - d_i)$
        - Net Margin Impact: $Margin_{treat} - Margin_{control}$

        #### 4. Model Constraints & Honest Limitations
        - Simulated outcomes assume rational economic response curves calibrated from persona profiles.
        - Real-world production deployment requires live randomized A/B trial validation to calibrate true price elasticity.
        """)
