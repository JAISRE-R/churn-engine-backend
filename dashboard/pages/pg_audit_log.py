"""
churn-engine/dashboard/pages/pg_audit_log.py
============================================
Audit Log Page — View recommendation override decision records and end-to-end lifecycle timeline.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.event_stream import CustomerStateManager, StreamManager
from src.response_simulator import InterventionSimulator, get_customer_persona
from src.utils import load_config


def render() -> None:
    """Render the Audit Log page with full 6-step intervention timeline."""

    # Initialization
    if "config" not in st.session_state:
        st.session_state.config = load_config()
    if "state_manager" not in st.session_state:
        st.session_state.state_manager = CustomerStateManager(st.session_state.config)
    if "stream_manager" not in st.session_state:
        st.session_state.stream_manager = StreamManager(
            st.session_state.config, st.session_state.state_manager
        )

    sm: StreamManager = st.session_state.stream_manager
    audit_data = sm.override_engine.get_audit_log()

    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Intervention Audit Logs</h1>
        <p class="page-subtitle">
        In-memory records from the current dashboard runtime. Simulated responses are labeled separately.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Status Bar
    stats = sm.get_stats()
    is_running = stats["status"] == "running"
    status_indicator = "🟢 LIVE FEED ACTIVE" if is_running else "⏸️ STREAM IDLE"
    st.markdown(
        f"<div style='background-color:#f6f8fa; border:1px solid #d0d7de; border-radius:6px; padding:0.6rem 1rem; margin-bottom:1.5rem; font-size:0.88rem; color:#1f2328;'>"
        f"<strong>Stream Status:</strong> <code>{status_indicator}</code> &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<strong>Total Override Interventions Recorded:</strong> <code>{len(audit_data):,}</code>"
        f"</div>",
        unsafe_allow_html=True
    )

    if not audit_data:
        st.info("No override interventions have been triggered yet. When a customer's churn risk crosses the threshold, audit records and the 6-step lifecycle timeline will populate here.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(audit_data)

    # Search / Filter tools
    st.markdown("### 🔎 Filter Audit Trail")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        search_cid = st.text_input("Search Customer ID", placeholder="e.g. C-1002").strip()
    with col_f2:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=["MEDIUM", "HIGH", "CRITICAL"],
            default=["MEDIUM", "HIGH", "CRITICAL"]
        )

    # Apply filters
    if search_cid:
        df = df[df["customer_id"].str.contains(search_cid, case=False)]
    if risk_filter:
        df = df[df["risk_level"].isin(risk_filter)]

    if df.empty:
        st.warning("No audit records matched the selected filters.")
        return

    # Select columns to display
    display_df = df[[
        "timestamp",
        "customer_id",
        "churn_probability",
        "threshold",
        "risk_level",
        "final_strategy",
        "discount_value",
        "override_reason"
    ]].copy()

    # Formatting columns
    display_df["churn_probability"] = display_df["churn_probability"].map(lambda x: f"{x:.2%}")
    display_df["threshold"] = display_df["threshold"].map(lambda x: f"{x:.2%}")
    display_df["discount_value"] = display_df["discount_value"].map(lambda x: f"{x:.0%}" if x > 0 else "0%")
    display_df.columns = [
        "Timestamp", "Customer ID", "Churn Prob", "Threshold", "Risk Level", "Strategy", "Configured discount rate", "Override Reason"
    ]

    st.markdown("### 📋 Override Intervention Records")
    st.dataframe(
        display_df.iloc[::-1],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # --- 6-STEP INTERVENTION LIFECYCLE TIMELINE ---
    st.markdown("### ⏱️ End-to-End Intervention Lifecycle Timeline")
    st.caption("Demonstrating the full flow: Customer Event → Updated Risk → Decision Gate → Recommendation → Override → Simulated Response")

    record_timestamps = df["timestamp"].iloc[::-1].tolist()
    selected_ts = st.selectbox(
        "Choose an intervention event to trace through the pipeline:",
        options=record_timestamps,
        index=0
    )

    if selected_ts:
        rec = df[df["timestamp"] == selected_ts].iloc[0]
        cid = rec["customer_id"]
        prob = float(rec["churn_probability"])
        thresh = float(rec["threshold"])
        risk_lvl = rec["risk_level"]
        discount_record = rec.get("discount_value")
        resp = None
        if discount_record is not None and pd.notna(discount_record):
            sim = InterventionSimulator(config=st.session_state.config, random_seed=42)
            resp = sim.simulate_single_event_response(
                customer_id=cid,
                churn_probability=prob,
                risk_level=risk_lvl,
                discount_offered=float(discount_record),
                persona=get_customer_persona(cid),
            )
        discount_val = float(discount_record) if discount_record is not None and pd.notna(discount_record) else None

        st.markdown(f"#### Pipeline record: Customer `{cid}` at `{selected_ts}`")

        # Visual 6-Step Timeline Cards
        steps = [
            {
                "num": "1",
                "stage": "Customer Event Ingestion",
                "summary": "Behavioral Signal Ingested",
                "detail": f"Real-time event recorded for {cid}. Session activity processed by state manager.",
                "color": "#0969da",
                "bg": "#ddf4ff",
            },
            {
                "num": "2",
                "stage": "Real-Time Churn Scoring",
                "summary": f"Risk Score: {prob:.2%}",
                "detail": f"Model inference classified customer as {risk_lvl} risk tier.",
                "color": "#9a6700" if prob < 0.8 else "#cf222e",
                "bg": "#fff8c5" if prob < 0.8 else "#ffebe9",
            },
            {
                "num": "3",
                "stage": "Decision Gate Middleware",
                "summary": f"Recorded gate decision ({prob:.2%} vs. {thresh:.2%})",
                "detail": "Intervention requirement verified against authoritative policy threshold.",
                "color": "#cf222e",
                "bg": "#ffebe9",
            },
            {
                "num": "4",
                "stage": "Baseline Recommender",
                "summary": "Baseline Catalog Formulated",
                "detail": "Standard collaborative filtering items compiled prior to intervention gate.",
                "color": "#656d76",
                "bg": "#f6f8fa",
            },
            {
                "num": "5",
                "stage": "Retention Override Engine",
                "summary": (
                    f"Configured tier rate: {discount_val:.0%}"
                    if discount_val is not None else "Configured tier rate unavailable"
                ),
                "detail": "See the recorded recommended product IDs below; this page does not infer a cart-wide offer.",
                "color": "#0969da",
                "bg": "#ddf4ff",
            },
            {
                "num": "6",
                "stage": "Simulated response (not observed)",
                "summary": (
                    f"Simulated outcome: {resp['outcome_status'].replace('_', ' ')}"
                    if resp else "Unavailable"
                ),
                "detail": (
                    f"Simulated revenue: ${resp['simulated_revenue']:.2f} | "
                    f"Discount cost: ${resp['discount_cost']:.2f} | "
                    f"Margin: ${resp['margin_contribution']:+.2f}"
                    if resp else "No recorded discount value is available for this simulation."
                ),
                "color": "#1a7f37" if resp and resp["simulated_converted"] else "#656d76",
                "bg": "#dafbe1" if resp and resp["simulated_converted"] else "#f6f8fa",
            },
        ]

        for s in steps:
            st.markdown(f"""
            <div style="
                display: flex;
                align-items: flex-start;
                gap: 1rem;
                background-color: #ffffff;
                border: 1px solid #d0d7de;
                border-left: 5px solid {s['color']};
                border-radius: 8px;
                padding: 0.85rem 1.25rem;
                margin-bottom: 0.6rem;
                box-shadow: 0 1px 2px rgba(0,0,0,0.03);
            ">
                <div style="
                    background-color: {s['bg']};
                    color: {s['color']};
                    font-weight: 800;
                    font-size: 0.95rem;
                    width: 32px;
                    height: 32px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-shrink: 0;
                ">
                    {s['num']}
                </div>
                <div style="flex: 1;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #656d76; letter-spacing: 0.05em;">
                            STEP {s['num']}: {s['stage']}
                        </span>
                        <span style="font-size: 0.85rem; font-weight: 700; color: {s['color']};">
                            {s['summary']}
                        </span>
                    </div>
                    <div style="font-size: 0.88rem; color: #1f2328; margin-top: 0.2rem;">
                        {s['detail']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**Injected Retention Offer Payload:**")
        rec_items = rec.get("recommended_products", [])
        st.write(", ".join([f"`{p}`" for p in rec_items]))
