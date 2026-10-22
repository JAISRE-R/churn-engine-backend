"""
churn-engine/dashboard/pages/pg_decision_gate.py
=================================================
Decision Gate Page — Threshold Control & Override Audit.
"""

from __future__ import annotations

import time
import pandas as pd
import streamlit as st
from datetime import datetime

from src.utils import load_config
from src.event_stream import StreamManager, CustomerStateManager
from src.decision_gate import DecisionGate


def render() -> None:
    """Render the Decision Gate / System Diagnostics page."""

    if "config" not in st.session_state:
        st.session_state.config = load_config()

    if "state_manager" not in st.session_state:
        st.session_state.state_manager = CustomerStateManager(st.session_state.config)

    if "stream_manager" not in st.session_state:
        st.session_state.stream_manager = StreamManager(
            st.session_state.config, st.session_state.state_manager
        )

    sm: StreamManager = st.session_state.stream_manager
    dg: DecisionGate = sm.decision_gate
    stats = sm.get_stats()
    is_running = stats["status"] == "running"

    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2.2rem; font-weight: 800; margin: 0; line-height: 1.2;">
            System Diagnostics &amp; Decision Gate
        </h1>
        <p style="font-size: 1.1rem; color: #8b949e; margin: 0.25rem 0 0 0; font-weight: 400;">
            Verify pipeline health, Decision Gate threshold settings, and live override distributions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load history from Decision Gate
    history = dg.get_history()
    crossing_events = dg.get_crossing_events()
    threshold_val = dg.threshold

    # Calculate statistics
    total_decisions = len(history)
    override_count = sum(1 for d in history if d["decision"] == "INTERVENTION_REQUIRED")
    normal_count = total_decisions - override_count
    override_rate = (override_count / total_decisions * 100.0) if total_decisions > 0 else 0.0
    avg_probability = (sum(d["churn_probability"] for d in history) / total_decisions) if total_decisions > 0 else 0.0

    # Get active customers above threshold from latest decisions
    active_above = [
        {
            "Customer ID": cid,
            "Latest Probability": f"{dec['churn_probability']:.4%}",
            "Risk Tier": dec["risk_level"],
            "Timestamp": dec["timestamp"]
        }
        for cid, dec in sm.latest_decisions.items()
        if dec["decision"] == "INTERVENTION_REQUIRED"
    ]

    # Row 1: Key Metrics
    st.markdown("### 📊 Decision Pipeline Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Decision Threshold", f"{threshold_val:.2%}", help="Loaded dynamically from models/decision_policy.json")
    with col2:
        st.metric("Decisions Audited", f"{total_decisions:,}")
    with col3:
        st.metric("Overrides Fired", f"{override_count:,} ({override_rate:.1f}%)")
    with col4:
        st.metric("Avg Churn Prob", f"{avg_probability:.2%}")

    st.markdown("---")

    # Row 2: Split Columns
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🚨 Threshold Crossing Events")
        st.caption("Customers who transitioned from below threshold to INTERVENTION_REQUIRED on their latest event.")
        
        if crossing_events:
            crossing_data = []
            for ev in reversed(crossing_events):
                crossing_data.append({
                    "Timestamp": ev["timestamp"],
                    "Customer ID": ev["customer_id"],
                    "Prev Prob": f"{ev['previous_probability']:.2%}",
                    "New Prob": f"{ev['churn_probability']:.2%}",
                    "Status": "⚠️ JUST CROSSED!"
                })
            
            st.dataframe(
                pd.DataFrame(crossing_data),
                use_container_width=True,
                hide_index=True
            )
            # Add visual notification of the latest crossing event
            latest_cross = crossing_events[-1]
            st.toast(
                f"🚨 Customer {latest_cross['customer_id']} crossed the threshold! "
                f"({latest_cross['previous_probability']:.2%} ➔ {latest_cross['churn_probability']:.2%})",
                icon="⚠️"
            )
        else:
            st.success("No threshold crossing transitions detected in the current run.")

        st.markdown("### 📈 Churn Probability Distribution")
        if history:
            probs = [d["churn_probability"] for d in history]
            bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            counts, _ = pd.cut(probs, bins=bins, retbins=True)
            dist_series = counts.value_counts().sort_index()
            dist_series.index = ["0%-20%", "20%-40%", "40%-60%", "60%-80%", "80%-100%"]
            st.bar_chart(dist_series)
        else:
            st.info("No probability distributions available. Start streaming events.")

    with col_right:
        st.markdown("### 👥 Active Flagged Profiles")
        st.caption("Total unique customers currently requiring intervention (probability >= threshold).")
        
        if active_above:
            st.dataframe(
                pd.DataFrame(active_above),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("All active customer profiles are currently within NORMAL risk limits.")

    st.markdown("---")

    # Row 3: Decision Log
    st.markdown("### 🗂️ Decision Audit Trail")
    st.caption("Cumulative log of all decisions evaluated by the DecisionGate rules engine.")
    
    if history:
        log_df = pd.DataFrame(history)[["timestamp", "customer_id", "churn_probability", "previous_probability", "decision", "threshold_crossed", "risk_level"]]
        log_df.columns = ["Timestamp", "Customer ID", "Risk Prob", "Prev Prob", "Decision", "Crossed?", "Risk Tier"]
        log_df["Risk Prob"] = log_df["Risk Prob"].map(lambda x: f"{x:.2%}")
        log_df["Prev Prob"] = log_df["Prev Prob"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "—")
        log_df["Crossed?"] = log_df["Crossed?"].map(lambda x: "⚠️ Yes" if x else "No")
        
        st.dataframe(
            log_df.iloc[::-1], # Newest first
            use_container_width=True,
            hide_index=True
        )
    else:
        st.caption("Awaiting decision audit logs...")

    if is_running:
        time.sleep(float(st.session_state.config["event_stream"].get("stream_refresh_interval", 1.5)))
        st.rerun()
