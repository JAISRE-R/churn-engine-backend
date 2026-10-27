"""
churn-engine/dashboard/pages/pg_decision_gate.py
=================================================
Decision Gate Page — Threshold Control & Override Audit.
"""

from __future__ import annotations

import time
import pandas as pd
import streamlit as st

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
    threshold_val = dg.threshold

    # Calculate statistics
    total_decisions = len(history)
    override_count = sum(1 for d in history if d["decision"] == "INTERVENTION_REQUIRED")
    override_rate = (override_count / total_decisions * 100.0) if total_decisions > 0 else 0.0

    st.markdown("### Decision pipeline")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Decision Threshold", f"{threshold_val:.2%}", help="Loaded dynamically from models/decision_policy.json")
    with col2:
        st.metric("Decisions Audited", f"{total_decisions:,}" if history else "No data")
    with col3:
        st.metric(
            "Overrides Fired",
            f"{override_count:,} ({override_rate:.1f}%)" if history else "No data",
        )

    st.markdown("### Decision audit trail")
    st.caption("In-memory decision history for the current dashboard runtime.")
    
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
