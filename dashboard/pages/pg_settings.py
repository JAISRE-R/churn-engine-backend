"""
churn-engine/dashboard/pages/pg_settings.py
============================================
Settings Page — Display System & Model Configuration.
"""

from __future__ import annotations

import streamlit as st
from src.utils import load_config
from src.event_stream import StreamManager, CustomerStateManager


def render() -> None:
    """Render the Settings page."""

    if "config" not in st.session_state:
        st.session_state.config = load_config()

    if "state_manager" not in st.session_state:
        st.session_state.state_manager = CustomerStateManager(st.session_state.config)

    if "stream_manager" not in st.session_state:
        st.session_state.stream_manager = StreamManager(
            st.session_state.config, st.session_state.state_manager
        )

    sm = st.session_state.stream_manager
    config = st.session_state.config

    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">⚙️ Settings</h1>
        <p class="page-subtitle">
            System, Simulator, and Machine Learning Model Configurations
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Ingestion & Simulator Config")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Simulation Interval (sec)", f"{config['event_stream'].get('stream_refresh_interval', 1.0)}s")
        st.metric("Risk History Window", f"{config['event_stream'].get('risk_history_size', 100)} records")
    with col2:
        st.metric("Max Recent Events Buffer", f"{config['event_stream'].get('max_recent_events', 1000)} events")
        st.metric("Customer Ingestion Limit", f"{config['event_stream'].get('active_user_pool_size', 50)} profiles")
    with col3:
        st.metric("Deterministic Seed", f"{config['synthetic_data'].get('random_state', 42)}")
        st.metric("Inference Batch Mode", "Real-Time / Streaming")

    st.markdown("---")

    st.markdown("### 🚦 Authoritative Intervention Control")
    col_gate1, col_gate2 = st.columns(2)
    with col_gate1:
        st.metric("Active Decision Threshold", f"{sm.decision_gate.threshold:.2%}")
        st.caption("🚨 Source of Truth: `models/decision_policy.json` (or config fallback)")
    with col_gate2:
        st.metric("Target Strategy Engine", "Override Intercept Middleware")
        st.caption("🔴 Activates when Customer Churn Risk >= Threshold")

    st.markdown("---")

    st.markdown("### 🧠 ML Model Specifications")
    col_ml1, col_ml2 = st.columns(2)
    with col_ml1:
        winning_model = sm.predictor.model.model.__class__.__name__ if sm.predictor.model else "XGBClassifier"
        st.text_input("Winning Model Pipeline", value=winning_model, disabled=True)
        st.text_input("Model Weights Path", value=f"models/churn_model.joblib", disabled=True)
    with col_ml2:
        st.text_input("Inference Scaling API", value="StandardScaler (pipeline wrapped)", disabled=True)
        st.text_input("Policy Settings Path", value="models/decision_policy.json", disabled=True)

    st.markdown("---")

    st.markdown("### 🗂️ Active Configurations (raw YAML view)")
    st.json(config)
