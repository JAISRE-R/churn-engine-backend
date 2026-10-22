"""
churn-engine/dashboard/pages/pg_sim_settings.py
===============================================
Simulation Settings Page — Control the real-time event generator.
"""

from __future__ import annotations

import streamlit as st
from src.utils import load_config
from src.event_stream import StreamManager, CustomerStateManager


def render() -> None:
    """Render the Simulation Settings page."""
    # Ensure StreamManager is initialised in session state
    if "config" not in st.session_state:
        st.session_state.config = load_config()

    if "state_manager" not in st.session_state:
        st.session_state.state_manager = CustomerStateManager(st.session_state.config)

    if "stream_manager" not in st.session_state:
        st.session_state.stream_manager = StreamManager(
            st.session_state.config, st.session_state.state_manager
        )

    sm: StreamManager = st.session_state.stream_manager
    stats = sm.get_stats()
    is_running = stats["status"] == "running"

    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Simulation Settings</h1>
        <p class="page-subtitle">
            Configure the customer behavior simulator and control the background event pipeline.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎛️ Simulator Controls")
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        if st.button("▶️ Start Simulator Stream", use_container_width=True, disabled=is_running, type="primary"):
            sm.start()
            st.rerun()
    with col_ctrl2:
        if st.button("⏸️ Stop Simulator Stream", use_container_width=True, disabled=not is_running):
            sm.stop()
            st.rerun()

    st.markdown("---")

    st.markdown("### ⚙️ Simulator Hyperparameters")
    
    col_slide1, col_slide2 = st.columns(2)
    with col_slide1:
        freq = st.slider(
            "Base Event Frequency (Hz)",
            min_value=0.5,
            max_value=10.0,
            value=float(st.session_state.config["event_stream"].get("event_frequency", 2.0)),
            step=0.5,
            help="Determines the target number of events generated per second."
        )
        sm.simulator.rate_hz = freq

    with col_slide2:
        speed = st.slider(
            "Simulation Speed Multiplier",
            min_value=1.0,
            max_value=20.0,
            value=float(st.session_state.config["event_stream"].get("simulation_speed", 1.0)),
            step=1.0,
            help="Accelerates the simulator's internal clock relative to actual elapsed time."
        )
        sm.simulator.simulation_speed = speed

    st.markdown("---")

    # Display status summary
    st.markdown("### 📋 Current Simulator Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pipeline State", "Active (LIVE)" if is_running else "Idle (STOPPED)")
    with col2:
        st.metric("Events Per Second (Target)", f"{freq * speed:.1f} ev/s")
    with col3:
        st.metric("Buffer Capacity", f"{sm.buffer.max_size} events")

    st.info(
        "ℹ️ **Operational Note:** In production, the simulator is replaced by an external event broker (e.g., Kafka or RabbitMQ). "
        "The background streaming worker runs continuously to process user behavioral signals, update features, "
        "predict churn probability on-the-fly, and feed the Decision Gate and Override Engine."
    )
