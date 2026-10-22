"""
churn-engine/dashboard/pages/pg_home.py
==========================================
Overview Page — Executive Churn & Intervention Dashboard.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils import load_config
from src.event_stream import StreamManager, CustomerStateManager


def render() -> None:
    """Render the Overview/Dashboard page."""

    if "config" not in st.session_state:
        st.session_state.config = load_config()

    if "state_manager" not in st.session_state:
        st.session_state.state_manager = CustomerStateManager(st.session_state.config)

    if "stream_manager" not in st.session_state:
        st.session_state.stream_manager = StreamManager(
            st.session_state.config, st.session_state.state_manager
        )

    sm = st.session_state.stream_manager
    state = st.session_state.state_manager
    stats = sm.get_stats()
    is_running = stats["status"] == "running"

    # Pre-populate predictions if empty so the dashboard has real data at startup
    if not sm.latest_predictions:
        for cid in list(state.customer_history.keys())[:30]:
            feats = state.get_features(cid)
            pred_res = sm.predictor.predict(cid, feats)
            sm.latest_predictions[cid] = pred_res
            # Initialize history
            from collections import deque
            hist = sm.prediction_history.setdefault(cid, deque(maxlen=sm.risk_history_size))
            hist.append({
                "timestamp": pred_res["timestamp"],
                "churn_probability": pred_res["churn_probability"],
                "threshold": pred_res["threshold"],
                "risk_level": pred_res["risk_level"],
                "intervention_status": pred_res["intervention_status"]
            })

    # Header / Title Block
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Live Customer Churn Intelligence</h1>
        <p class="page-subtitle">
            Real-time customer risk detection, Decision Gate routing, and retention recommendation overrides.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Status Bar
    if is_running:
        st.markdown("""
        <div style="background-color: #dafbe1; border: 1px solid #1a7f37; border-radius: 6px; padding: 0.6rem 1rem; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.6rem;">
            <span style="color: #1a7f37; font-size: 1.2rem; line-height: 1;">●</span>
            <span style="font-weight: 700; color: #1a7f37;">LIVE STREAM ACTIVE</span>
            <span style="color: #424a53; font-size: 0.88rem;">— Behavioral events are being ingested and scored in real-time</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #ffebe9; border: 1px solid #cf222e; border-radius: 6px; padding: 0.6rem 1rem; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 0.6rem;">
            <span style="color: #cf222e; font-size: 1.2rem; line-height: 1;">●</span>
            <span style="font-weight: 700; color: #cf222e;">STREAM STOPPED</span>
            <span style="color: #424a53; font-size: 0.88rem;">— Event generator is currently paused. Visit Simulation Settings to restart.</span>
        </div>
        """, unsafe_allow_html=True)

    # Metrics
    total_customers = len(state.customer_history)
    loaded_event_count = getattr(
        state,
        "loaded_event_count",
        max(0, sum(len(history) for history in state.customer_history.values()) - stats["total_events"]),
    )
    total_events = loaded_event_count + stats["total_events"]
    at_risk_count = sum(
        1 for p in sm.latest_predictions.values()
        if p["churn_probability"] >= p["threshold"]
    )
    interventions_triggered = len(sm.override_engine.get_audit_log())

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Customers", f"{total_customers:,}", help="Total unique customer accounts tracked in system")
    with col_m2:
        st.metric("Events Processed", f"{total_events:,}", help="Cumulative behavioral events logged")
    with col_m3:
        st.metric("Customers At Risk", f"{at_risk_count}", help="Customers currently above intervention threshold")
    with col_m4:
        st.metric("Interventions Triggered", f"{interventions_triggered}", help="Active overrides executed by recommendation engine")

    st.markdown("---")

    # Customer Risk Profile Summary (Low, Medium, High Risk)
    st.markdown("### Customer Risk Registry")
    
    # Categorize current state
    low_risk_list = []
    med_risk_list = []
    high_risk_list = []

    for cid, pred in sm.latest_predictions.items():
        prob = pred["churn_probability"]
        thresh = pred["threshold"]
        risk_lvl = pred["risk_level"]
        decision = pred["intervention_status"]
        
        # Get recommendation strategy summary
        rec_data = sm.latest_recommendations.get(cid)
        if rec_data:
            rec_strategy = rec_data.get("strategy", "BASELINE")
            recs_list = [r["product_id"] for r in rec_data.get("recommendations", [])]
            if rec_strategy == "RETENTION_OVERRIDE":
                discount_pct = rec_data.get("metadata", {}).get("discount_pct", 0)
                rec_action = f"Offer {discount_pct}% Discount Coupon & Product Bundle"
            else:
                rec_action = "Serve standard recommendation list"
        else:
            rec_action = "Standard recommendation list"

        cust_info = {
            "Customer ID": cid,
            "Churn Risk": prob,
            "Risk Tier": risk_lvl,
            "Decision": "INTERVENTION" if decision == "INTERVENTION_REQUIRED" else "NORMAL",
            "Action": rec_action
        }

        if prob >= thresh:
            high_risk_list.append(cust_info)
        elif risk_lvl == "MEDIUM":
            med_risk_list.append(cust_info)
        else:
            low_risk_list.append(cust_info)

    # Render side-by-side columns
    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        st.markdown("<h4 style='color: #2ea44f;'>🟢 Low Churn Risk</h4>", unsafe_allow_html=True)
        if low_risk_list:
            df_low = pd.DataFrame(low_risk_list).sort_values("Churn Risk", ascending=True).head(5)
            df_low["Churn Risk"] = df_low["Churn Risk"].map(lambda x: f"{x:.2%}")
            st.dataframe(df_low[["Customer ID", "Churn Risk", "Action"]], use_container_width=True, hide_index=True)
        else:
            st.caption("No customers currently in Low Risk tier.")

    with col_c2:
        st.markdown("<h4 style='color: #dbab09;'>🟡 Medium Churn Risk</h4>", unsafe_allow_html=True)
        if med_risk_list:
            df_med = pd.DataFrame(med_risk_list).sort_values("Churn Risk", ascending=False).head(5)
            df_med["Churn Risk"] = df_med["Churn Risk"].map(lambda x: f"{x:.2%}")
            st.dataframe(df_med[["Customer ID", "Churn Risk", "Action"]], use_container_width=True, hide_index=True)
        else:
            st.caption("No customers currently in Medium Risk tier.")

    with col_c3:
        st.markdown("<h4 style='color: #f85149;'>🔴 High Churn Risk</h4>", unsafe_allow_html=True)
        if high_risk_list:
            df_high = pd.DataFrame(high_risk_list).sort_values("Churn Risk", ascending=False).head(5)
            df_high["Churn Risk"] = df_high["Churn Risk"].map(lambda x: f"{x:.2%}")
            st.dataframe(df_high[["Customer ID", "Churn Risk", "Action"]], use_container_width=True, hide_index=True)
        else:
            st.caption("No customers currently in High Risk tier.")

    st.markdown("---")

    # Recent Activity with Risk Deltas
    st.markdown("### Recent System Activity & Risk Trajectory Changes")
    
    df_recent = sm.get_recent_events(8)
    if not df_recent.empty:
        activity_rows = []
        for _, row in df_recent.iterrows():
            cid = row["customer_id"]
            ts = pd.to_datetime(row["event_timestamp"]).strftime("%H:%M:%S")
            etype = row["event_type"]
            
            # Find current probability and delta
            hist = list(sm.prediction_history.get(cid, []))
            if len(hist) >= 2:
                prob = hist[-1]["churn_probability"]
                prev_prob = hist[-2]["churn_probability"]
                delta = prob - prev_prob
            else:
                prob = sm.latest_predictions.get(cid, {}).get("churn_probability", 0.0)
                delta = 0.0
            
            delta_str = f"{delta:+.2%}" if abs(delta) > 0.0001 else "0.00%"
            risk_tier = sm.latest_predictions.get(cid, {}).get("risk_level", "LOW")
            
            activity_rows.append({
                "Timestamp": ts,
                "Customer ID": cid,
                "Behavioral Event": etype.replace("_", " ").title(),
                "Risk Level": risk_tier,
                "Current Risk": f"{prob:.2%}",
                "Risk Shift": delta_str
            })
            
        df_act = pd.DataFrame(activity_rows)
        st.dataframe(df_act.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.caption("Waiting for simulation stream activity...")

    # Thread Refresh Loop for Dashboard (if running)
    if is_running:
        import time
        time.sleep(float(st.session_state.config["event_stream"].get("stream_refresh_interval", 1.5)))
        st.rerun()
