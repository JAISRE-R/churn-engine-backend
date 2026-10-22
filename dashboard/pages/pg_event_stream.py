"""
churn-engine/dashboard/pages/pg_event_stream.py
=================================================
Live Activity Page — Real-time event stream monitor.
Supports both Simulation Mode (synthetic simulator thread)
and Live Store Mode (connected to FastAPI live event buffer).
"""

from __future__ import annotations

import time
import json
import urllib.request
import urllib.error
import pandas as pd
import streamlit as st

from src.utils import load_config
from src.event_stream import StreamManager, CustomerStateManager


def fetch_live_events(api_url: str = "http://127.0.0.1:8000", limit: int = 50) -> tuple[bool, list[dict], int]:
    """Fetch recent live events from FastAPI admin endpoint."""
    url = f"{api_url.rstrip('/')}/api/admin/events/recent?limit={limit}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "StreamlitDashboard/1.0"})
        with urllib.request.urlopen(req, timeout=2.0) as res:
            if res.status == 200:
                data = json.loads(res.read().decode("utf-8"))
                return True, data.get("events", []), data.get("total_events", 0)
    except Exception:
        pass
    return False, [], 0


def render() -> None:
    """Render the Live Activity page."""
    # Ensure configuration in session state
    if "config" not in st.session_state:
        st.session_state.config = load_config()

    if "state_manager" not in st.session_state:
        st.session_state.state_manager = CustomerStateManager(st.session_state.config)

    if "stream_manager" not in st.session_state:
        st.session_state.stream_manager = StreamManager(
            st.session_state.config, st.session_state.state_manager
        )

    mode = st.session_state.get("event_source_mode", "simulation")

    if mode == "live":
        render_live_store_mode()
    else:
        render_simulation_mode()


def render_live_store_mode() -> None:
    """Render Live Store Mode powered by real FastAPI events buffer."""
    api_url = st.session_state.get("api_base_url", "http://127.0.0.1:8000")
    connected, live_events, total_events = fetch_live_events(api_url=api_url, limit=50)

    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🌐 Live Storefront Activity Feed</h1>
        <p class="page-subtitle">
            Real-time feed of live customer actions ingested via FastAPI from the e-commerce storefront.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Ingestion stats grid
    active_customers = len(set(e.get("customer_id") for e in live_events)) if live_events else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Source Connection", "🟢 CONNECTED" if connected else "🔴 OFFLINE")
    with col2:
        st.metric("Storefront Events (Total)", f"{total_events:,}")
    with col3:
        st.metric("Buffer View (Recent)", f"{len(live_events):,}")
    with col4:
        st.metric("Profiled Customers", f"{active_customers:,}")

    st.markdown("---")

    # Split view: Latest active risk updates and Churn risk alerts
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Latest Active Customer Risk")
        if connected and live_events:
            latest = live_events[0]  # Newest first
            cid = latest.get("customer_id", "Unknown")
            ev_type = latest.get("event_type", "event").replace("_", " ").title()
            pid = latest.get("product_id") or "N/A"
            region = latest.get("region", "Global")
            prob = float(latest.get("churn_probability", 0.0))
            risk_level = latest.get("risk_level", "LOW")
            decision = latest.get("decision", "NORMAL")
            strategy = latest.get("strategy", "BASELINE")
            discount = int(latest.get("discount", 0))

            # Compute delta if same customer appeared earlier in buffer
            prev_prob = None
            for e in live_events[1:]:
                if e.get("customer_id") == cid:
                    prev_prob = float(e.get("churn_probability", prob))
                    break

            prob_change = (prob - prev_prob) if prev_prob is not None else None

            st.markdown(f"**Customer:** `{cid}` &nbsp;|&nbsp; **Event:** `{ev_type}` &nbsp;|&nbsp; **Product:** `{pid}` &nbsp;|&nbsp; **Region:** `{region}`")

            # Metrics delta view
            st.metric(
                label="Churn Probability",
                value=f"{prob:.4%}",
                delta=f"{prob_change:+.2%}" if prob_change is not None and abs(prob_change) > 0.0001 else None,
                delta_color="inverse"
            )

            st.progress(min(1.0, max(0.0, prob)))

            # Render risk status badge
            color_map = {"LOW": "#1a7f37", "MEDIUM": "#9a6700", "HIGH": "#d29922", "CRITICAL": "#cf222e"}
            badge_color = color_map.get(risk_level, "#656d76")
            is_intervention = decision == "INTERVENTION_REQUIRED" or strategy == "RETENTION_OVERRIDE"

            st.markdown(f"""
            <div style="background-color: #f6f8fa; border: 1px solid #d0d7de; border-left: 5px solid {badge_color}; border-radius: 6px; padding: 0.75rem 1rem; margin-top: 0.5rem;">
                <b>Risk Category:</b> <span style="color: {badge_color}; font-weight: bold;">{risk_level}</span><br/>
                <b>Decision Gate:</b> <span style="font-weight: 600; color: {badge_color};">{"🚨 INTERVENTION REQUIRED" if is_intervention else "🟢 NORMAL SYSTEM STATUS"}</span><br/>
                <b>Strategy:</b> <code>{strategy}</code> &nbsp;|&nbsp; <b>Discount:</b> <b>{discount}%</b>
            </div>
            """, unsafe_allow_html=True)
        elif not connected:
            st.warning("Cannot reach FastAPI at `http://127.0.0.1:8000`. Ensure backend is running.")
        else:
            st.info("Awaiting live events from the Lovable storefront... (Perform actions in the store to see live updates)")

    with col_right:
        st.markdown("### Recent Interventions Triggered")
        interventions = [
            e for e in live_events
            if e.get("decision") == "INTERVENTION_REQUIRED" or e.get("strategy") == "RETENTION_OVERRIDE"
        ] if live_events else []

        if interventions:
            alerts_data = []
            for r in interventions[:5]:
                alerts_data.append({
                    "Timestamp": r.get("timestamp", "")[-8:] if len(r.get("timestamp", "")) >= 8 else r.get("timestamp", ""),
                    "Customer ID": r.get("customer_id", ""),
                    "Risk Score": f"{float(r.get('churn_probability', 0.0)):.2%}",
                    "Risk Tier": r.get("risk_level", "HIGH"),
                    "Strategy": r.get("strategy", "RETENTION_OVERRIDE"),
                    "Discount": f"{r.get('discount', 0)}%",
                })
            st.dataframe(pd.DataFrame(alerts_data), use_container_width=True, hide_index=True)
        else:
            st.success("No active interventions triggered in recent live storefront events.")

    st.markdown("---")

    # Live Feed Table
    st.markdown("### Live Storefront Events Feed")
    if live_events:
        table_rows = []
        for e in live_events[:25]:
            ts = e.get("timestamp", "")
            if "T" in ts:
                time_str = ts.split("T")[1].split(".")[0]
            elif " " in ts:
                time_str = ts.split(" ")[1]
            else:
                time_str = ts

            table_rows.append({
                "Timestamp": time_str,
                "Customer ID": e.get("customer_id", ""),
                "Event Type": e.get("event_type", ""),
                "Product ID": e.get("product_id") or "-",
                "Session ID": e.get("session_id", ""),
                "Churn Risk": f"{float(e.get('churn_probability', 0.0)):.2%}",
                "Risk Level": e.get("risk_level", "LOW"),
                "Decision": e.get("decision", "NORMAL"),
                "Strategy": e.get("strategy", "BASELINE"),
                "Discount": f"{e.get('discount', 0)}%",
                "Region": e.get("region", "-"),
            })
        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.caption("Awaiting live behavioral events from the storefront...")

    # Auto refresh loop for live monitoring
    refresh_rate = float(st.session_state.config["event_stream"].get("stream_refresh_interval", 1.5))
    time.sleep(refresh_rate)
    st.rerun()


def render_simulation_mode() -> None:
    """Render existing Simulation Mode powered by synthetic StreamManager."""
    sm: StreamManager = st.session_state.stream_manager
    stats = sm.get_stats()
    is_running = stats["status"] == "running"

    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">⚡ Synthetic Behavioral Activity Feed</h1>
        <p class="page-subtitle">
            Real-time feed of ingested behavioral events, live inference scoring, and immediate intervention alerts (Simulation Mode).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Ingestion stats grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Simulation Status", "🟢 ACTIVE" if is_running else "⏸️ INACTIVE")
    with col2:
        st.metric("Events Ingested (Session)", f"{stats['total_events']:,}")
    with col3:
        st.metric("Ingestion Velocity", f"{stats['events_per_second']:.2f} ev/sec")
    with col4:
        st.metric("Active Customers Profiled", f"{stats['active_customers_count']:,}")

    st.markdown("---")

    # Split view: Latest active risk updates and Churn risk alerts
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Latest Active Customer Risk")
        latest_event = stats["latest_event"]
        
        if latest_event is not None:
            active_cid = latest_event.customer_id
            pred = sm.latest_predictions.get(active_cid, {})
            
            if pred:
                # Find history and calculate delta
                hist = list(sm.prediction_history.get(active_cid, []))
                prob = pred["churn_probability"]
                threshold = pred["threshold"]
                risk_level = pred["risk_level"]
                status = pred["intervention_status"]

                if len(hist) >= 2:
                    prev_prob = hist[-2]["churn_probability"]
                else:
                    prev_prob = prob

                prob_change = prob - prev_prob

                st.markdown(f"**Customer:** `{active_cid}` &nbsp;&nbsp;|&nbsp;&nbsp; **Event:** `{latest_event.event_type.replace('_', ' ').title()}`")

                # Metrics delta view
                st.metric(
                    label="Churn Probability",
                    value=f"{prob:.4%}",
                    delta=f"{prob_change:+.2%}" if abs(prob_change) > 0.0001 else None,
                    delta_color="inverse"
                )

                st.progress(prob)
                
                # Render risk status badge
                color_map = {"LOW": "#1a7f37", "MEDIUM": "#9a6700", "HIGH": "#d29922", "CRITICAL": "#cf222e"}
                badge_color = color_map.get(risk_level, "#656d76")
                
                st.markdown(f"""
                <div style="background-color: #f6f8fa; border: 1px solid #d0d7de; border-left: 5px solid {badge_color}; border-radius: 6px; padding: 0.75rem 1rem; margin-top: 0.5rem;">
                    <b>Risk Category:</b> <span style="color: {badge_color}; font-weight: bold;">{risk_level}</span><br/>
                    <b>Decision Gate:</b> <span style="font-weight: 600; color: {badge_color};">{"🚨 INTERVENTION REQUIRED" if status == "INTERVENTION_REQUIRED" else "🟢 NORMAL SYSTEM STATUS"}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.caption("Awaiting risk prediction on incoming event...")
        else:
            st.caption("No events ingested yet. Visit Simulation Settings to start the simulator.")

    with col_right:
        st.markdown("### Recent Interventions Triggered")
        alerts = sm.override_engine.get_audit_log()
        if alerts:
            alerts_data = []
            for r in reversed(alerts):
                alerts_data.append({
                    "Timestamp": r["timestamp"],
                    "Customer ID": r["customer_id"],
                    "Risk Score": f"{r['churn_probability']:.2%}",
                    "Risk Tier": r["risk_level"],
                    "Discount Code": f"COUPON-CRIT-25" if r["risk_level"] == "CRITICAL" else f"COUPON-SOFT-{int(r['discount_value']*100)}"
                })
                if len(alerts_data) >= 5:
                    break
            st.dataframe(pd.DataFrame(alerts_data), use_container_width=True, hide_index=True)
        else:
            st.success("No active interventions triggered in the last 1000 events.")

    st.markdown("---")

    # Live Feed Table
    st.markdown("### Live Ingested Events Feed")
    df_feed = sm.get_recent_events(20)
    if len(df_feed) > 0:
        display_cols = ["event_timestamp", "customer_id", "event_type", "product_id", "session_id", "device_type"]
        available = [c for c in display_cols if c in df_feed.columns]
        df_styled = df_feed[available].copy()
        if "event_timestamp" in df_styled.columns:
            df_styled["event_timestamp"] = pd.to_datetime(df_styled["event_timestamp"]).dt.strftime("%H:%M:%S")
        df_styled.columns = [
            {"event_timestamp": "Timestamp", "customer_id": "Customer ID",
             "event_type": "Event Type", "product_id": "Product ID",
             "session_id": "Session ID", "device_type": "Device"}.get(c, c)
            for c in df_styled.columns
        ]
        st.dataframe(
            df_styled.iloc[::-1], # Newest first
            use_container_width=True,
            hide_index=True
        )
    else:
        st.caption("Awaiting simulated behavioral events...")

    # Thread Refresh Loop
    if is_running:
        time.sleep(float(st.session_state.config["event_stream"].get("stream_refresh_interval", 1.5)))
        st.rerun()
