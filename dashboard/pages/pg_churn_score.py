"""
churn-engine/dashboard/pages/pg_churn_score.py
================================================
Customer Risk Page — Customer 360 View, Risk Diagnostics & Explainability.
"""

from __future__ import annotations

from collections import deque
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.event_stream import CustomerStateManager, StreamManager
from src.demo_mode import demo_classification, local_demo_enabled
from src.utils import load_config


def explain_customer_risk(df_f: pd.DataFrame, cid: str, prob: float, threshold: float) -> list[dict[str, str]]:
    """Generate explainable behavioral drivers for the selected customer.

    Evaluates the customer's engineered features against behavioral thresholds
    to identify the specific reasons why churn risk is elevated or suppressed.
    """
    reasons = []
    row = df_f.loc[cid]

    def feature_value(name: str) -> float | None:
        value = row.get(name)
        return float(value) if value is not None and pd.notna(value) else None

    recency = feature_value("recency_days")
    purch_recency = feature_value("days_since_last_purchase")
    abandon = feature_value("cart_abandonment_rate")
    tickets = feature_value("support_ticket_count")
    events_7d = feature_value("events_7d")

    if prob >= threshold:
        if recency is not None and recency > 30.0:
            reasons.append({
                "severity": "HIGH",
                "title": "Extended Browsing Inactivity",
                "detail": f"Customer has been inactive for {recency:.1f} days without session activity."
            })
        if purch_recency is not None and purch_recency > 45.0:
            reasons.append({
                "severity": "HIGH",
                "title": "Purchase Dormancy",
                "detail": f"Last completed transaction occurred {purch_recency:.1f} days ago."
            })
        if abandon is not None and abandon >= 0.50:
            reasons.append({
                "severity": "MEDIUM",
                "title": "Severe Cart Abandonment",
                "detail": f"Recorded cart abandonment rate is {abandon*100:.1f}%."
            })
        if tickets is not None and tickets >= 2:
            reasons.append({
                "severity": "MEDIUM",
                "title": "Customer Support Friction",
                "detail": f"Recorded support interactions: {tickets:.0f}."
            })
        if events_7d is not None and events_7d == 0:
            reasons.append({
                "severity": "MEDIUM",
                "title": "Velocity Drop-Off (Past 7 Days)",
                "detail": "Zero behavioral events recorded during the rolling 7-day observation window."
            })
    else:
        if recency is not None and recency <= 14.0:
            reasons.append({
                "severity": "GOOD",
                "title": "Active Engagement",
                "detail": f"Recent session activity within the last {recency:.1f} days."
            })
        if purch_recency is not None and purch_recency <= 30.0:
            reasons.append({
                "severity": "GOOD",
                "title": "Recent Completed Order",
                "detail": f"Last recorded purchase was {purch_recency:.1f} days ago."
            })
        if tickets is not None and tickets == 0:
            reasons.append({
                "severity": "GOOD",
                "title": "Zero Support Friction",
                "detail": "No support tickets are recorded in the available customer features."
            })

    return reasons


def render() -> None:
    """Render the Customer Risk page with Customer 360 View."""

    if "config" not in st.session_state:
        st.session_state.config = load_config()

    if "state_manager" not in st.session_state:
        st.session_state.state_manager = CustomerStateManager(st.session_state.config)

    if "stream_manager" not in st.session_state:
        st.session_state.stream_manager = StreamManager(
            st.session_state.config, st.session_state.state_manager
        )

    sm: StreamManager = st.session_state.stream_manager
    state: CustomerStateManager = st.session_state.state_manager

    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Customer Risk</h1>
        <p class="page-subtitle">Inspect a customer’s current score, configured gate decision, and recorded behavior.</p>
    </div>
    """, unsafe_allow_html=True)

    cids = sorted(list(state.customer_history.keys()))
    if not cids:
        st.info("No customer records are available.")
        return

    preferred_customer = st.session_state.get("dashboard_inspect_customer")
    default_index = cids.index(preferred_customer) if preferred_customer in cids else 0
    if st.session_state.get("dashboard_customer_risk_selector") not in cids:
        st.session_state.dashboard_customer_risk_selector = cids[default_index]

    selected_cid = st.selectbox(
        "Customer",
        options=cids,
        index=default_index,
        key="dashboard_customer_risk_selector",
        help="Select a customer to inspect the available risk and behavioral data."
    )

    if selected_cid not in sm.latest_predictions:
        feats = state.get_features(selected_cid)
        pred_res = sm.predictor.predict(selected_cid, feats)
        sm.latest_predictions[selected_cid] = pred_res

        hist = sm.prediction_history.setdefault(selected_cid, deque(maxlen=sm.risk_history_size))
        hist.append({
            "timestamp": pred_res["timestamp"],
            "churn_probability": pred_res["churn_probability"],
            "threshold": pred_res["threshold"],
            "risk_level": pred_res["risk_level"],
            "intervention_status": pred_res["intervention_status"]
        })

    pred_res = sm.latest_predictions[selected_cid]
    prob = pred_res["churn_probability"]
    threshold = pred_res["threshold"]
    risk_level = pred_res["risk_level"]
    status = pred_res["intervention_status"]

    hist_list = list(sm.prediction_history.get(selected_cid, []))
    st.markdown("### Current assessment")
    risk_cols = st.columns(4)
    risk_cols[0].metric("Churn probability", f"{prob:.2%}")
    risk_cols[1].metric("Risk classification", risk_level)
    risk_cols[2].metric("Decision threshold", f"{threshold:.2%}")
    risk_cols[3].metric(
        "Decision gate",
        "Intervention required" if status == "INTERVENTION_REQUIRED" else "Normal",
    )
    if local_demo_enabled(st.session_state.config):
        classification = demo_classification(selected_cid)
        if classification:
            st.info(
                f"Simulated demo assignment: {classification.replace('_', ' ').title()}. "
                "This assignment is separate from the model probability and Decision Gate."
            )

    st.markdown("### Risk history")
    if len(hist_list) > 1:
        df_hist = pd.DataFrame(hist_list)
        fig_traj = go.Figure()
        fig_traj.add_trace(go.Scatter(
            x=df_hist["timestamp"],
            y=df_hist["churn_probability"],
            mode="lines+markers",
            name="Churn probability",
            line={"color": "#4263A5", "width": 2},
            marker={"size": 5},
        ))
        fig_traj.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text=f"Configured threshold ({threshold:.2%})",
        )
        fig_traj.update_layout(
            template="plotly_white",
            height=260,
            margin={"l": 20, "r": 20, "t": 30, "b": 30},
            yaxis={"range": [0, 1], "tickformat": ".0%", "title": "Probability"},
            xaxis_title="Recorded prediction time",
            showlegend=False,
        )
        st.plotly_chart(fig_traj, use_container_width=True)
    else:
        st.caption("No multi-point prediction history is available for this customer.")

    st.markdown("### Behavioral signals")
    st.caption("Rule-based descriptions of recorded features; these are not model-attribution values.")
    df_f = state.get_features(selected_cid)
    explanations = explain_customer_risk(df_f, selected_cid, prob, threshold)
    if explanations:
        st.dataframe(
            pd.DataFrame(explanations)[["title", "severity", "detail"]].rename(
                columns={"title": "Signal", "severity": "Level", "detail": "Recorded feature"}
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No configured behavioral explanation rule matched the available features.")

    feature_labels = {
        "frequency": "Total interactions",
        "monetary_total": "Lifetime purchase value",
        "recency_days": "Days since last event",
        "days_since_last_purchase": "Days since last purchase",
        "purchase_count": "Completed purchases",
        "avg_order_value": "Average order value",
        "coupon_usage_ratio": "Historical coupon usage ratio",
        "cart_abandonment_rate": "Cart abandonment rate",
        "support_ticket_count": "Support tickets",
        "events_7d": "Events in past 7 days",
        "events_30d": "Events in past 30 days",
        "session_frequency": "Session frequency",
        "wishlist_click_count": "Wishlist interactions",
        "device_switch_count": "Device switches",
    }
    feature_rows = []
    feature_values = df_f.loc[selected_cid]
    for feature, label in feature_labels.items():
        if feature not in feature_values or pd.isna(feature_values[feature]):
            value = "Unavailable"
        else:
            value = f"{feature_values[feature]:,.2f}"
            if feature in {"coupon_usage_ratio", "cart_abandonment_rate"}:
                value = f"{feature_values[feature] * 100:.1f}%"
        feature_rows.append({"Feature": label, "Value": value})
    st.markdown("### Recorded customer features")
    st.dataframe(pd.DataFrame(feature_rows), use_container_width=True, hide_index=True)

    cust_events = state.customer_history.get(selected_cid, [])
    if cust_events:
        st.markdown("### Recent customer behavior")
        df_evs = pd.DataFrame(cust_events).tail(5).copy()
        available_columns = [
            column for column in
            ["event_timestamp", "event_type", "product_id", "session_id", "order_value"]
            if column in df_evs
        ]
        df_evs = df_evs[available_columns]
        if "event_timestamp" in df_evs:
            df_evs["event_timestamp"] = pd.to_datetime(
                df_evs["event_timestamp"]
            ).dt.strftime("%Y-%m-%d %H:%M:%S")
        if "order_value" in df_evs:
            df_evs["order_value"] = df_evs["order_value"].map(
                lambda value: f"${value:.2f}" if pd.notna(value) else "Unavailable"
            )
        df_evs.columns = [
            {
                "event_timestamp": "Timestamp",
                "event_type": "Event",
                "product_id": "Product ID",
                "session_id": "Session ID",
                "order_value": "Order value",
            }.get(column, column)
            for column in df_evs.columns
        ]
        st.dataframe(df_evs.iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("No recorded behavioral events are available for this customer.")
