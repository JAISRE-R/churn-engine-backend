"""
churn-engine/dashboard/pages/pg_churn_score.py
================================================
Customer Risk Page — Customer 360 View, Risk Diagnostics & Explainability.
"""

from __future__ import annotations

import random
from collections import deque
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.event_stream import PERSONAS, CustomerStateManager, StreamManager
from src.utils import load_config


def get_customer_persona(cid: str) -> str:
    """Helper to deterministically reconstruct the customer persona from ID."""
    try:
        idx = int(cid.split("-")[1]) - 1000
        p_names = list(PERSONAS.keys())
        p_weights = [PERSONAS[p]["weight"] for p in p_names]
        temp_rng = random.Random(42)
        p_list = temp_rng.choices(p_names, weights=p_weights, k=max(1000, idx + 1))
        if 0 <= idx < len(p_list):
            return p_list[idx].replace("_", " ").title()
    except Exception:
        pass
    return "General Customer"


def explain_customer_risk(df_f: pd.DataFrame, cid: str, prob: float, threshold: float) -> list[dict[str, str]]:
    """Generate explainable behavioral drivers for the selected customer.

    Evaluates the customer's engineered features against behavioral thresholds
    to identify the specific reasons why churn risk is elevated or suppressed.
    """
    reasons = []
    row = df_f.loc[cid]

    recency = float(row.get("recency_days", 0.0))
    purch_recency = float(row.get("days_since_last_purchase", 0.0))
    abandon = float(row.get("cart_abandonment_rate", 0.0))
    tickets = int(row.get("support_ticket_count", 0))
    events_7d = int(row.get("events_7d", 0))
    coupon_ratio = float(row.get("coupon_usage_ratio", 0.0))

    if prob >= threshold:
        # Customer is at risk — identify primary risk catalysts
        if recency > 30.0:
            reasons.append({
                "severity": "HIGH",
                "title": "Extended Browsing Inactivity",
                "detail": f"Customer has been inactive for {recency:.1f} days without session activity."
            })
        if purch_recency > 45.0:
            reasons.append({
                "severity": "HIGH",
                "title": "Purchase Dormancy",
                "detail": f"Last completed transaction occurred {purch_recency:.1f} days ago."
            })
        if abandon >= 0.50:
            reasons.append({
                "severity": "MEDIUM",
                "title": "Severe Cart Abandonment",
                "detail": f"Cart abandonment rate is {abandon*100:.1f}%, indicating pricing hesitation or checkout friction."
            })
        if tickets >= 2:
            reasons.append({
                "severity": "MEDIUM",
                "title": "Customer Support Friction",
                "detail": f"Logged {tickets} support interactions, suggesting product or order dissatisfaction."
            })
        if events_7d == 0:
            reasons.append({
                "severity": "MEDIUM",
                "title": "Velocity Drop-Off (Past 7 Days)",
                "detail": "Zero behavioral events recorded during the rolling 7-day observation window."
            })
        if not reasons:
            reasons.append({
                "severity": "LOW",
                "title": "Moderate Engagement Decline",
                "detail": "General interaction frequency is below baseline expectation for this customer segment."
            })
    else:
        # Customer is retained / low risk — identify protective factors
        if recency <= 14.0:
            reasons.append({
                "severity": "GOOD",
                "title": "Active Engagement",
                "detail": f"Recent session activity within the last {recency:.1f} days."
            })
        if purch_recency <= 30.0:
            reasons.append({
                "severity": "GOOD",
                "title": "Recent Completed Order",
                "detail": f"Purchased within the past {purch_recency:.1f} days with positive transaction history."
            })
        if tickets == 0:
            reasons.append({
                "severity": "GOOD",
                "title": "Zero Support Friction",
                "detail": "No logged customer service tickets or unresolved inquiries."
            })
        if not reasons:
            reasons.append({
                "severity": "GOOD",
                "title": "Stable Baseline Metrics",
                "detail": "Browsing patterns and transaction cadence align with loyal customer profiles."
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
        <h1 class="page-title">Customer 360 Risk Registry</h1>
        <p class="page-subtitle">
            Comprehensive customer health profiling, real-time churn inference, behavioral analytics, and intervention diagnostics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 1. "Which customers are at risk right now?" Section
    st.markdown("### 🚨 Active Risk Alert Desk")

    at_risk_profiles = []
    for cid, pred in sm.latest_predictions.items():
        if pred["churn_probability"] >= pred["threshold"]:
            at_risk_profiles.append({
                "Customer ID": cid,
                "Persona": get_customer_persona(cid),
                "Churn Prob": pred["churn_probability"],
                "Risk Level": pred["risk_level"],
                "Threshold": pred["threshold"],
                "Intervention Status": "INTERVENTION REQUIRED"
            })

    if at_risk_profiles:
        df_risk = pd.DataFrame(at_risk_profiles).sort_values("Churn Prob", ascending=False)
        df_risk["Churn Prob"] = df_risk["Churn Prob"].map(lambda x: f"{x:.2%}")
        df_risk["Threshold"] = df_risk["Threshold"].map(lambda x: f"{x:.2%}")
        st.dataframe(df_risk, use_container_width=True, hide_index=True)
    else:
        st.success("🟢 No monitored customers currently violate the intervention threshold.")

    st.markdown("---")

    # 2. Select & Inspect Customer Profile (Customer 360 View)
    st.markdown("### 🔍 Customer 360 Inspector")
    cids = sorted(list(state.customer_history.keys()))
    if not cids:
        st.error("No customer records found. Please generate or load the dataset first.")
        return

    selected_cid = st.selectbox(
        "Choose a Customer ID to inspect 360 profile:",
        options=cids,
        index=0,
        help="Search or select a customer ID to load real-time predictions, trajectory, feature signals, and override status."
    )

    # Score on-demand if the selected customer has not been scored yet in this session
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

    # Retrieve current prediction values
    pred_res = sm.latest_predictions[selected_cid]
    prob = pred_res["churn_probability"]
    threshold = pred_res["threshold"]
    risk_level = pred_res["risk_level"]
    status = pred_res["intervention_status"]
    persona = get_customer_persona(selected_cid)

    hist_list = list(sm.prediction_history[selected_cid])
    if len(hist_list) >= 2:
        prev_prob = hist_list[-2]["churn_probability"]
    else:
        prev_prob = prob
    prob_delta = prob - prev_prob

    # Header Card: Customer 360 Summary
    col_hdr1, col_hdr2 = st.columns([1, 1])

    with col_hdr1:
        st.markdown(f"#### 👤 Account: `{selected_cid}` &nbsp;|&nbsp; Persona: **{persona}**")

        color_map = {"LOW": "#1a7f37", "MEDIUM": "#9a6700", "HIGH": "#d29922", "CRITICAL": "#cf222e"}
        bg_map = {"LOW": "#f6fbf7", "MEDIUM": "#fffdf0", "HIGH": "#fff8ea", "CRITICAL": "#fff5f5"}
        badge_color = color_map.get(risk_level, "#656d76")
        card_bg = bg_map.get(risk_level, "#ffffff")
        pct = int(prob * 100)

        st.markdown(f"""
        <div style="
            background-color: {card_bg};
            border: 1px solid {badge_color};
            border-left: 6px solid {badge_color};
            border-radius: 8px;
            padding: 1.25rem;
            margin: 0.5rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        ">
            <div style="font-size: 0.72rem; color: #656d76; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">
                ESTIMATED CHURN PROBABILITY
            </div>
            <div style="font-size: 2.8rem; font-weight: 800; color: {badge_color}; line-height: 1.1; margin: 0.2rem 0;">
                {pct}%
            </div>
            <div style="display: flex; gap: 1rem; font-size: 0.82rem; color: #656d76; margin-top: 0.35rem;">
                <span>Tier: <strong style="color:{badge_color};">{risk_level}</strong></span>
                <span>Threshold: <strong>{int(threshold*100)}%</strong></span>
            </div>
            <div style="font-size: 0.85rem; font-weight: 700; color: {badge_color}; margin-top: 0.5rem;">
                {"🚨 DECISION: INTERVENTION REQUIRED (RETENTION OVERRIDE)" if status == "INTERVENTION_REQUIRED" else "🟢 DECISION: NORMAL OPERATION (BASELINE RECOMMENDATIONS)"}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.metric(
            label="Churn Probability Change",
            value=f"{prob:.2%}",
            delta=f"{prob_delta:+.2%}" if abs(prob_delta) > 0.0001 else "No shift",
            delta_color="inverse"
        )

    with col_hdr2:
        st.markdown("#### 📈 Churn Risk Trajectory")
        if len(hist_list) > 1:
            df_hist = pd.DataFrame(hist_list)
            fig_traj = go.Figure()
            fig_traj.add_trace(go.Scatter(
                x=list(range(len(df_hist))),
                y=df_hist["churn_probability"],
                mode="lines+markers",
                name="Churn Risk",
                line=dict(color=badge_color, width=2.5),
                marker=dict(size=6),
            ))
            fig_traj.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="#cf222e",
                annotation_text=f"Threshold ({threshold:.2f})",
                annotation_position="top right"
            )
            fig_traj.update_layout(
                template="plotly_white",
                height=220,
                margin=dict(l=10, r=10, t=20, b=20),
                yaxis=dict(range=[0, 1.0], tickformat=".0%"),
                xaxis_title="Event Progression",
                yaxis_title="Probability",
            )
            st.plotly_chart(fig_traj, use_container_width=True)
        else:
            st.info("Single snapshot recorded. Real-time background simulation updates will record trajectory shifts.")

        # Latest Recommendation Status Card
        rec_data = sm.latest_recommendations.get(selected_cid)
        if not rec_data:
            cust_events = state.customer_history.get(selected_cid, [])
            base_recs = sm.recommender.recommend(selected_cid, cust_events)
            dec_gate_res = sm.decision_gate.decide(selected_cid, prob, prev_prob, risk_level)
            rec_data = sm.override_engine.process(base_recs, dec_gate_res, cust_events)
            sm.latest_recommendations[selected_cid] = rec_data

        strat = rec_data.get("strategy", "BASELINE")
        items = rec_data.get("recommendations", [])
        st.markdown(f"**Current Recommendation Strategy:** `{strat}`")
        if strat == "RETENTION_OVERRIDE":
            disc = rec_data.get("metadata", {}).get("discount_pct", 15)
            st.caption(f"🎁 Active Incentive: **{disc}% Off Coupon + Targeted Product Bundle**")
        else:
            st.caption("Standard catalog recommendations based on customer browsing history.")

    st.markdown("---")

    # 3. "Why is this customer at risk?" Section
    st.markdown("### 💡 Why is this customer at risk? (Diagnostic Explanations)")
    df_f = state.get_features(selected_cid)
    explanations = explain_customer_risk(df_f, selected_cid, prob, threshold)

    cols_exp = st.columns(len(explanations) if explanations else 1)
    for idx, exp in enumerate(explanations):
        col_idx = cols_exp[idx % len(cols_exp)]
        sev = exp["severity"]
        if sev == "HIGH":
            border_c = "#cf222e"
            bg_c = "#fff5f5"
            icon = "🔴"
        elif sev == "MEDIUM":
            border_c = "#d29922"
            bg_c = "#fff8ea"
            icon = "🟡"
        else:
            border_c = "#1a7f37"
            bg_c = "#f6fbf7"
            icon = "🟢"

        with col_idx:
            st.markdown(f"""
            <div style="
                background-color: {bg_c};
                border: 1px solid {border_c};
                border-radius: 8px;
                padding: 0.85rem 1rem;
                min-height: 110px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            ">
                <div style="font-size: 0.85rem; font-weight: 700; color: #1f2328; margin-bottom: 0.25rem;">
                    {icon} {exp['title']}
                </div>
                <div style="font-size: 0.8rem; color: #656d76; line-height: 1.35;">
                    {exp['detail']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # 4. Behavioral Feature Analytics Table
    st.markdown("### 📊 Customer 360 Behavioral Features")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("**🛒 RFM & Transaction Cadence**")
        rfm_data = {
            "Total Interactions": f"{df_f.loc[selected_cid, 'frequency']:.0f} events",
            "Monetary Value (Lifetime)": f"${df_f.loc[selected_cid, 'monetary_total']:,.2f}",
            "Recency (Days Since Last Event)": f"{df_f.loc[selected_cid, 'recency_days']:.1f} days",
            "Days Since Last Completed Purchase": f"{df_f.loc[selected_cid, 'days_since_last_purchase']:.1f} days",
            "Completed Purchases Count": f"{df_f.loc[selected_cid, 'purchase_count']:.0f} orders",
            "Average Order Value (AOV)": f"${df_f.loc[selected_cid, 'avg_order_value']:.2f}",
            "Coupon Usage Ratio": f"{df_f.loc[selected_cid, 'coupon_usage_ratio']*100:.1f}%",
        }
        st.table(pd.DataFrame(list(rfm_data.items()), columns=["Signal", "Value"]))

    with col_f2:
        st.markdown("**🌐 Engagement Dynamics & Friction Points**")
        session_data = {
            "Cart Abandonment Rate": f"{df_f.loc[selected_cid, 'cart_abandonment_rate']*100.0:.1f}%",
            "Customer Support Tickets": f"{df_f.loc[selected_cid, 'support_ticket_count']:.0f} tickets",
            "Rolling 7-Day Event Count": f"{df_f.loc[selected_cid, 'events_7d']:.0f} events",
            "Rolling 30-Day Event Count": f"{df_f.loc[selected_cid, 'events_30d']:.0f} events",
            "Session Frequency": f"{df_f.loc[selected_cid, 'session_frequency']:.2f} sessions/active day",
            "Wishlist Interactions": f"{df_f.loc[selected_cid, 'wishlist_click_count']:.0f} clicks",
            "Distinct Device Switches": f"{df_f.loc[selected_cid, 'device_switch_count']:.0f} devices",
        }
        st.table(pd.DataFrame(list(session_data.items()), columns=["Signal", "Value"]))

    # Customer activity log tail
    cust_events = state.customer_history.get(selected_cid, [])
    if cust_events:
        st.markdown("#### 📜 Recent Behavioral Log")
        df_evs = pd.DataFrame(cust_events)[["event_timestamp", "event_type", "session_id", "order_value"]].tail(5)
        df_evs["event_timestamp"] = pd.to_datetime(df_evs["event_timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        df_evs["order_value"] = df_evs["order_value"].fillna(0.0).map(lambda x: f"${x:.2f}" if x > 0 else "—")
        df_evs.columns = ["Timestamp", "Event Type", "Session ID", "Order Value"]
        st.dataframe(df_evs.iloc[::-1], use_container_width=True, hide_index=True)
