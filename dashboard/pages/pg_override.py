"""
churn-engine/dashboard/pages/pg_override.py
=============================================
Override Engine Page — Retention Override Interventions.
"""

from __future__ import annotations

import time
import pandas as pd
import streamlit as st
from datetime import datetime

from src.utils import load_config
from src.event_stream import StreamManager, CustomerStateManager
from src.recommender import get_product_category


def render() -> None:
    """Render the Interventions page."""

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
    stats = sm.get_stats()
    is_running = stats["status"] == "running"

    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Intervention Strategy Desk</h1>
        <p class="page-subtitle">
            Side-by-side comparison of baseline recommendations and personalized retention overrides.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cids = sorted(list(state.customer_history.keys()))
    if not cids:
        st.error("No customer records found. Please generate the dataset first.")
        return

    # Customer Selector
    st.markdown("### 🔍 Select Customer Profile")
    selected_cid = st.selectbox(
        "Choose a Customer ID to inspect before/after recommendations:",
        options=cids,
        index=0
    )

    st.markdown("---")

    # Generate or fetch on-demand data
    from collections import deque
    if selected_cid not in sm.latest_decisions:
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

        dec_res = sm.decision_gate.decide(
            customer_id=selected_cid,
            churn_probability=pred_res["churn_probability"],
            previous_probability=None,
            risk_level=pred_res["risk_level"]
        )
        sm.latest_decisions[selected_cid] = dec_res

    cust_events = state.customer_history.get(selected_cid, [])
    baseline_result = sm.recommender.recommend(selected_cid, cust_events)
    dec_res = sm.latest_decisions[selected_cid]
    final_result = sm.override_engine.process(baseline_result, dec_res, cust_events)

    # Risk Profile Header
    col_risk1, col_risk2 = st.columns(2)
    with col_risk1:
        st.markdown(f"#### 👤 Customer: `{selected_cid}`")
        st.write(f"**Churn Risk Probability:** `{dec_res['churn_probability']:.2%}`")
        st.write(f"**Risk Tier Level:** `{dec_res['risk_level']}`")
    with col_risk2:
        st.markdown("#### 🚦 Gate Status")
        if dec_res["decision"] == "INTERVENTION_REQUIRED":
            st.markdown("""
            <div style="background-color: #ffebe9; border: 1px solid #cf222e; border-radius: 6px; padding: 0.5rem 1rem; font-weight: 700; color: #cf222e; text-align: center;">
                🚨 INTERVENTION REQUIRED — OVERRIDE ACTIVE
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #dafbe1; border: 1px solid #1a7f37; border-radius: 6px; padding: 0.5rem 1rem; font-weight: 700; color: #1a7f37; text-align: center;">
                🟢 NORMAL SYSTEM STATUS — BASELINE PASS-THROUGH
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # BEFORE VS AFTER COMPARISON
    st.markdown("### 📋 Before vs. After Intervention Comparison")
    col_before, col_after = st.columns(2)

    with col_before:
        st.markdown("<h4 style='color: #0969da;'>🟢 BEFORE INTERVENTION (Baseline)</h4>", unsafe_allow_html=True)
        st.caption("Strategy: BASELINE")
        
        base_recs = baseline_result.get("recommendations", [])
        base_data = []
        for idx, r in enumerate(base_recs):
            base_data.append({
                "Rank": idx + 1,
                "Product ID": r["product_id"],
                "Category": r["category"],
                "Score": f"{r['score']:.2f}",
                "Reason": r["reason"]
            })
        st.dataframe(pd.DataFrame(base_data), use_container_width=True, hide_index=True)

    with col_after:
        st.markdown("<h4 style='color: #8250df;'>🔮 AFTER INTERVENTION (Resolved)</h4>", unsafe_allow_html=True)
        st.caption(f"Strategy: {final_result['strategy']}")
        
        final_recs = final_result.get("recommendations", [])
        final_data = []
        for idx, r in enumerate(final_recs):
            final_data.append({
                "Rank": idx + 1,
                "Product ID": r["product_id"],
                "Category": r["category"],
                "Score": f"{r['score']:.2f}",
                "Reason": r["reason"]
            })
        st.dataframe(pd.DataFrame(final_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # OVERRIDE DIAGNOSTIC DETAILS
    st.markdown("### 🔍 Intervention Strategy Details")
    
    if final_result["strategy"] == "RETENTION_OVERRIDE":
        base_pids = {r["product_id"] for r in base_recs}
        final_pids = {r["product_id"] for r in final_recs}
        
        added = final_pids - base_pids
        removed = base_pids - final_pids
        replaced_count = len(removed)
        
        col_an1, col_an2 = st.columns(2)
        with col_an1:
            st.write(f"**Intervention Trigger Reason:** *{final_result['metadata']['override_reason']}*")
            st.write(f"**Applied Discount Value:** `{final_result['metadata']['discount_pct']}%`")
            st.write(f"**Number of Products Replaced:** `{replaced_count}` of `{len(base_recs)}` items")
            st.info("💡 **Intervention status:** Intervention offer generated and served to user session.")
        with col_an2:
            st.markdown("**➕ Retention Incentives Injected:**")
            for item in final_recs:
                if item["product_id"] in added:
                    st.write(f"- `{item['product_id']}` ({item['category']}): *{item['reason']}*")
    else:
        st.info("No override was triggered. Recommender is running in baseline pass-through mode because the customer's churn risk is below threshold.")

    # Live Polling
    if is_running:
        time.sleep(float(st.session_state.config["event_stream"].get("stream_refresh_interval", 1.5)))
        st.rerun()
