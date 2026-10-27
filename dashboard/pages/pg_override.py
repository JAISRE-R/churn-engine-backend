"""
churn-engine/dashboard/pages/pg_override.py
=============================================
Override Engine Page — Retention Override Interventions.
"""

from __future__ import annotations

import time
import pandas as pd
import streamlit as st

from src.utils import load_config
from src.event_stream import StreamManager, CustomerStateManager


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
        <h1 class="page-title">Retention Interventions</h1>
        <p class="page-subtitle">Compare baseline recommendations with the Decision Gate outcome.</p>
    </div>
    """, unsafe_allow_html=True)

    cids = sorted(list(state.customer_history.keys()))
    if not cids:
        st.error("No customer records found. Please generate the dataset first.")
        return

    # Customer Selector
    selected_cid = st.selectbox(
        "Customer",
        options=cids,
        index=0
    )

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

    base_recs = baseline_result.get("recommendations", [])
    final_recs = final_result.get("recommendations", [])
    baseline_by_id = {item["product_id"]: item for item in base_recs}
    final_by_id = {item["product_id"]: item for item in final_recs}
    all_product_ids = list(dict.fromkeys([*baseline_by_id, *final_by_id]))

    gate_status = dec_res.get("decision", "Unavailable")
    probability = dec_res.get("churn_probability")
    threshold = dec_res.get("threshold")
    status_columns = st.columns(4)
    status_columns[0].metric(
        "Customer", selected_cid, help="Selected customer ID from local event history."
    )
    status_columns[1].metric(
        "Churn probability",
        f"{probability:.2%}" if probability is not None else "Unavailable",
    )
    status_columns[2].metric(
        "Configured threshold",
        f"{threshold:.2%}" if threshold is not None else "Unavailable",
    )
    status_columns[3].metric(
        "Decision Gate",
        str(gate_status).replace("_", " ").title(),
        help=f"Recorded risk classification: {dec_res.get('risk_level', 'Unavailable')}",
    )

    if final_result.get("strategy") == "RETENTION_OVERRIDE":
        st.caption(
            f"Intervention reason: {final_result.get('metadata', {}).get('override_reason', 'Unavailable')}"
        )
    else:
        st.caption("No intervention was generated; baseline recommendations remain in effect.")

    comparison_rows = []
    for product_id in all_product_ids:
        baseline_item = baseline_by_id.get(product_id)
        resolved_item = final_by_id.get(product_id)
        discount = resolved_item.get("discount") if resolved_item else None
        if isinstance(discount, (int, float)) and discount > 0:
            offer_status = f"{discount}% offer for this product"
        else:
            offer_status = "No product-level offer recorded"
        comparison_rows.append({
            "Product ID": product_id,
            "Baseline recommendation": (
                baseline_item.get("reason", "Recommended")
                if baseline_item else "—"
            ),
            "Resolved recommendation": (
                resolved_item.get("reason", "Recommended")
                if resolved_item else "—"
            ),
            "Product-specific offer": offer_status,
        })

    st.markdown("### Baseline and resolved recommendations")
    if comparison_rows:
        st.dataframe(
            pd.DataFrame(comparison_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No recommendations are available for this customer.")

    if final_result.get("strategy") == "RETENTION_OVERRIDE":
        st.caption(
            "Offers are shown only when the Override Engine includes a product-level discount "
            "on that recommendation. No customer outcome is recorded on this page."
        )

    # Live Polling
    if is_running:
        time.sleep(float(st.session_state.config["event_stream"].get("stream_refresh_interval", 1.5)))
        st.rerun()
