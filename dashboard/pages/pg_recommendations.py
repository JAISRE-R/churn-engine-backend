"""
churn-engine/dashboard/pages/pg_recommendations.py
====================================================
Recommendations Page — Standard baseline recommendations before interventions.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from src.utils import load_config
from src.event_stream import StreamManager, CustomerStateManager


def render() -> None:
    """Render the Recommendations page."""

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
        <h1 class="page-title">Baseline Recommendations</h1>
        <p class="page-subtitle">
            Standard behavioral affinity and collaborative filtering recommendations prior to Decision Gate overrides.
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
        "Choose a Customer ID to view standard recommendations:",
        options=cids,
        index=0,
        help="Select a customer to inspect standard collaborative filtering and behavioral affinity recommendations."
    )

    st.markdown("---")

    # Generate baseline recommendations on-demand
    cust_events = state.customer_history.get(selected_cid, [])
    baseline_res = sm.recommender.recommend(selected_cid, cust_events)

    col_profile, col_strategy = st.columns(2)
    with col_profile:
        st.markdown(f"#### 👤 Customer Profile: `{selected_cid}`")
        st.write(f"**Total Events logged:** {len(cust_events)}")
    with col_strategy:
        st.markdown("#### 🎯 Active Strategy")
        st.markdown("""
        <div style="background-color: #ddf4ff; border: 1px solid #54aeff; border-radius: 6px; padding: 0.6rem 1rem; text-align: center;">
            <span style="font-weight: 700; color: #0969da; font-family: monospace;">STRATEGY: BASELINE (NON-INTERVENTION)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Recommendations Output Table
    st.markdown("### 🛍️ Baseline Recommendations List")
    recs_list = baseline_res.get("recommendations", [])
    if recs_list:
        df_recs = pd.DataFrame(recs_list)[["product_id", "category", "score", "reason"]]
        df_recs.columns = ["Product ID", "Category", "Score", "Recommendation Reason"]
        st.dataframe(df_recs, use_container_width=True, hide_index=True)
        
        # Product Card Grid
        st.markdown("#### 📦 Product Details View")
        cols = st.columns(len(recs_list))
        for idx, item in enumerate(recs_list):
            with cols[idx]:
                st.markdown(f"""
                <div style="background-color: #ffffff; padding: 1rem; border-radius: 8px; border: 1px solid #d0d7de; box-shadow: 0 1px 3px rgba(0,0,0,0.04); height: 180px;">
                    <div style="font-size: 0.75rem; color: #656d76; text-transform: uppercase; font-weight: 600;">{item['category']}</div>
                    <div style="font-size: 1.1rem; font-weight: bold; color: #0969da; margin: 0.25rem 0;">{item['product_id']}</div>
                    <div style="font-size: 0.8rem; color: #1f2328; margin-bottom: 0.5rem; line-height: 1.3;">{item['reason']}</div>
                    <div style="font-family: monospace; font-size: 0.95rem; color: #1a7f37; font-weight: bold;">Score: {item['score']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.caption("No recommendations available for this customer profile.")
