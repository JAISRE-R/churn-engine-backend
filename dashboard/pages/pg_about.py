"""
churn-engine/dashboard/pages/pg_about.py
==========================================
About Page — Project Documentation & Academic Context.
"""

from __future__ import annotations
import pandas as pd
import streamlit as st


def render() -> None:
    """Render the About page."""

    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">ℹ️ About This Project</h1>
        <p class="page-subtitle">
            Live Customer Churn Risk Assessment Engine with Real-Time Recommendation Overrides
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Project Objective ---
    st.markdown("### 🎯 Project Objective")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ❌ Problem")
        st.markdown("""
        Traditional churn prediction is often **batch-oriented** and **disconnected from
        recommendation systems**.

        A model may identify high-risk customers once per day, but by the time that
        information reaches the recommendation engine, the opportunity to intervene
        has already passed.

        The customer has churned.
        """)
    with col2:
        st.markdown("#### ✅ Solution")
        st.markdown("""
        This system **continuously scores customer risk** from real-time behavioural
        events and **dynamically overrides standard recommendations** when
        intervention is required.

        When a customer crosses the configured churn probability threshold,
        the Decision Gate triggers and the Override Engine **immediately** replaces
        generic recommendations with personalised retention incentives.
        """)

    st.markdown("---")

    # --- Architecture ---
    st.markdown("### 🏗️ Architecture")
    st.code("""
  Live Behavioural Events
          │
          ▼
  ┌─────────────────────┐
  │   Feature Engineering│   25 RFM + Session + Purchase features
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │    Churn Model      │   XGBoost (best of LR / RF / XGB)
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │    Decision Gate    │   Threshold from models/decision_policy.json
  └──────────┬──────────┘
             │
     ┌───────┴────────┐
     │                │
  NORMAL        INTERVENTION
     │           REQUIRED
     │                │
     ▼                ▼
  Standard       Override Engine
  Recs           (Retention Offers)
     │                │
     └───────┬────────┘
             ▼
     Final Recommendation
    """, language=None)

    st.markdown("---")

    # --- Technology ---
    st.markdown("### 🛠️ Technology Stack")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Data & ML**")
        st.markdown("- Python 3.12\n- pandas / numpy\n- scikit-learn\n- XGBoost\n- joblib")
    with col2:
        st.markdown("**Dashboard**")
        st.markdown("- Streamlit\n- Plotly\n- matplotlib / seaborn")
    with col3:
        st.markdown("**Infrastructure**")
        st.markdown("- PyYAML\n- pytest\n- threading\n- logging")

    st.markdown("---")

    # --- Source Index ---
    st.markdown("### 📂 Module Index")
    files = [
        ("src/utils.py",              "Config loading, logging, filesystem helpers"),
        ("src/preprocessing.py",      "DataLoader, DataCleaner, PreprocessingPipeline"),
        ("src/feature_engineering.py","RFMBuilder, RollingFeatureBuilder, FeatureEncoder"),
        ("src/train_model.py",        "ModelSelectionEngine, TrainingPipeline"),
        ("src/evaluation.py",         "Metrics, plots, experiment logging"),
        ("src/predictor.py",          "ChurnPredictor — real-time XGBoost inference"),
        ("src/event_stream.py",       "EventSimulator, CustomerStateManager, StreamManager"),
        ("src/decision_gate.py",      "DecisionGate — threshold evaluation & history"),
        ("src/recommender.py",        "BaselineRecommender — behavioural affinity recs"),
        ("src/override_engine.py",    "OverrideEngine — retention intervention middleware"),
    ]
    st.dataframe(pd.DataFrame(files, columns=["Module", "Responsibility"]),
                 use_container_width=True, hide_index=True)
