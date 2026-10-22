"""
churn-engine/dashboard/pages/pg_dataset.py
===========================================
Dataset Page — Data Overview, Quality Health Report & Feature Explorer.
"""

from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.utils import load_config, get_project_root


def render() -> None:
    """Render the Dataset dashboard page."""

    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🗄️ Dataset &amp; EDA</h1>
        <p class="page-subtitle">
            Persona Event Data, Pre-Preprocessing Validation &amp; Customer Feature Mart
        </p>
    </div>
    """, unsafe_allow_html=True)

    config = load_config()
    root = get_project_root()

    raw_path = root / config["paths"]["data_raw"] / config["data"].get("raw_filename", "raw_events.csv")
    report_path = root / config["paths"]["logs"] / "data_validation_report.json"
    cleaned_path = root / config["paths"]["data_processed"] / "cleaned_events.parquet"
    features_path = root / config["paths"]["data_features"] / "customer_features.parquet"

    # Check if pipeline has been run
    data_ready = raw_path.exists() and features_path.exists()

    if not data_ready:
        st.warning(
            "⚠️ **Data pipeline outputs not found.** "
            "Click the button below to generate synthetic persona data, execute validation, "
            "cleaning, and feature engineering."
        )
        if st.button("🚀 Run Phase 2 Data Pipeline", type="primary"):
            with st.spinner("Generating persona data & executing data pipeline..."):
                from src.generate_data import generate_synthetic_dataset
                from src.validation import DataValidator
                from src.preprocessing import PreprocessingPipeline
                from src.feature_engineering import FeatureEngineeringPipeline

                # 1. Generate
                raw_df = generate_synthetic_dataset(num_customers=500, seed=42)
                raw_dir = root / config["paths"]["data_raw"]
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_df.to_csv(raw_path, index=False)

                # 2. Validate
                DataValidator(config).validate(raw_df)

                # 3. Clean
                clean_df = PreprocessingPipeline(config).run()

                # 4. Feature Engineering
                FeatureEngineeringPipeline(config).build(clean_df)

                st.success("✅ Phase 2 Data Pipeline completed successfully!")
                st.rerun()
        return

    # Load datasets
    df_raw = pd.read_csv(raw_path)
    df_features = pd.read_parquet(features_path)

    val_report = {}
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            val_report = json.load(f)

    # Top KPI Metrics
    st.markdown("### 📈 Pipeline Summary Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Events", f"{len(df_raw):,}")
    with col2:
        st.metric("Unique Customers", f"{len(df_features):,}")
    with col3:
        churn_rate = (df_features["churn"].mean() * 100)
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with col4:
        st.metric("Feature Count", f"{df_features.shape[1] - 1}")
    with col5:
        val_status = val_report.get("status", "UNKNOWN")
        badge = "🟢 PASSED" if val_status == "PASSED" else "🟡 WARNINGS"
        st.metric("Data Health", badge)

    st.markdown("---")

    # --- Section 1: Pre-Preprocessing Validation Report ---
    st.markdown("### 🛡️ Pre-Preprocessing Validation Health Checks")
    if val_report:
        checks_data = []
        for check_name, details in val_report.get("checks", {}).items():
            checks_data.append({
                "Validation Rule": check_name.replace("_", " ").title(),
                "Failed Record Count": details.get("failed_count", 0),
                "Status": "✅ PASS" if details.get("status") == "PASSED" else "❌ FAIL",
            })
        st.dataframe(pd.DataFrame(checks_data), use_container_width=True, hide_index=True)
    else:
        st.info("No validation report found.")

    st.markdown("---")

    # --- Section 2: EDA & Feature Visualisations ---
    st.markdown("### 📊 Behavioural Data & Feature Visualisations")
    tab1, tab2, tab3 = st.columns(3)

    with tab1:
        st.markdown("#### Target Class Balance")
        churn_counts = df_features["churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn Status", "Count"]
        churn_counts["Churn Status"] = churn_counts["Churn Status"].map({0: "Retained (0)", 1: "Churned (1)"})
        fig_churn = px.bar(
            churn_counts, x="Churn Status", y="Count",
            color="Churn Status",
            color_discrete_map={"Retained (0)": "#3fb950", "Churned (1)": "#f85149"},
            title="Customer Churn Label Distribution",
        )
        fig_churn.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig_churn, use_container_width=True)

    with tab2:
        st.markdown("#### Event Types Frequency")
        event_counts = df_raw["event_type"].value_counts().reset_index()
        event_counts.columns = ["Event Type", "Count"]
        fig_events = px.pie(
            event_counts, values="Count", names="Event Type",
            title="Distribution of Behavioural Event Types",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Dark24,
        )
        fig_events.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig_events, use_container_width=True)

    with tab3:
        st.markdown("#### Top Feature Correlations with Churn")
        numeric_features = df_features.select_dtypes(include=[np.number])
        corrs = numeric_features.corr()["churn"].drop("churn").sort_values()
        fig_corr = px.bar(
            x=corrs.values, y=corrs.index, orientation="h",
            labels={"x": "Pearson Correlation", "y": "Feature"},
            title="Feature Correlation to Target (Churn)",
            color=corrs.values,
            color_continuous_scale="RdBu_r",
        )
        fig_corr.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # --- Section 3: Interactive Data Mart Explorer ---
    st.markdown("### 🔍 Customer Feature Mart Explorer")
    st.caption("Inspect customer-level unscaled business domain features derived from behavioural logs.")

    # Search & Filter
    search_id = st.text_input("Filter by Customer ID (e.g. C-1005):", "")
    filtered_df = df_features.copy()
    if search_id.strip():
        filtered_df = filtered_df[filtered_df.index.astype(str).str.contains(search_id.strip(), case=False)]

    st.dataframe(filtered_df.head(100), use_container_width=True)
    st.caption(f"Showing {min(100, len(filtered_df))} of {len(filtered_df)} customer rows.")
