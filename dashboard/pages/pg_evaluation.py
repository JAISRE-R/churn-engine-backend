"""
churn-engine/dashboard/pages/pg_evaluation.py
==============================================
Evaluation Page — Multi-Model Benchmark, Threshold Tuning & Visualisations.
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


def _style_chart(fig: go.Figure, height: int) -> go.Figure:
    """Apply the dashboard's light, readable chart styling."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        font={"family": "Inter, sans-serif", "size": 12, "color": "#424a53"},
        title={"x": 0, "xanchor": "left", "font": {"size": 16, "color": "#1f2328"}},
        margin={"l": 48, "r": 24, "t": 64, "b": 48},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#d0d7de", ticks="outside")
    fig.update_yaxes(showgrid=True, gridcolor="#eaeef2", zeroline=False, linecolor="#d0d7de")
    return fig


def render() -> None:
    """Render the Evaluation dashboard page."""

    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">🧪 Model Evaluation</h1>
        <p class="page-subtitle">
            Multi-model benchmark · threshold optimisation · performance diagnostics
        </p>
    </div>
    """, unsafe_allow_html=True)

    config = load_config()
    root = get_project_root()

    model_path = root / config["paths"]["models"] / "churn_model.joblib"
    meta_path = root / config["paths"]["models"] / "model_metadata.json"
    policy_path = root / config["paths"]["models"] / "decision_policy.json"
    report_path = root / config["paths"]["logs"] / "evaluation_report.json"
    exp_path = root / config["paths"]["reports"] / "experiment_history.csv"
    sample_path = root / config["paths"]["reports"] / "sample_predictions.csv"
    features_path = root / config["paths"]["data_features"] / "customer_features.parquet"

    model_ready = model_path.exists() and meta_path.exists()

    if not model_ready:
        st.warning(
            "⚠️ **No trained model artefact found in `models/`.** "
            "Click below to benchmark models (Logistic Regression, Random Forest, tuned XGBoost), "
            "select the winner, optimize threshold, and export metadata."
        )
        if st.button("🚀 Run Phase 3 Model Training & Selection", type="primary"):
            if not features_path.exists():
                st.error("Feature matrix `customer_features.parquet` missing. Run Phase 2 first!")
                return
            with st.spinner("Benchmarking models & optimizing decision threshold..."):
                from src.train_model import TrainingPipeline
                features_df = pd.read_parquet(features_path)
                X = features_df.drop(columns=["churn"])
                y = features_df["churn"]
                results = TrainingPipeline(config).run(X, y)
                st.success(f"✅ Selected Winning Model: **{results['winning_model']}** with Optimal Cutoff **{results['optimal_threshold']:.2f}**!")
                st.rerun()
        return

    # Load metadata and reports
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    policy = {}
    if policy_path.exists():
        with open(policy_path, "r", encoding="utf-8") as f:
            policy = json.load(f)

    report = {}
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

    winning_model = meta.get("model_name", meta.get("model_type", "XGBClassifier"))

    # --- WHY THIS MODEL? ---
    st.markdown("### 🏆 Selected Model")
    col_win1, col_win2 = st.columns([1, 2])
    with col_win1:
        st.markdown(f"""
        <div style="background:#ffffff; border:1px solid #d0d7de; border-left:4px solid #1a7f37; border-radius:8px; padding:1.25rem 1.5rem; box-shadow:0 1px 2px rgba(31,35,40,0.04);">
            <div style="font-size:0.72rem; font-weight:600; color:#656d76; text-transform:uppercase; letter-spacing:0.05em;">Winning Model</div>
            <div style="font-size:1.45rem; font-weight:700; color:#1f2328; margin:0.4rem 0;">{winning_model}</div>
            <div style="font-size:0.85rem; color:#656d76;">Selected threshold: {meta.get("selected_threshold", 0.5):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_win2:
        st.markdown("#### Why this model?")
        st.markdown(f"""
        **{winning_model}** was selected because it achieved the **strongest validation performance**
        among the three evaluated candidates (Logistic Regression, Random Forest, XGBoost).

        Selection criteria prioritised **ROC-AUC** (area under the receiver operating characteristic curve),
        followed by **F1 Score** as a tiebreaker — balancing precision and recall for a business use case
        where both false positives (wasted discount spend) and false negatives (undetected churners) carry cost.
        """)

    st.markdown("---")

    # Top Metric KPI Cards
    st.markdown("### 📊 Test Set Performance Metrics")
    m = meta.get("test_metrics", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ROC-AUC", f"{m.get('roc_auc', 0.0):.4f}")
    with col2:
        st.metric("F1 Score", f"{m.get('f1', 0.0):.4f}")
    with col3:
        st.metric("MCC", f"{m.get('mcc', 0.0):.4f}")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Precision", f"{m.get('precision', 0.0):.4f}")
    with col5:
        st.metric("Recall", f"{m.get('recall', 0.0):.4f}")
    with col6:
        st.metric("Optimal Cutoff", f"{meta.get('selected_threshold', 0.5):.2f}")

    st.markdown("---")

    # --- Section 1: Multi-Model Benchmark Comparison Table ---
    st.markdown("### 🏎️ Multi-Model Benchmark Comparison")
    st.caption("Candidates evaluated on validation set. Winner selected based on highest ROC-AUC.")

    bench_data = report.get("benchmark_summary", [])
    if bench_data:
        df_b = pd.DataFrame(bench_data)
        st.dataframe(df_b.style.format(precision=4), use_container_width=True, hide_index=True)
    else:
        st.info("Benchmark summary data unavailable.")

    st.markdown("---")

    # --- Section 2: Decision Policy & Threshold Optimization ---
    st.markdown("### 🚦 Decision Policy & Threshold Optimization")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Selected Decision Policy (`decision_policy.json`)")
        st.json(policy)

    with col_right:
        st.markdown("#### Threshold Sensitivity Analysis")
        # Recompute threshold metrics from exported test predictions.
        thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
        # Re-compute or render sensitivity
        if sample_path.exists():
            df_samp = pd.read_csv(sample_path)
            y_t = df_samp["actual_label"]
            y_p = df_samp["predicted_probability"]
            from src.evaluation import evaluate_thresholds
            df_thresh = evaluate_thresholds(y_t, y_p, thresholds)[0]

            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(x=df_thresh["threshold"], y=df_thresh["f1"], name="F1 Score", mode="lines+markers", line={"color": "#0969da", "width": 2.5}))
            fig_t.add_trace(go.Scatter(x=df_thresh["threshold"], y=df_thresh["precision"], name="Precision", mode="lines+markers", line={"color": "#1a7f37", "width": 2}))
            fig_t.add_trace(go.Scatter(x=df_thresh["threshold"], y=df_thresh["recall"], name="Recall", mode="lines+markers", line={"color": "#9a6700", "width": 2}))
            fig_t.add_trace(go.Scatter(x=df_thresh["threshold"], y=df_thresh["fpr"], name="FPR", mode="lines+markers", line={"dash": "dash", "color": "#cf222e", "width": 2}))

            fig_t.add_vline(x=meta.get("selected_threshold", 0.5), line_dash="dot", line_color="#656d76", annotation_text=f"Selected ({meta.get('selected_threshold', 0.5):.2f})")
            _style_chart(fig_t, height=360)
            fig_t.update_layout(title="Metrics vs Decision Threshold", xaxis_title="Decision threshold", yaxis_title="Metric score")
            st.plotly_chart(fig_t, use_container_width=True)

    st.markdown("---")

    # --- Section 3: Diagnostic Visualisations Tabs ---
    st.markdown("### 🔍 Model Diagnostic Visualisations")

    t1, t2, t3, t4, t5 = st.tabs([
        "📊 Probability Distribution",
        "🔢 Confusion Matrix",
        "📋 Feature Importances",
        "📜 Experiment Log History",
        "🎯 Sample Test Predictions"
    ])

    with t1:
        if sample_path.exists():
            df_s = pd.read_csv(sample_path)
            fig_dist = px.histogram(
                df_s, x="predicted_probability", color="actual_label",
                barmode="overlay", nbins=30,
                color_discrete_map={0: "#1a7f37", 1: "#cf222e"},
                title="Predicted Churn Probability Distribution (Churned vs Retained)",
                labels={"actual_label": "Actual outcome", "predicted_probability": "Predicted churn probability"}
            )
            fig_dist.add_vline(x=meta.get("selected_threshold", 0.5), line_dash="dash", line_color="#656d76", annotation_text="Selected cutoff")
            fig_dist.for_each_trace(lambda trace: trace.update(name={"0": "Retained (0)", "1": "Churned (1)"}.get(trace.name, trace.name)))
            _style_chart(fig_dist, height=420)
            fig_dist.update_layout(title="Predicted churn probability distribution")
            fig_dist.update_xaxes(title_text="Predicted churn probability")
            fig_dist.update_yaxes(title_text="Customer count")
            st.plotly_chart(fig_dist, use_container_width=True)

    with t2:
        cm_data = report.get("confusion_matrix", [[0, 0], [0, 0]])
        cm_arr = np.array(cm_data)
        fig_cm = px.imshow(
            cm_arr, text_auto=True,
            x=["Predicted Retained (0)", "Predicted Churned (1)"],
            y=["Actual Retained (0)", "Actual Churned (1)"],
            color_continuous_scale="Blues",
            title="Confusion Matrix Heatmap"
        )
        _style_chart(fig_cm, height=400)
        fig_cm.update_layout(coloraxis_colorbar_title="Customers")
        fig_cm.update_xaxes(title_text="Predicted outcome")
        fig_cm.update_yaxes(title_text="Actual outcome")
        st.plotly_chart(fig_cm, use_container_width=True)

    with t3:
        if features_path.exists() and model_path.exists():
            import joblib
            pipeline = joblib.load(model_path)
            feat_names = pipeline.feature_names
            if hasattr(pipeline.model, "feature_importances_"):
                imps = pipeline.model.feature_importances_
            else:
                imps = np.abs(getattr(pipeline.model, "coef_", [np.zeros(len(feat_names))])[0])

            df_imp = pd.DataFrame({"Feature": feat_names, "Importance": imps}).sort_values("Importance", ascending=False).head(20)
            fig_imp = px.bar(
                df_imp, x="Importance", y="Feature", orientation="h",
                title="Top 20 Feature Importances"
            )
            fig_imp.update_traces(marker_color="#0969da")
            _style_chart(fig_imp, height=480)
            fig_imp.update_layout(showlegend=False)
            fig_imp.update_xaxes(title_text="Importance")
            fig_imp.update_yaxes(title_text="Feature", autorange="reversed")
            st.plotly_chart(fig_imp, use_container_width=True)

    with t4:
        if exp_path.exists():
            df_exp = pd.read_csv(exp_path)
            st.dataframe(df_exp.style.format(precision=4), use_container_width=True, hide_index=True)
        else:
            st.info("No experiment history logged yet.")

    with t5:
        if sample_path.exists():
            df_samp = pd.read_csv(sample_path)
            st.dataframe(df_samp.head(100).style.format(precision=4), use_container_width=True, hide_index=True)
            st.caption(f"Showing sample test predictions ({len(df_samp)} rows exported to `reports/sample_predictions.csv`).")
