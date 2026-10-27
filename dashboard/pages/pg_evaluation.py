"""
churn-engine/dashboard/pages/pg_evaluation.py
==============================================
Evaluation Page — Multi-Model Benchmark, Threshold Tuning & Visualisations.
"""

from __future__ import annotations

import json
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
        font={"family": "Inter, sans-serif", "size": 12, "color": "#4B5563"},
        title={"x": 0, "xanchor": "left", "font": {"size": 16, "color": "#1F2937"}},
        margin={"l": 48, "r": 24, "t": 64, "b": 48},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#E1E5EB", ticks="outside")
    fig.update_yaxes(showgrid=True, gridcolor="#EBEEF2", zeroline=False, linecolor="#E1E5EB")
    return fig


def render() -> None:
    """Render the Evaluation dashboard page."""

    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Offline Model Evaluation</h1>
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

    st.markdown("### A. Offline model evaluation")
    st.caption(
        "Historical test-set and benchmark artifacts. These metrics do not describe "
        "current production performance or prove runtime model identity."
    )
    winning_model = meta.get("model_name", meta.get("model_type"))
    st.write(f"Model recorded in offline metadata: **{winning_model or 'Not specified'}**")

    m = meta.get("test_metrics") or {}
    available_metrics = [
        {"Metric": label, "Value": m[key]}
        for key, label in [
            ("roc_auc", "ROC-AUC"),
            ("f1", "F1 score"),
            ("mcc", "Matthews correlation coefficient"),
            ("precision", "Precision"),
            ("recall", "Recall"),
        ]
        if key in m and m[key] is not None
    ]
    if available_metrics:
        metrics_df = pd.DataFrame(available_metrics)
        metrics_df["Value"] = metrics_df["Value"].map(
            lambda value: f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.info("Offline test metrics are unavailable in the model metadata.")

    st.markdown("#### Model comparison")
    bench_data = report.get("benchmark_summary") or []
    if bench_data:
        df_b = pd.DataFrame(bench_data)
        st.dataframe(df_b.style.format(precision=4), use_container_width=True, hide_index=True)
    else:
        st.info("Historical benchmark comparison is unavailable.")

    st.markdown("#### Offline decision-threshold analysis")
    offline_threshold = meta.get("selected_threshold")
    if offline_threshold is None:
        offline_threshold = policy.get("selected_threshold")
    if offline_threshold is not None:
        st.caption(f"Threshold recorded in offline artifacts: **{float(offline_threshold):.2f}**")
    else:
        st.caption("Offline threshold: Unavailable")

    if sample_path.exists():
        df_samp = pd.read_csv(sample_path)
        required_columns = {"actual_label", "predicted_probability"}
        if required_columns.issubset(df_samp.columns) and not df_samp.empty:
            thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
            from src.evaluation import evaluate_thresholds
            df_thresh = evaluate_thresholds(
                df_samp["actual_label"],
                df_samp["predicted_probability"],
                thresholds,
            )[0]
            fig_t = go.Figure()
            for metric, color in [
                ("f1", "#4263A5"),
                ("precision", "#10B981"),
                ("recall", "#F59E0B"),
            ]:
                if metric in df_thresh:
                    fig_t.add_trace(go.Scatter(
                        x=df_thresh["threshold"],
                        y=df_thresh[metric],
                        name=metric.replace("_", " ").title(),
                        mode="lines+markers",
                        line={"color": color, "width": 2},
                    ))
            if offline_threshold is not None:
                fig_t.add_vline(
                    x=float(offline_threshold),
                    line_dash="dot",
                    line_color="#6B7280",
                    annotation_text="Recorded offline threshold",
                )
            _style_chart(fig_t, height=330)
            fig_t.update_layout(
                title="Offline test metrics by decision threshold",
                xaxis_title="Decision threshold",
                yaxis_title="Metric score",
            )
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("Offline threshold analysis is unavailable: prediction artifact columns are missing.")
    else:
        st.info("Offline threshold analysis is unavailable because sample predictions were not exported.")

    confusion_matrix = report.get("confusion_matrix")
    if confusion_matrix is not None:
        cm_arr = np.asarray(confusion_matrix)
        if cm_arr.shape == (2, 2):
            st.markdown("#### Offline confusion matrix")
            fig_cm = px.imshow(
                cm_arr,
                text_auto=True,
                x=["Predicted retained", "Predicted churned"],
                y=["Actual retained", "Actual churned"],
                color_continuous_scale="Blues",
            )
            _style_chart(fig_cm, height=330)
            fig_cm.update_layout(coloraxis_colorbar_title="Customers")
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Offline confusion matrix is unavailable or has an unsupported shape.")
    else:
        st.info("Offline confusion matrix is unavailable.")

    if features_path.exists() and model_path.exists():
        import joblib
        pipeline = joblib.load(model_path)
        feat_names = getattr(pipeline, "feature_names", None)
        model = getattr(pipeline, "model", None)
        if feat_names is not None and len(feat_names) > 0 and model is not None:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_[0])
            else:
                importances = None

            if importances is not None and len(importances) == len(feat_names):
                st.markdown("#### Loaded artifact feature importance")
                st.caption("Derived from the local model artifact; not a production performance metric.")
                df_imp = pd.DataFrame({
                    "Feature": feat_names,
                    "Importance": importances,
                }).sort_values("Importance", ascending=False).head(15)
                fig_imp = px.bar(
                    df_imp,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    template="plotly_white",
                )
                fig_imp.update_layout(height=380, yaxis={"autorange": "reversed"})
                fig_imp.update_traces(marker_color="#4263A5")
                st.plotly_chart(fig_imp, use_container_width=True)
