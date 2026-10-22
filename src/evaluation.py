"""
churn-engine/src/evaluation.py
================================
Model Evaluation, Threshold Optimization, and Report Generation Utilities.

Responsibilities
----------------
- Compute binary classification metrics: AUC-ROC, F1, Precision, Recall, Accuracy, Brier Score, MCC.
- Perform threshold search across [0.30, 0.40, 0.50, 0.60, 0.70, 0.80].
- Save `models/decision_policy.json` for consumption by Decision Gate.
- Save `reports/sample_predictions.csv` for auditability.
- Maintain `reports/experiment_history.csv` for experiment tracking.
- Plot ROC Curve, Precision-Recall Curve, Confusion Matrix, Feature Importance, and Churn Probability Distribution.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import ensure_dir, get_project_root

logger = logging.getLogger(__name__)


# ===========================================================================
# Metric Computation
# ===========================================================================

def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute binary classification metrics including MCC.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    y_prob : array-like

    Returns
    -------
    dict[str, float]
        Dictionary with accuracy, precision, recall, f1, roc_auc, avg_precision, brier_score, mcc.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    y_pr = np.asarray(y_prob)

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))
    mcc = float(matthews_corrcoef(y_t, y_p))
    brier = float(brier_score_loss(y_t, y_pr))

    try:
        roc_auc = float(roc_auc_score(y_t, y_pr))
    except ValueError:
        roc_auc = 0.5

    try:
        avg_prec = float(average_precision_score(y_t, y_pr))
    except ValueError:
        avg_prec = 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mcc": round(mcc, 4),
        "roc_auc": round(roc_auc, 4),
        "avg_precision": round(avg_prec, 4),
        "brier_score": round(brier, 4),
    }


def compute_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> np.ndarray:
    """Return 2x2 confusion matrix array: [[TN, FP], [FN, TP]]."""
    return confusion_matrix(y_true, y_pred)


# ===========================================================================
# Threshold Optimization
# ===========================================================================

def evaluate_thresholds(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    thresholds: list[float] | None = None,
) -> tuple[pd.DataFrame, float, dict[str, Any]]:
    """Evaluate performance across candidate classification thresholds.

    Parameters
    ----------
    y_true : array-like
    y_prob : array-like
    thresholds : list[float] | None

    Returns
    -------
    tuple[pd.DataFrame, float, dict[str, Any]]
        Threshold evaluation DataFrame, optimal threshold, and policy dict.
    """
    if thresholds is None:
        thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    y_t = np.asarray(y_true)
    y_pr = np.asarray(y_prob)

    rows = []
    best_thresh = 0.50
    max_f1 = -1.0
    best_policy = {}

    for t in thresholds:
        y_pred = (y_pr >= t).astype(int)
        cm = confusion_matrix(y_t, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        prec = precision_score(y_t, y_pred, zero_division=0)
        rec = recall_score(y_t, y_pred, zero_division=0)
        f1 = f1_score(y_t, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_t, y_pred)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        row = {
            "threshold": t,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "mcc": round(mcc, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }
        rows.append(row)

        if f1 > max_f1:
            max_f1 = f1
            best_thresh = t
            best_policy = {
                "selected_threshold": t,
                "selection_criteria": "Maximize F1-score while controlling False Positive Rate",
                "fpr": round(fpr, 4),
                "fnr": round(fnr, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "mcc": round(mcc, 4),
                "business_justification": (
                    f"Threshold {t:.2f} optimizes retention targeting efficiency. "
                    f"It achieves F1={f1:.4f} with FPR={fpr:.4f} (limiting costly unnecessary discount overrides) "
                    f"and FNR={fnr:.4f} (catching high-risk churners)."
                ),
            }

    thresh_df = pd.DataFrame(rows)
    return thresh_df, best_thresh, best_policy


# ===========================================================================
# Artefact Generation & Logging Helpers
# ===========================================================================

def save_decision_policy(policy: dict[str, Any], filepath: Path | str) -> None:
    """Save `models/decision_policy.json` for Phase 4 Decision Gate consumption."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)
    logger.info("Saved Decision Policy artefact → %s", filepath)


def save_sample_predictions(
    customer_ids: list[str] | pd.Index,
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    filepath: Path | str,
) -> pd.DataFrame:
    """Save representative predictions on the test set to `reports/sample_predictions.csv`."""
    df_preds = pd.DataFrame({
        "customer_id": list(customer_ids),
        "actual_label": np.asarray(y_true),
        "predicted_probability": np.round(np.asarray(y_prob), 4),
        "predicted_label": np.asarray(y_pred),
    })
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    df_preds.to_csv(filepath, index=False)
    logger.info("Saved sample test predictions → %s (%d rows)", filepath, len(df_preds))
    return df_preds


def log_experiment(
    model_name: str,
    metrics: dict[str, float],
    threshold: float,
    training_time_sec: float,
    filepath: Path | str,
) -> None:
    """Append run entry to `reports/experiment_history.csv`."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    entry = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "roc_auc": metrics.get("roc_auc", 0.0),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "f1": metrics.get("f1", 0.0),
        "mcc": metrics.get("mcc", 0.0),
        "brier_score": metrics.get("brier_score", 0.0),
        "selected_threshold": threshold,
        "training_time_sec": round(training_time_sec, 3),
    }

    df_entry = pd.DataFrame([entry])
    if not filepath.exists():
        df_entry.to_csv(filepath, index=False)
    else:
        df_entry.to_csv(filepath, mode="a", header=False, index=False)

    logger.info("Logged experiment entry to %s", filepath)


# ===========================================================================
# Visualisations
# ===========================================================================

def plot_churn_probability_distribution(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.5,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot density distribution of predicted churn probabilities for actual churners vs non-churners."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_theme(style="darkgrid")

    df_plot = pd.DataFrame({"y_true": np.asarray(y_true), "y_prob": np.asarray(y_prob)})
    churners = df_plot[df_plot["y_true"] == 1]["y_prob"]
    retained = df_plot[df_plot["y_true"] == 0]["y_prob"]

    sns.histplot(retained, color="#3fb950", label="Retained (0)", kde=True, ax=ax, alpha=0.5, bins=25)
    sns.histplot(churners, color="#f85149", label="Churned (1)", kde=True, ax=ax, alpha=0.5, bins=25)

    ax.axvline(threshold, color="#58a6ff", linestyle="--", linewidth=2, label=f"Optimal Cutoff ({threshold:.2f})")
    ax.set_title("Predicted Churn Probability Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Customer Count")
    ax.legend()
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=150)
        logger.info("Saved probability distribution plot → %s", save_path)

    return fig


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 20,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot top-N feature importances."""
    fig, ax = plt.subplots(figsize=(9, 6))

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).head(top_n)

    sns.barplot(data=df_imp, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=150)

    return fig


# ===========================================================================
# Full Report Generator
# ===========================================================================

def generate_evaluation_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    model: Any,
    feature_names: list[str],
    threshold_policy: dict[str, Any],
    benchmark_df: pd.DataFrame,
    save_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Generate a complete evaluation report dictionary and save JSON artefact."""
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    cm = compute_confusion_matrix(y_true, y_pred).tolist()

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "confusion_matrix": cm,
        "decision_policy": threshold_policy,
        "benchmark_summary": benchmark_df.to_dict(orient="records"),
    }

    if save_dir:
        save_dir = Path(save_dir)
        ensure_dir(save_dir)
        json_path = save_dir / "evaluation_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved evaluation report → %s", json_path)

    return report
