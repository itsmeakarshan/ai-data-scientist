"""
AutoDS Visualizer Tool
Generates high-resolution statistical charts and diagnostic figures saved to reports/artifacts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from backend.app.core.config import settings

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def save_plot_artifact(fig: plt.Figure, filename: str) -> str:
    """Save matplotlib figure to reports/artifacts/ and return relative path."""
    artifacts_dir = settings.reports_artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    file_path = artifacts_dir / filename
    fig.savefig(file_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(file_path.relative_to(Path(settings.REPORTS_DIR).parent))


def generate_roc_pr_plots(
    roc_data: Dict[str, Any],
    pr_data: Dict[str, Any],
    roc_auc: float,
    pr_auc: float,
    model_name: str,
    run_id: str
) -> Dict[str, str]:
    """Generate and save ROC and Precision-Recall curve plots (supports binary & multiclass One-vs-Rest)."""
    artifacts = {}

    # 1. ROC Curve
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    is_mc = roc_data.get("is_multiclass", False)

    if is_mc and roc_data.get("per_class"):
        per_class = roc_data["per_class"]
        macro_fpr = roc_data.get("macro_fpr", [0, 1])
        macro_tpr = roc_data.get("macro_tpr", [0, 1])
        macro_auc = roc_data.get("macro_auc", roc_auc)

        # Plot macro-average curve
        ax.plot(macro_fpr, macro_tpr, color="#1e40af", lw=2.5, label=f"Macro-average (AUC = {macro_auc:.3f})")

        # Plot per-class curves (up to 8 classes with distinct colors)
        palette = sns.color_palette("tab10", n_colors=max(len(per_class), 3))
        for idx, (cls_name, c_data) in enumerate(list(per_class.items())[:8]):
            c_fpr = c_data.get("fpr", [])
            c_tpr = c_data.get("tpr", [])
            c_auc = c_data.get("auc", 0.5)
            if c_fpr and c_tpr:
                ax.plot(c_fpr, c_tpr, lw=1.5, alpha=0.7, color=palette[idx % len(palette)], label=f"Class {cls_name} (AUC = {c_auc:.3f})")

        ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Chance (AUC = 0.500)")
        ax.set_title(f"ROC Curve (OvR) — {model_name}", fontsize=11, fontweight="bold")
    else:
        fpr = roc_data.get("fpr", [0, 1])
        tpr = roc_data.get("tpr", [0, 1])
        ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Chance (AUC = 0.500)")
        ax.set_title(f"ROC Curve — {model_name}", fontsize=11, fontweight="bold")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=8 if is_mc else 9)
    artifacts["roc_curve_path"] = save_plot_artifact(fig, f"{run_id}_{model_name}_roc.png")

    # 2. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    is_mc_pr = pr_data.get("is_multiclass", False)

    if is_mc_pr and pr_data.get("per_class"):
        per_class_pr = pr_data["per_class"]
        macro_rec = pr_data.get("macro_recall", [0, 1])
        macro_prec = pr_data.get("macro_precision", [1, 0])
        macro_pr_auc = pr_data.get("macro_pr_auc", pr_auc)
        baseline_prev = pr_data.get("baseline_prevalence", 0.1)

        # Plot macro-average PR curve
        ax.plot(macro_rec, macro_prec, color="#065f46", lw=2.5, label=f"Macro-average (PR-AUC = {macro_pr_auc:.3f})")

        # Plot per-class PR curves
        palette = sns.color_palette("tab10", n_colors=max(len(per_class_pr), 3))
        for idx, (cls_name, c_data) in enumerate(list(per_class_pr.items())[:8]):
            c_rec = c_data.get("recall", [])
            c_prec = c_data.get("precision", [])
            c_pr_auc = c_data.get("pr_auc", 0.0)
            if c_rec and c_prec:
                ax.plot(c_rec, c_prec, lw=1.5, alpha=0.7, color=palette[idx % len(palette)], label=f"Class {cls_name} (PR-AUC = {c_pr_auc:.3f})")

        ax.plot([0, 1], [baseline_prev, baseline_prev], color="#9ca3af", lw=1.5, linestyle="--", label=f"Chance Baseline ({baseline_prev:.3f})")
        ax.set_title(f"Precision-Recall Curve (OvR) — {model_name}", fontsize=11, fontweight="bold")
    else:
        prec = pr_data.get("precision", [1, 0])
        rec = pr_data.get("recall", [0, 1])
        baseline_prev = pr_data.get("baseline_prevalence", 0.1)
        ax.plot(rec, prec, color="#10b981", lw=2, label=f"{model_name} (PR-AUC = {pr_auc:.3f})")
        ax.plot([0, 1], [baseline_prev, baseline_prev], color="#9ca3af", lw=1.5, linestyle="--", label=f"Chance Baseline ({baseline_prev:.3f})")
        ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=11, fontweight="bold")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Positive Class)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (Positive Class)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=8 if is_mc_pr else 9)
    artifacts["pr_curve_path"] = save_plot_artifact(fig, f"{run_id}_{model_name}_pr.png")

    return artifacts


def generate_confusion_matrix_plot(
    cm: List[List[int]],
    model_name: str,
    run_id: str,
    class_labels: Optional[List[Any]] = None,
    threshold: Optional[float] = None
) -> str:
    """Generate and save Confusion Matrix heatmap with explicit decision threshold caption (supports binary & multiclass)."""
    cm_arr = np.array(cm)
    if class_labels is None or len(class_labels) != len(cm_arr):
        if cm_arr.shape == (2, 2):
            labels = ["Negative (0)", "Positive (1)"]
        else:
            labels = [f"Class {i}" for i in range(len(cm_arr))]
    else:
        labels = [str(l) for l in class_labels]

    fig_w = max(5.5, len(cm_arr) * 0.85)
    fig_h = max(4.5, len(cm_arr) * 0.75)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
        annot_kws={"size": 11 if len(cm_arr) <= 6 else 9, "weight": "bold"}
    )
    ax.set_xlabel("Predicted Label (Holdout Test Set)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Ground Truth", fontsize=11, fontweight="bold")

    if threshold is not None:
        title_str = f"Confusion Matrix — {model_name} (Locked Threshold = {threshold:.2f})"
    elif len(cm_arr) > 2:
        title_str = f"Multiclass Confusion Matrix — {model_name} (Holdout Test Set)"
    else:
        title_str = f"Confusion Matrix — {model_name} (Holdout Test Set)"

    ax.set_title(title_str, fontsize=11, fontweight="bold")
    return save_plot_artifact(fig, f"{run_id}_{model_name}_cm.png")


def generate_feature_importance_plot(
    feature_rankings: List[Dict[str, Any]],
    model_name: str,
    run_id: str,
    top_n: int = 12
) -> str:
    """Generate horizontal bar chart of top predictive features."""
    top_items = feature_rankings[:top_n]
    if not top_items:
        return ""

    features = [r["feature"] for r in reversed(top_items)]
    importances = [r["importance_pct"] for r in reversed(top_items)]

    fig, ax = plt.subplots(figsize=(7.5, max(4.5, len(features) * 0.38)))
    bars = ax.barh(features, importances, color="#3b82f6", alpha=0.9, edgecolor="#1d4ed8")
    ax.set_xlabel("Relative Importance (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top Predictive Drivers — {model_name}", fontsize=12, fontweight="bold", pad=12)

    # Value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%",
                va="center", ha="left", fontsize=9, color="#1e293b", weight="bold")

    ax.set_xlim(0, max(importances) * 1.25 if importances else 100)
    fig.text(
        0.5, -0.01,
        "Note: These are model-derived predictive associations, not causal effects.",
        ha="center", fontsize=8, color="#64748b", style="italic"
    )
    return save_plot_artifact(fig, f"{run_id}_{model_name}_feature_imp.png")


def generate_actual_vs_predicted_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    run_id: str,
    problem_type: str = "regression"
) -> str:
    """Generate Actual vs Predicted scatter or time series comparison."""
    fig, ax = plt.subplots(figsize=(6.5, 5))

    if problem_type == "forecasting":
        # Plot sequence
        steps = min(len(y_true), 120)
        ax.plot(range(steps), y_true[:steps], label="Actual Ground Truth", color="#1e293b", lw=2)
        ax.plot(range(steps), y_pred[:steps], label="Forecasted Horizon", color="#0284c7", lw=2, linestyle="--")
        ax.set_xlabel("Time Horizon Steps", fontsize=11, fontweight="bold")
        ax.set_ylabel("Target Value", fontsize=11, fontweight="bold")
        ax.set_title(f"Forecasting Trajectory — {model_name}", fontsize=12, fontweight="bold")
        ax.legend(frameon=True)
    else:
        # Scatter plot
        indices = np.random.choice(len(y_true), min(len(y_true), 500), replace=False)
        ax.scatter(y_true[indices], y_pred[indices], alpha=0.55, color="#6366f1", edgecolors="none")

        min_v = min(float(np.min(y_true)), float(np.min(y_pred)))
        max_v = max(float(np.max(y_true)), float(np.max(y_pred)))
        ax.plot([min_v, max_v], [min_v, max_v], color="#ef4444", linestyle="--", lw=2, label="Ideal 45° Line")

        ax.set_xlabel("Actual Ground Truth (Holdout)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted Value", fontsize=11, fontweight="bold")
        ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=12, fontweight="bold")
        ax.legend(frameon=True)

    return save_plot_artifact(fig, f"{run_id}_{model_name}_actual_vs_pred.png")


def generate_residual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    run_id: str,
    problem_type: str = "regression"
) -> str:
    """Generate Residual Diagnostics plot (Residuals vs Predicted and Residual Error Distribution)."""
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: Residuals vs Predicted
    indices = np.random.choice(len(residuals), min(len(residuals), 500), replace=False)
    ax1.scatter(y_pred[indices], residuals[indices], alpha=0.55, color="#6366f1", edgecolors="none")
    ax1.axhline(0, color="#ef4444", linestyle="--", lw=2, label="Zero Error Line")
    ax1.set_xlabel("Predicted Values", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Residuals (Actual - Predicted)", fontsize=10, fontweight="bold")
    ax1.set_title("Residuals vs Predicted", fontsize=11, fontweight="bold")
    ax1.legend(frameon=True, fontsize=9)

    # Right: Residual Distribution
    sns.histplot(residuals, kde=True, color="#0284c7", ax=ax2, bins=25)
    ax2.axvline(0, color="#ef4444", linestyle="--", lw=1.5)
    ax2.set_xlabel("Residual Error", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=10, fontweight="bold")
    ax2.set_title("Residual Error Distribution", fontsize=11, fontweight="bold")

    fig.suptitle(f"Residual Diagnostics — {model_name}", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    return save_plot_artifact(fig, f"{run_id}_{model_name}_residuals.png")


def generate_correlation_heatmap(
    correlations: Dict[str, Any],
    run_id: str
) -> Optional[str]:
    """Generate Correlation matrix heatmap for top numeric features."""
    matrix = correlations.get("matrix", {})
    if not matrix or len(matrix) < 2:
        return None

    cols = list(matrix.keys())[:15]
    sub_matrix = [[matrix[r].get(c, 0.0) for c in cols] for r in cols]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        sub_matrix,
        xticklabels=cols,
        yticklabels=cols,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        ax=ax
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    ax.set_title("Numeric Feature Correlation Matrix", fontsize=12, fontweight="bold")
    return save_plot_artifact(fig, f"{run_id}_eda_correlation.png")
