"""
AutoDS Visualizer Tool
Generates high-resolution statistical charts and diagnostic figures saved to reports/artifacts.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from backend.app.core.config import settings
from backend.app.core.logging import logger


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
    """Generate and save ROC and Precision-Recall curve plots."""
    artifacts = {}
    
    # 1. ROC Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr = roc_data.get("fpr", [0, 1])
    tpr = roc_data.get("tpr", [0, 1])
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title(f"Receiver Operating Characteristic — {model_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    artifacts["roc_curve_path"] = save_plot_artifact(fig, f"{run_id}_{model_name}_roc.png")

    # 2. Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    prec = pr_data.get("precision", [1, 0])
    rec = pr_data.get("recall", [0, 1])
    ax.plot(rec, prec, color="#10b981", lw=2, label=f"{model_name} (PR-AUC = {pr_auc:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", frameon=True)
    artifacts["pr_curve_path"] = save_plot_artifact(fig, f"{run_id}_{model_name}_pr.png")

    return artifacts


def generate_confusion_matrix_plot(
    cm: List[List[int]],
    model_name: str,
    run_id: str,
    class_labels: Optional[List[str]] = None
) -> str:
    """Generate and save Confusion Matrix heatmap."""
    cm_arr = np.array(cm)
    labels = class_labels or [f"Class {i}" for i in range(len(cm_arr))]
    
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"}
    )
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual Ground Truth", fontsize=11, fontweight="bold")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
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

    fig, ax = plt.subplots(figsize=(7, max(4.0, len(features) * 0.35)))
    bars = ax.barh(features, importances, color="#3b82f6", alpha=0.9, edgecolor="#1d4ed8")
    ax.set_xlabel("Relative Importance (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top Predictive Drivers — {model_name}", fontsize=12, fontweight="bold")
    
    # Value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%",
                va="center", ha="left", fontsize=9, color="#1e293b", weight="bold")

    ax.set_xlim(0, max(importances) * 1.2 if importances else 100)
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
        ax.plot(range(steps), y_true[:steps], label="Actual Sales/Demand", color="#1e293b", lw=2)
        ax.plot(range(steps), y_pred[:steps], label="Forecasted", color="#0284c7", lw=2, linestyle="--")
        ax.set_xlabel("Time Horizon Steps", fontsize=11, fontweight="bold")
        ax.set_ylabel("Target Value", fontsize=11, fontweight="bold")
        ax.set_title(f"Forecasting Trajectory — {model_name}", fontsize=12, fontweight="bold")
        ax.legend(frameon=True)
    else:
        # Scatter plot
        indices = np.random.choice(len(y_true), min(len(y_true), 500), replace=False)
        ax.scatter(y_true[indices], y_pred[indices], alpha=0.5, color="#6366f1", edgecolors="none")
        
        min_v = min(np.min(y_true), np.min(y_pred))
        max_v = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_v, max_v], [min_v, max_v], color="#ef4444", linestyle="--", lw=2, label="Perfect Line")
        
        ax.set_xlabel("Actual Ground Truth", fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted Value", fontsize=11, fontweight="bold")
        ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=12, fontweight="bold")
        ax.legend(frameon=True)

    return save_plot_artifact(fig, f"{run_id}_{model_name}_actual_vs_pred.png")


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
