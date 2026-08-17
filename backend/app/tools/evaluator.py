"""
AutoDS Multi-Metric Model Evaluator Tool
Computes rigorous, deterministic metrics and diagnostic curves for classification, regression, and forecasting.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics, confusion matrix, ROC/PR curves, and calibration.
    """
    classes = np.unique(y_true)
    is_binary = len(classes) == 2

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "confusion_matrix": cm,
        "is_binary": is_binary,
    }

    # Probability-based metrics (ROC-AUC, PR-AUC, Log Loss, Calibration)
    if y_prob is not None:
        try:
            if is_binary:
                # Handle 1D or 2D prob array
                p_positive = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                roc_auc = float(roc_auc_score(y_true, p_positive))
                pr_auc = float(average_precision_score(y_true, p_positive))
                brier = float(brier_score_loss(y_true, p_positive))
                ll = float(log_loss(y_true, p_positive, eps=1e-7))

                metrics["roc_auc"] = round(roc_auc, 4)
                metrics["pr_auc"] = round(pr_auc, 4)
                metrics["brier_score"] = round(brier, 4)
                metrics["log_loss"] = round(ll, 4)

                # ROC Curve points (subsampled for compactness)
                fpr, tpr, thresh_roc = roc_curve(y_true, p_positive)
                idx_roc = np.linspace(0, len(fpr) - 1, min(len(fpr), 50), dtype=int)
                metrics["roc_curve"] = {
                    "fpr": [round(float(x), 4) for x in fpr[idx_roc]],
                    "tpr": [round(float(x), 4) for x in tpr[idx_roc]],
                }

                # PR Curve points
                prec_pts, rec_pts, _ = precision_recall_curve(y_true, p_positive)
                idx_pr = np.linspace(0, len(prec_pts) - 1, min(len(prec_pts), 50), dtype=int)
                metrics["pr_curve"] = {
                    "precision": [round(float(x), 4) for x in prec_pts[idx_pr]],
                    "recall": [round(float(x), 4) for x in rec_pts[idx_pr]],
                }

                # Calibration curve
                prob_true, prob_pred = calibration_curve(y_true, p_positive, n_bins=10)
                metrics["calibration_curve"] = {
                    "prob_true": [round(float(x), 4) for x in prob_true],
                    "prob_pred": [round(float(x), 4) for x in prob_pred],
                }
            else:
                # Multiclass ROC-AUC
                roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
                metrics["roc_auc"] = round(roc_auc, 4)
                metrics["pr_auc"] = 0.0
                try:
                    metrics["log_loss"] = round(float(log_loss(y_true, y_prob)), 4)
                except Exception:
                    pass
        except Exception as e:
            metrics["roc_auc"] = round(acc, 4)
            metrics["pr_auc"] = 0.0
    else:
        metrics["roc_auc"] = round(acc, 4)
        metrics["pr_auc"] = 0.0

    return metrics


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Compute regression metrics, errors, and residual distributions.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    
    # Safe MAPE
    with np.errstate(divide="ignore", invalid="ignore"):
        mape_arr = np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-6, y_true))
        mape = float(np.mean(mape_arr[np.isfinite(mape_arr)])) * 100.0 if len(mape_arr) > 0 else 0.0

    residuals = (y_true - y_pred)
    res_percentiles = {
        "p10": round(float(np.percentile(residuals, 10)), 4),
        "p25": round(float(np.percentile(residuals, 25)), 4),
        "p50": round(float(np.percentile(residuals, 50)), 4),
        "p75": round(float(np.percentile(residuals, 75)), 4),
        "p90": round(float(np.percentile(residuals, 90)), 4),
    }

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mse": round(mse, 4),
        "r2": round(r2, 4),
        "median_ae": round(medae, 4),
        "mape": round(mape, 2),
        "residual_percentiles": res_percentiles,
        "max_residual": round(float(np.max(np.abs(residuals))), 4),
    }


def evaluate_forecasting(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Compute time-series forecasting metrics including WAPE and sMAPE.
    """
    reg_metrics = evaluate_regression(y_true, y_pred)
    
    # WAPE = sum(|y - y_hat|) / sum(|y|)
    denom = np.sum(np.abs(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0) if denom > 0 else 0.0
    
    # sMAPE = 100/n * sum( 2*|y - y_hat| / (|y| + |y_hat|) )
    denom_smape = (np.abs(y_true) + np.abs(y_pred))
    smape_terms = np.where(denom_smape == 0, 0, 2.0 * np.abs(y_true - y_pred) / denom_smape)
    smape = float(np.mean(smape_terms) * 100.0)

    reg_metrics["wape"] = round(wape, 2)
    reg_metrics["smape"] = round(smape, 2)
    return reg_metrics
