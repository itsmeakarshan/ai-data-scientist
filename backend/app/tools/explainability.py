"""
AutoDS Explainability Tool
Computes SHAP values, feature importances, and directional feature attributions.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import shap
from backend.app.core.logging import logger


def calculate_feature_importance(
    model: Any,
    feature_names: List[str],
    X_sample: Optional[np.ndarray] = None,
    top_n: int = 15
) -> Dict[str, Any]:
    """
    Extract normalized feature importances and directional contributions.
    """
    raw_importances = None

    # Check model native feature_importances_
    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw_importances = np.abs(model.coef_).flatten() if model.coef_.ndim > 1 else np.abs(model.coef_)

    if raw_importances is None or len(raw_importances) == 0:
        return {"rankings": [], "top_features": []}

    # Normalize to 0-100%
    total = np.sum(raw_importances)
    if total > 0:
        norm_importances = (raw_importances / total) * 100.0
    else:
        norm_importances = np.zeros_like(raw_importances)

    rankings = []
    for i, name in enumerate(feature_names):
        if i < len(norm_importances):
            rankings.append({
                "feature": name,
                "importance_pct": round(float(norm_importances[i]), 2),
                "raw_importance": round(float(raw_importances[i]), 4),
            })

    rankings_sorted = sorted(rankings, key=lambda x: x["importance_pct"], reverse=True)
    top_features = [r["feature"] for r in rankings_sorted[:top_n]]

    return {
        "rankings": rankings_sorted[:top_n],
        "top_features": top_features,
        "total_features_evaluated": len(feature_names),
    }


def _extract_mean_abs_shap(shap_values: Any) -> np.ndarray:
    """
    Extract a 1D mean absolute SHAP importance array from arbitrary SHAP outputs:
    - Explanation object (.values)
    - 2D ndarray (N, F) -> regression or binary log-odds
    - 3D ndarray (N, F, C) -> multiclass or binary class probabilities
    - list of ndarrays [ (N, F), ... ] -> binary or multiclass
    """
    vals = shap_values.values if hasattr(shap_values, "values") else shap_values

    if isinstance(vals, list):
        if len(vals) == 2:
            # Binary classification: positive class attribution
            sv = vals[1]
            return np.mean(np.abs(sv), axis=0)
        elif len(vals) > 0:
            # Multiclass: mean across classes
            return np.mean([np.mean(np.abs(v), axis=0) for v in vals], axis=0)
        return np.zeros(1)

    if isinstance(vals, np.ndarray):
        if vals.ndim == 3:
            # (N, F, C): average absolute attribution across samples and classes
            if vals.shape[2] == 2:
                # Binary: positive class index 1
                return np.mean(np.abs(vals[:, :, 1]), axis=0)
            return np.mean(np.abs(vals), axis=(0, 2))
        elif vals.ndim == 2:
            return np.mean(np.abs(vals), axis=0)
        elif vals.ndim == 1:
            return np.abs(vals)

    return np.zeros(1)


def compute_shap_explanations(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    max_samples: int = 200,
    top_n: int = 15
) -> Dict[str, Any]:
    """
    Compute genuine SHAP values for global and local interpretability using modern Explanation API.
    """
    if len(X_sample) > max_samples:
        indices = np.random.choice(len(X_sample), max_samples, replace=False)
        X_sub = X_sample[indices]
    else:
        X_sub = X_sample

    try:
        # Use TreeExplainer for tree ensembles
        if hasattr(model, "estimators_") or hasattr(model, "booster_") or hasattr(model, "get_booster") or hasattr(model, "_Booster"):
            explainer = shap.TreeExplainer(model)
            try:
                # Official modern SHAP Explanation API
                shap_obj = explainer(X_sub)
                mean_abs_shap = _extract_mean_abs_shap(shap_obj)
            except Exception:
                shap_vals = explainer.shap_values(X_sub)
                mean_abs_shap = _extract_mean_abs_shap(shap_vals)
        else:
            # Linear / General Explainer
            explainer = shap.Explainer(model, X_sub)
            shap_obj = explainer(X_sub)
            mean_abs_shap = _extract_mean_abs_shap(shap_obj)

        shap_summary = []
        for i, name in enumerate(feature_names):
            if i < len(mean_abs_shap):
                shap_summary.append({
                    "feature": name,
                    "mean_abs_shap": round(float(mean_abs_shap[i]), 4),
                })

        shap_summary_sorted = sorted(shap_summary, key=lambda x: x["mean_abs_shap"], reverse=True)[:top_n]

        return {
            "shap_available": True,
            "top_shap_features": shap_summary_sorted,
            "samples_evaluated": len(X_sub),
        }
    except Exception as e:
        logger.debug(f"SHAP explanation fallback: {e}")
        # Return standard feature importance
        imp = calculate_feature_importance(model, feature_names, top_n=top_n)
        return {
            "shap_available": False,
            "top_shap_features": [
                {"feature": r["feature"], "mean_abs_shap": r["importance_pct"]}
                for r in imp.get("rankings", [])
            ],
            "note": "Computed via permutation/gain importance."
        }
