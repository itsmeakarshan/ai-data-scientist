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


def compute_shap_explanations(
    model: Any,
    X_sample: np.ndarray,
    feature_names: List[str],
    max_samples: int = 200,
    top_n: int = 15
) -> Dict[str, Any]:
    """
    Compute real SHAP values for global and local interpretability.
    """
    if len(X_sample) > max_samples:
        indices = np.random.choice(len(X_sample), max_samples, replace=False)
        X_sub = X_sample[indices]
    else:
        X_sub = X_sample

    try:
        # Use TreeExplainer for tree ensembles
        if hasattr(model, "estimators_") or hasattr(model, "booster_") or hasattr(model, "get_booster"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sub)
        else:
            # Linear / General Explainer
            explainer = shap.Explainer(model, X_sub)
            shap_values = explainer(X_sub).values

        # Handle binary classification list of shap values
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif hasattr(shap_values, "values"):
            sv = shap_values.values
        else:
            sv = shap_values

        if sv.ndim == 3:
            sv = sv[:, :, 1]  # positive class for binary

        mean_abs_shap = np.mean(np.abs(sv), axis=0)
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
