"""
AutoDS Critic Agent & Methodology Auditor Tool
Audits experiments for data leakage, severe overfitting, improper validation, and flawed metric selection.
"""

from typing import Any, Dict, List, Optional
import numpy as np


def critique_experiment(
    model_name: str,
    problem_type: str,
    metrics: Dict[str, Any],
    feature_names: List[str],
    validation_strategy: str,
    target_column: Optional[str] = None,
    raw_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Perform deep methodological audit of an experiment run.
    Identifies leakage, overfitting, validation hazards, and recommends corrective iterations.
    """
    findings: List[Dict[str, Any]] = []
    requires_iteration = False
    remediation_actions: List[str] = []

    test_metrics = metrics.get("test", {})
    train_metrics = metrics.get("train", {})
    cv_mean = metrics.get("cv_mean", 0.0)

    # 1. Audit Overfitting (Train vs Test divergence)
    if problem_type == "classification":
        train_score = train_metrics.get("roc_auc", train_metrics.get("accuracy", 0.0))
        test_score = test_metrics.get("roc_auc", test_metrics.get("accuracy", 0.0))
        divergence = train_score - test_score

        if train_score >= 0.98 and test_score < 0.82:
            findings.append({
                "issue_type": "severe_overfitting",
                "severity": "critical",
                "description": f"Extreme train-test gap ({divergence:.3f}). Model memorized training set (Train ROC-AUC: {train_score:.3f}, Test ROC-AUC: {test_score:.3f}).",
                "affected_components": [model_name],
                "remediation": "Regularize model (limit tree depth, increase min_child_samples, add L1/L2 penalties)."
            })
            requires_iteration = True
            remediation_actions.append("REGULARIZE_MODEL")
        elif divergence > 0.12:
            findings.append({
                "issue_type": "moderate_overfitting",
                "severity": "warning",
                "description": f"Noticeable train-test gap ({divergence:.3f}). Train score: {train_score:.3f}, Test score: {test_score:.3f}.",
                "affected_components": [model_name],
                "remediation": "Consider early stopping or hyperparameter tuning to improve test generalization."
            })

    elif problem_type in ("regression", "forecasting"):
        train_rmse = train_metrics.get("rmse", 0.0)
        test_rmse = test_metrics.get("rmse", 0.0)
        if train_rmse > 0 and (test_rmse / train_rmse) > 1.8:
            findings.append({
                "issue_type": "severe_overfitting",
                "severity": "critical",
                "description": f"Test RMSE ({test_rmse:.2f}) is {test_rmse/train_rmse:.1f}x higher than Train RMSE ({train_rmse:.2f}).",
                "affected_components": [model_name],
                "remediation": "Apply L2 shrinkage (Ridge), prune tree estimators, or reduce feature space."
            })
            requires_iteration = True
            remediation_actions.append("REGULARIZE_MODEL")

    # 2. Audit Potential Data / Target Leakage
    # Check for known prospective domain leakage variables (e.g. call duration in telemarketing, post-event metrics)
    prospective_leakage_indicators = ["duration", "post_event", "post_call", "after_outcome", "future_val"]
    if raw_columns:
        for col in raw_columns:
            col_lower = col.lower()
            if any(ind in col_lower for ind in prospective_leakage_indicators):
                in_features = any(col_lower == str(f).lower() or col_lower in str(f).lower() for f in feature_names)
                if in_features:
                    findings.append({
                        "issue_type": "domain_target_leakage",
                        "severity": "critical",
                        "description": f"Feature '{col}' represents prospective/post-event information unknown at prediction time, creating deployment-time data leakage.",
                        "affected_components": [col],
                        "remediation": f"Drop '{col}' feature and retrain model suite to obtain honest prospective performance."
                    })
                    requires_iteration = True
                    remediation_actions.append("REMOVE_LEAKY_FEATURES")

    # Check suspiciously high test score indicative of target proxy
    if problem_type == "classification" and test_metrics.get("roc_auc", 0.0) >= 0.995:
        findings.append({
            "issue_type": "suspicious_performance",
            "severity": "warning",
            "description": f"ROC-AUC is {test_metrics.get('roc_auc'):.4f}, which is unusually high for real-world tabular data. Check for indirect target proxies or ID duplicates.",
            "affected_components": [model_name],
            "remediation": "Audit individual feature correlations and inspect SHAP distributions for proxy identifiers."
        })

    # 3. Audit Validation Strategy
    if problem_type == "forecasting" and "kfold" in validation_strategy.lower() and "time" not in validation_strategy.lower():
        findings.append({
            "issue_type": "improper_validation",
            "severity": "critical",
            "description": f"Random K-Fold splitting ({validation_strategy}) was used on time-series data, leaking future timestamps into past training.",
            "affected_components": ["validation_pipeline"],
            "remediation": "Switch to chronological Walk-Forward / TimeSeriesSplit validation."
        })
        requires_iteration = True
        remediation_actions.append("SWITCH_VALIDATION_STRATEGY")

    # 4. Calibration Audit (Classification)
    if problem_type == "classification" and test_metrics.get("is_binary"):
        brier = test_metrics.get("brier_score", 0.0)
        if brier > 0.20:
            findings.append({
                "issue_type": "poor_calibration",
                "severity": "warning",
                "description": f"Brier score is {brier:.3f}, indicating poorly calibrated output probabilities.",
                "affected_components": [model_name],
                "remediation": "Apply Isotonic or Platt scaling calibration on validation predictions."
            })

    # Overall Status
    audit_status = "PASSED"
    if any(f["severity"] == "critical" for f in findings):
        audit_status = "CRITICAL_ISSUES_FOUND"
    elif len(findings) > 0:
        audit_status = "WARNINGS_IDENTIFIED"

    return {
        "audit_status": audit_status,
        "requires_iteration": requires_iteration,
        "findings": findings,
        "remediation_actions": list(set(remediation_actions)),
        "model_audited": model_name,
    }
