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
    raw_columns: Optional[List[str]] = None,
    remediated_features: Optional[List[str]] = None,
    leakage_explanations: Optional[Dict[str, str]] = None
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

    # 1. Audit Overfitting (Train vs Validation / Test divergence)
    if problem_type == "classification":
        train_score = train_metrics.get("roc_auc", train_metrics.get("accuracy", 0.0))
        eval_score = test_metrics.get("roc_auc", test_metrics.get("accuracy", 0.0))
        eval_context = "Test"
        if not test_metrics and cv_mean > 0:
            eval_score = cv_mean
            eval_context = "Validation CV"
            
        divergence = train_score - eval_score

        if train_score >= 0.98 and eval_score < 0.82 and eval_score > 0:
            findings.append({
                "issue_type": "severe_overfitting",
                "severity": "critical",
                "description": f"Extreme generalization gap ({divergence:.3f}). Model memorized training set (Train ROC-AUC: {train_score:.3f}, {eval_context} ROC-AUC: {eval_score:.3f}).",
                "affected_components": [model_name],
                "remediation": "Regularize model (limit tree depth, increase min_child_samples, add L1/L2 penalties)."
            })
            requires_iteration = True
            remediation_actions.append("REGULARIZE_MODEL")
        elif divergence > 0.15 and eval_score > 0:
            findings.append({
                "issue_type": "moderate_overfitting",
                "severity": "warning",
                "description": f"Noticeable train-validation gap ({divergence:.3f}). Train score: {train_score:.3f}, {eval_context} score: {eval_score:.3f}.",
                "affected_components": [model_name],
                "remediation": "Consider early stopping or hyperparameter tuning to improve validation generalization."
            })

    elif problem_type in ("regression", "forecasting"):
        train_rmse = train_metrics.get("rmse", 0.0)
        test_rmse = test_metrics.get("rmse", 0.0)
        eval_rmse = test_rmse if test_metrics else (cv_mean if cv_mean > 0 else 0.0)
        if train_rmse > 0 and eval_rmse > 0 and (eval_rmse / train_rmse) > 1.8:
            findings.append({
                "issue_type": "severe_overfitting",
                "severity": "critical",
                "description": f"Evaluation RMSE ({eval_rmse:.2f}) is {eval_rmse/train_rmse:.1f}x higher than Train RMSE ({train_rmse:.2f}).",
                "affected_components": [model_name],
                "remediation": "Apply L2 shrinkage (Ridge), prune tree estimators, or reduce feature space."
            })
            requires_iteration = True
            remediation_actions.append("REGULARIZE_MODEL")

    # 2. Audit Potential Data / Target Leakage
    prospective_leakage_indicators = [
        "casual", "registered", "duration", "post_event", "post_call",
        "after_outcome", "future_val", "future_outcome", "post_sale", "post_conversion"
    ]
    if raw_columns:
        for col in raw_columns:
            if col == target_column:
                continue
            col_lower = col.lower()
            if any(ind == col_lower or (ind in col_lower and not any(leg in col_lower for leg in ["lag", "roll", "cal"])) for ind in prospective_leakage_indicators):
                in_features = any(col_lower == str(f).lower() or col_lower in str(f).lower() for f in feature_names)
                if in_features:
                    findings.append({
                        "issue_type": "domain_target_leakage",
                        "severity": "critical",
                        "description": f"Feature '{col}' represents prospective/target-component information unknown at prediction time, creating deployment-time data leakage.",
                        "affected_components": [col],
                        "remediation": f"Drop '{col}' feature and retrain model suite to obtain honest prospective performance."
                    })
                    requires_iteration = True
                    remediation_actions.append("REMOVE_LEAKY_FEATURES")

    # Record remediation findings if leaky features were safely excluded
    leakage_remediated = bool(remediated_features and len(remediated_features) > 0)
    if leakage_remediated:
        for rf in (remediated_features or []):
            expl = (leakage_explanations or {}).get(
                rf,
                f"Feature '{rf}' is a component of target '{target_column}' and was excluded prior to model training."
            )
            findings.append({
                "issue_type": "target_component_leakage_remediated",
                "severity": "info",
                "description": f"Leakage prevention excluded '{rf}': {expl}",
                "affected_components": [rf],
                "remediation": f"Excluded from feature matrix before model training to ensure prediction-time validity."
            })

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

    # 5. Class Imbalance & Metric Hazard Audit
    if problem_type == "classification" and test_metrics.get("is_imbalanced"):
        prev = test_metrics.get("positive_class_prevalence", test_metrics.get("prevalence", 0.0))
        acc = test_metrics.get("accuracy", 0.0)
        rec = test_metrics.get("positive_recall", test_metrics.get("recall_positive", 0.0))
        f1 = test_metrics.get("f1_positive", test_metrics.get("f1", 0.0))
        
        findings.append({
            "issue_type": "class_imbalance_accuracy_hazard",
            "severity": "warning",
            "description": f"Target exhibits class imbalance (positive prevalence: {prev*100:.1f}%). While raw accuracy is {acc*100:.1f}%, default cutoff captures only {rec*100:.1f}% of positive cases (F1: {f1:.4f}). Accuracy is misleading for operational decision-making.",
            "affected_components": ["evaluation_metrics", model_name],
            "remediation": "Optimize decision threshold and prioritize PR-AUC, ROC-AUC, F1, and F2 scores over accuracy."
        })

    # Overall Status Determination
    if any(f["severity"] == "critical" for f in findings):
        audit_status = "CRITICAL_ISSUES_FOUND"
    elif leakage_remediated:
        audit_status = "PASSED (Remediated)"
    elif any(f["severity"] == "warning" for f in findings):
        audit_status = "WARNINGS_IDENTIFIED"
    else:
        audit_status = "PASSED"

    return {
        "audit_status": audit_status,
        "requires_iteration": requires_iteration,
        "findings": findings,
        "remediation_actions": list(set(remediation_actions)),
        "remediated_features": remediated_features or [],
        "leakage_remediated": leakage_remediated,
        "model_audited": model_name,
    }
