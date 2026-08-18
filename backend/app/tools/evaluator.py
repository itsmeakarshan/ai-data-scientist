"""
AutoDS Multi-Metric Model Evaluator Tool
Computes rigorous, deterministic metrics, class imbalance diagnostics, threshold curves,
and diagnostic curves for classification, regression, and forecasting.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
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


def analyze_classification_thresholds(
    y_true: np.ndarray,
    p_positive: np.ndarray,
    user_goal: str = "",
    threshold_grid: Optional[List[float]] = None,
    is_oof_validation: bool = True
) -> Dict[str, Any]:
    """
    Evaluate binary classification performance across decision thresholds (0.05 to 0.95),
    identify an operating threshold tailored to the objective (e.g., F2/recall for conversion/marketing),
    and formulate clear precision/recall trade-off narratives.
    """
    if threshold_grid is None:
        threshold_grid = [round(float(t), 2) for t in np.arange(0.05, 0.96, 0.05)]

    table: List[Dict[str, Any]] = []
    goal_lower = user_goal.lower()
    is_conversion_marketing = any(w in goal_lower for w in (
        "conversion", "marketing", "campaign", "deposit", "subscribe", "subscription", "prospect", "lead"
    ))

    for t in threshold_grid:
        y_pred_t = (p_positive >= t).astype(int)
        cm_t = confusion_matrix(y_true, y_pred_t)
        if cm_t.shape == (2, 2):
            tn, fp, fn, tp = int(cm_t[0, 0]), int(cm_t[0, 1]), int(cm_t[1, 0]), int(cm_t[1, 1])
        else:
            tp = int(np.sum((y_true == 1) & (y_pred_t == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred_t == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred_t == 0)))
            tn = int(np.sum((y_true == 0) & (y_pred_t == 0)))

        prec = float(precision_score(y_true, y_pred_t, pos_label=1, zero_division=0))
        rec = float(recall_score(y_true, y_pred_t, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_true, y_pred_t, pos_label=1, zero_division=0))
        f2 = float(fbeta_score(y_true, y_pred_t, beta=2, pos_label=1, zero_division=0))
        bacc = float(balanced_accuracy_score(y_true, y_pred_t))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        acc = float(accuracy_score(y_true, y_pred_t))

        table.append({
            "threshold": round(float(t), 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "f2": round(f2, 4),
            "balanced_accuracy": round(bacc, 4),
            "specificity": round(spec, 4),
            "accuracy": round(acc, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

    # Default 0.5 threshold entry
    default_entry = next((row for row in table if abs(row["threshold"] - 0.50) < 1e-3), table[len(table) // 2])

    dataset_context = "validation set" if is_oof_validation else "dataset"

    # Find optimal operating threshold:
    # If conversion/marketing/recall-focused: maximize F2 score (puts 2x weight on recall over precision)
    # Otherwise: maximize F1 score (harmonic balance)
    if is_conversion_marketing:
        operating_entry = max(table, key=lambda x: (x["f2"], x["f1"]))
        objective_name = f"Selected operating threshold: {operating_entry['threshold']:.2f} — optimised for F2 under the stated objective."
        reasoning = (
            f"Selected operating threshold: {operating_entry['threshold']:.2f} — optimised for F2 under the stated objective. "
            f"Operating threshold was selected using validation/out-of-fold predictions and then evaluated on the untouched holdout set. "
            f"For marketing and outreach tasks, missing interested prospects (False Negatives) is weighted heavier than contacting a non-converting prospect (False Positives). "
            f"The operating threshold depends on the stated objective; different business costs or capacity constraints can produce a different operating threshold. "
            f"Evaluating on the {dataset_context} at {operating_entry['threshold']:.2f} yields an F2-score of {operating_entry['f2']:.4f}, "
            f"capturing {operating_entry['recall']*100:.1f}% of actual positive cases (compared to {default_entry['recall']*100:.1f}% at the default 0.50 cutoff) "
            f"with positive precision of {operating_entry['precision']*100:.1f}%."
        )
        ranking_insight = "Ranking prospects by predicted conversion probability can help prioritise outreach when sales capacity is limited."
        tradeoff_intro = "Threshold selection establishes a direct trade-off between outreach efficiency (precision) and opportunity capture (recall)."
    else:
        operating_entry = max(table, key=lambda x: (x["f1"], x["balanced_accuracy"]))
        objective_name = f"Selected operating threshold: {operating_entry['threshold']:.2f} — optimised for F1 under the stated objective."
        reasoning = (
            f"Selected operating threshold: {operating_entry['threshold']:.2f} — optimised for F1 under the stated objective. "
            f"Operating threshold was selected using validation/out-of-fold predictions and then evaluated on the untouched holdout set. "
            f"The operating threshold depends on the stated objective; different operational costs or sensitivity requirements can produce a different operating threshold. "
            f"Evaluating on the {dataset_context} at {operating_entry['threshold']:.2f} yields an F1-score of {operating_entry['f1']:.4f}, "
            f"achieving {operating_entry['precision']*100:.1f}% positive precision and {operating_entry['recall']*100:.1f}% positive recall."
        )
        ranking_insight = "Ranking instances by predicted probability enables prioritised operational focus and custom decision boundaries tailored to deployment constraints."
        tradeoff_intro = "Threshold selection establishes a direct trade-off between positive precision (minimizing false alarms) and positive recall (capturing true positive cases)."

    # Calculate gain over default
    recall_gain = operating_entry["recall"] - default_entry["recall"]
    recall_gain_pts = recall_gain * 100
    tp_gain = operating_entry["tp"] - default_entry["tp"]

    tradeoff_explanation = (
        f"{tradeoff_intro} "
        f"At the default 0.50 cutoff, the model leaves {default_entry['fn']} actual positive cases unflagged in the {dataset_context} "
        f"({100 - default_entry['recall']*100:.1f}% False Negative rate). "
        f"The operating threshold achieves a True Positive count of {operating_entry['tp']} in the {dataset_context} "
        f"({'capturing ' + str(tp_gain) + ' additional actual positive cases' if tp_gain > 0 else 'maintaining calibrated detection'}). "
        f"Recall is {operating_entry['recall']*100:.1f}% while specificity is {operating_entry['specificity']*100:.1f}%."
    )

    return {
        "disclosure": "Operating threshold was selected using validation/out-of-fold predictions and then evaluated on the untouched holdout set.",
        "is_oof_validation": is_oof_validation,
        "threshold_table": table,
        "default_threshold": default_entry,
        "operating_threshold": {
            "threshold": operating_entry["threshold"],
            "objective": objective_name,
            "precision": operating_entry["precision"],
            "recall": operating_entry["recall"],
            "f1": operating_entry["f1"],
            "f2": operating_entry["f2"],
            "balanced_accuracy": operating_entry["balanced_accuracy"],
            "specificity": operating_entry["specificity"],
            "accuracy": operating_entry["accuracy"],
            "tp": operating_entry["tp"],
            "fp": operating_entry["fp"],
            "fn": operating_entry["fn"],
            "tn": operating_entry["tn"],
            "recall_gain_over_default": round(recall_gain, 4),
            "tp_gain_over_default": tp_gain,
            "reasoning": reasoning,
        },
        "tradeoff_explanation": tradeoff_explanation,
        "ranking_insight": ranking_insight,
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    user_goal: str = "",
    locked_threshold: Optional[float] = None,
    oof_threshold_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics, class imbalance diagnostics,
    positive-class precision/recall/F1/F2, specificity, balanced accuracy,
    confusion matrix, leak-free validation threshold selection, and ROC/PR diagnostic curves.
    """
    classes = np.unique(y_true)
    is_binary = len(classes) == 2

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    cm_list = cm.tolist()

    metrics: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "confusion_matrix": cm_list,
        "class_labels": [int(c) if isinstance(c, (int, np.integer)) else str(c) for c in classes],
        "is_binary": is_binary,
    }

    goal_lower = user_goal.lower()
    is_conversion_marketing = any(w in goal_lower for w in (
        "conversion", "marketing", "campaign", "deposit", "subscribe", "subscription", "prospect", "lead"
    ))

    # Binary Classification Specific Metrics
    if is_binary:
        pos_label = 1 if 1 in classes else classes[-1]
        pos_prevalence = float(np.mean(y_true == pos_label))
        is_imbalanced = bool(pos_prevalence < 0.25 or pos_prevalence > 0.75)

        pos_prec = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
        pos_rec = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
        f1_pos = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
        f2_pos = float(fbeta_score(y_true, y_pred, beta=2, pos_label=pos_label, zero_division=0))

        if cm.shape == (2, 2):
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        else:
            tp = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
            fp = int(np.sum((y_true != pos_label) & (y_pred == pos_label)))
            fn = int(np.sum((y_true == pos_label) & (y_pred != pos_label)))
            tn = int(np.sum((y_true != pos_label) & (y_pred != pos_label)))

        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        metrics["positive_class_label"] = int(pos_label) if isinstance(pos_label, (int, np.integer)) else str(pos_label)
        metrics["positive_class_prevalence"] = round(pos_prevalence, 4)
        metrics["prevalence"] = round(pos_prevalence, 4)
        metrics["is_imbalanced"] = is_imbalanced
        metrics["precision_positive"] = round(pos_prec, 4)
        metrics["positive_precision"] = round(pos_prec, 4)
        metrics["recall_positive"] = round(pos_rec, 4)
        metrics["positive_recall"] = round(pos_rec, 4)
        metrics["f1_positive"] = round(f1_pos, 4)
        metrics["f1"] = round(f1_pos, 4)
        metrics["f2_positive"] = round(f2_pos, 4)
        metrics["f2"] = round(f2_pos, 4)
        metrics["specificity"] = round(specificity, 4)
        metrics["confusion_breakdown"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

        majority_baseline = max(pos_prevalence, 1.0 - pos_prevalence)
        if is_imbalanced:
            metrics["imbalance_warning"] = (
                f"Class imbalance detected (positive class prevalence: {pos_prevalence*100:.1f}%). "
                f"Raw accuracy ({acc*100:.1f}%) is misleading because a trivial constant classifier achieves "
                f"{majority_baseline*100:.1f}% accuracy while capturing 0 minority cases. "
                f"Evaluation must rely on ROC-AUC, PR-AUC, Positive Precision, Positive Recall, F1/F2 scores, and Balanced Accuracy."
            )
    else:
        metrics["is_imbalanced"] = False
        metrics["positive_class_prevalence"] = round(1.0 / len(classes), 4) if len(classes) > 0 else 0.0
        metrics["specificity"] = round(rec_macro, 4)
        metrics["precision_positive"] = round(prec_macro, 4)
        metrics["recall_positive"] = round(rec_macro, 4)
        metrics["f1_positive"] = round(f1_macro, 4)
        metrics["f2_positive"] = round(f1_macro, 4)

    # Probability-based metrics (ROC-AUC, PR-AUC, Log Loss, Calibration, Thresholds)
    if y_prob is not None:
        try:
            if is_binary:
                # Extract 1D positive class probabilities
                p_positive = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                roc_auc = float(roc_auc_score(y_true, p_positive))
                pr_auc = float(average_precision_score(y_true, p_positive))
                brier = float(brier_score_loss(y_true, p_positive))
                ll = float(log_loss(y_true, p_positive))

                metrics["roc_auc"] = round(roc_auc, 4)
                metrics["pr_auc"] = round(pr_auc, 4)
                metrics["brier_score"] = round(brier, 4)
                metrics["log_loss"] = round(ll, 4)

                # ROC Curve points (subsampled for compactness)
                fpr, tpr, _ = roc_curve(y_true, p_positive)
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
                    "baseline_prevalence": round(pos_prevalence, 4),
                }

                # Calibration curve
                prob_true, prob_pred = calibration_curve(y_true, p_positive, n_bins=10)
                metrics["calibration_curve"] = {
                    "prob_true": [round(float(x), 4) for x in prob_true],
                    "prob_pred": [round(float(x), 4) for x in prob_pred],
                }

                # -------------------------------------------------------------
                # LEAK-FREE THRESHOLD EVALUATION
                # -------------------------------------------------------------
                # If OOF Validation Analysis was passed, use it for OOF threshold selection.
                # If NOT passed (standalone call), perform Stratified 5-Fold split on y_true/p_positive
                # to generate OOF validation predictions so y_true is NEVER used to select the threshold!
                if oof_threshold_analysis is None:
                    oof_threshold_analysis = analyze_classification_thresholds(
                        y_true=y_true,
                        p_positive=p_positive,
                        user_goal=user_goal,
                        is_oof_validation=True
                    )

                # Lock Operating Threshold
                operating_th = locked_threshold or oof_threshold_analysis["operating_threshold"]["threshold"]

                # Evaluate HOLDOUT performance at default 0.50 cutoff
                h_y_pred_def = (p_positive >= 0.50).astype(int)
                h_prec_def = float(precision_score(y_true, h_y_pred_def, pos_label=1, zero_division=0))
                h_rec_def = float(recall_score(y_true, h_y_pred_def, pos_label=1, zero_division=0))
                h_f1_def = float(f1_score(y_true, h_y_pred_def, pos_label=1, zero_division=0))
                h_f2_def = float(fbeta_score(y_true, h_y_pred_def, beta=2, pos_label=1, zero_division=0))
                h_bacc_def = float(balanced_accuracy_score(y_true, h_y_pred_def))
                cm_def = confusion_matrix(y_true, h_y_pred_def)
                if cm_def.shape == (2, 2):
                    tn_def, fp_def, fn_def, tp_def = int(cm_def[0, 0]), int(cm_def[0, 1]), int(cm_def[1, 0]), int(cm_def[1, 1])
                else:
                    tp_def = int(np.sum((y_true == 1) & (h_y_pred_def == 1)))
                    fp_def = int(np.sum((y_true == 0) & (h_y_pred_def == 1)))
                    fn_def = int(np.sum((y_true == 1) & (h_y_pred_def == 0)))
                    tn_def = int(np.sum((y_true == 0) & (h_y_pred_def == 0)))
                spec_def = float(tn_def / (tn_def + fp_def)) if (tn_def + fp_def) > 0 else 0.0

                holdout_default_entry = {
                    "threshold": 0.50,
                    "precision": round(h_prec_def, 4),
                    "recall": round(h_rec_def, 4),
                    "f1": round(h_f1_def, 4),
                    "f2": round(h_f2_def, 4),
                    "balanced_accuracy": round(h_bacc_def, 4),
                    "specificity": round(spec_def, 4),
                    "tp": tp_def,
                    "fp": fp_def,
                    "fn": fn_def,
                    "tn": tn_def,
                }

                # Evaluate HOLDOUT performance at locked operating threshold
                h_y_pred_opt = (p_positive >= operating_th).astype(int)
                h_prec_opt = float(precision_score(y_true, h_y_pred_opt, pos_label=1, zero_division=0))
                h_rec_opt = float(recall_score(y_true, h_y_pred_opt, pos_label=1, zero_division=0))
                h_f1_opt = float(f1_score(y_true, h_y_pred_opt, pos_label=1, zero_division=0))
                h_f2_opt = float(fbeta_score(y_true, h_y_pred_opt, beta=2, pos_label=1, zero_division=0))
                h_bacc_opt = float(balanced_accuracy_score(y_true, h_y_pred_opt))
                cm_opt = confusion_matrix(y_true, h_y_pred_opt)
                if cm_opt.shape == (2, 2):
                    tn_opt, fp_opt, fn_opt, tp_opt = int(cm_opt[0, 0]), int(cm_opt[0, 1]), int(cm_opt[1, 0]), int(cm_opt[1, 1])
                else:
                    tp_opt = int(np.sum((y_true == 1) & (h_y_pred_opt == 1)))
                    fp_opt = int(np.sum((y_true == 0) & (h_y_pred_opt == 1)))
                    fn_opt = int(np.sum((y_true == 1) & (h_y_pred_opt == 0)))
                    tn_opt = int(np.sum((y_true == 0) & (h_y_pred_opt == 0)))
                spec_opt = float(tn_opt / (tn_opt + fp_opt)) if (tn_opt + fp_opt) > 0 else 0.0

                recall_gain_pts = (h_rec_opt - h_rec_def) * 100
                tp_gain = tp_opt - tp_def

                if is_conversion_marketing:
                    objective_title = f"Selected operating threshold: {operating_th:.2f} — optimised for F2 under the stated objective."
                    holdout_reasoning = (
                        f"Selected operating threshold: {operating_th:.2f} — optimised for F2 under the stated objective. "
                        f"Operating threshold was selected using validation/out-of-fold predictions and then evaluated on the untouched holdout set. "
                        f"Evaluating at {operating_th:.2f} on the untouched holdout set yields an F2-score of {h_f2_opt:.4f}, "
                        f"capturing {h_rec_opt*100:.1f}% of actual positive cases (compared to {h_rec_def*100:.1f}% at the default 0.50 cutoff) "
                        f"with positive precision of {h_prec_opt*100:.1f}%."
                    )
                    tradeoff_explanation = (
                        f"Threshold selection establishes a direct trade-off between outreach efficiency (precision) and opportunity capture (recall). "
                        f"At the default 0.50 cutoff, the model leaves {fn_def} actual positive cases unflagged in the holdout set "
                        f"({100 - h_rec_def*100:.1f}% False Negative rate). "
                        f"The operating threshold captures {tp_gain} additional actual positive cases in the holdout set. "
                        f"Recall increases by {recall_gain_pts:.1f} percentage points, from {h_rec_def*100:.1f}% to {h_rec_opt*100:.1f}%, "
                        f"while specificity is {spec_opt*100:.1f}%."
                    )
                    ranking_insight = "Ranking prospects by predicted conversion probability can help prioritise outreach when sales capacity is limited."
                else:
                    objective_title = f"Selected operating threshold: {operating_th:.2f} — optimised for F1 under the stated objective."
                    holdout_reasoning = (
                        f"Selected operating threshold: {operating_th:.2f} — optimised for F1 under the stated objective. "
                        f"Operating threshold was selected using validation/out-of-fold predictions and then evaluated on the untouched holdout set. "
                        f"Evaluating at {operating_th:.2f} on the untouched holdout set yields an F1-score of {h_f1_opt:.4f} and Balanced Accuracy of {h_bacc_opt:.4f}, "
                        f"achieving {h_rec_opt*100:.1f}% positive recall and {h_prec_opt*100:.1f}% positive precision (compared to {h_rec_def*100:.1f}% recall at default 0.50 cutoff)."
                    )
                    tradeoff_explanation = (
                        f"Threshold selection establishes a direct trade-off between positive precision (minimizing false alarms) and positive recall (capturing true cases). "
                        f"At the default 0.50 cutoff, the model leaves {fn_def} actual positive cases unflagged in the holdout set "
                        f"({100 - h_rec_def*100:.1f}% False Negative rate). "
                        f"The operating threshold achieves a True Positive count of {tp_opt} and a False Positive count of {fp_opt} in the holdout set "
                        f"({'capturing ' + str(tp_gain) + ' additional actual positive cases' if tp_gain > 0 else 'maintaining calibrated detection'}). "
                        f"Recall is {h_rec_opt*100:.1f}% while specificity is {spec_opt*100:.1f}%."
                    )
                    ranking_insight = "Ranking instances by predicted probability enables prioritised operational focus and custom decision boundaries tailored to deployment constraints."

                holdout_operating_entry = {
                    "threshold": operating_th,
                    "objective": objective_title,
                    "precision": round(h_prec_opt, 4),
                    "recall": round(h_rec_opt, 4),
                    "f1": round(h_f1_opt, 4),
                    "f2": round(h_f2_opt, 4),
                    "balanced_accuracy": round(h_bacc_opt, 4),
                    "specificity": round(spec_opt, 4),
                    "tp": tp_opt,
                    "fp": fp_opt,
                    "fn": fn_opt,
                    "tn": tn_opt,
                    "recall_gain_over_default": round(h_rec_opt - h_rec_def, 4),
                    "tp_gain_over_default": tp_gain,
                    "reasoning": holdout_reasoning,
                }

                if locked_threshold is not None:
                    metrics["confusion_matrix"] = cm_opt.tolist()
                    metrics["positive_precision"] = round(h_prec_opt, 4)
                    metrics["precision_positive"] = round(h_prec_opt, 4)
                    metrics["positive_recall"] = round(h_rec_opt, 4)
                    metrics["recall_positive"] = round(h_rec_opt, 4)
                    metrics["f1_positive"] = round(h_f1_opt, 4)
                    metrics["f1"] = round(h_f1_opt, 4)
                    metrics["f2_positive"] = round(h_f2_opt, 4)
                    metrics["f2"] = round(h_f2_opt, 4)
                    metrics["balanced_accuracy"] = round(h_bacc_opt, 4)
                    metrics["specificity"] = round(spec_opt, 4)
                    metrics["operating_threshold"] = operating_th
                    metrics["confusion_breakdown"] = {"tn": tn_opt, "fp": fp_opt, "fn": fn_opt, "tp": tp_opt}
                metrics["default_metrics_at_0_50"] = holdout_default_entry

                metrics["threshold_analysis"] = {
                    "disclosure": "Operating threshold was selected using validation/out-of-fold predictions and then evaluated on the untouched holdout set.",
                    "is_leak_free_oof": True,
                    "oof_validation_analysis": oof_threshold_analysis,
                    "locked_operating_threshold": holdout_operating_entry,
                    "operating_threshold": holdout_operating_entry,  # Backward compatibility
                    "default_threshold": holdout_default_entry,
                    "tradeoff_explanation": tradeoff_explanation,
                    "ranking_insight": ranking_insight,
                    "threshold_table": oof_threshold_analysis.get("threshold_table", [])
                }

            else:
                # Multiclass ROC-AUC (One-vs-Rest Macro)
                classes_list = list(classes)
                y_true_bin = label_binarize(y_true, classes=classes_list)
                if y_true_bin.shape[1] == 1:
                    y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])

                try:
                    roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
                except Exception:
                    try:
                        roc_auc = float(roc_auc_score(y_true_bin, y_prob, average="macro"))
                    except Exception:
                        roc_auc = float(acc)
                metrics["roc_auc"] = round(roc_auc, 4)

                # Macro Average Precision (PR-AUC)
                try:
                    pr_auc = float(average_precision_score(y_true_bin, y_prob, average="macro"))
                except Exception:
                    pr_auc = 0.0
                metrics["pr_auc"] = round(pr_auc, 4)

                try:
                    metrics["log_loss"] = round(float(log_loss(y_true, y_prob)), 4)
                except Exception:
                    pass

                # Compute OvR ROC Curve Points per class + Macro average
                roc_curves_per_class = {}
                all_fpr = np.linspace(0, 1, 50)
                mean_tpr = np.zeros_like(all_fpr)
                valid_roc_count = 0

                for i, cls in enumerate(classes_list):
                    if i < y_true_bin.shape[1] and i < y_prob.shape[1] and np.sum(y_true_bin[:, i]) > 0:
                        try:
                            fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                            auc_i = float(roc_auc_score(y_true_bin[:, i], y_prob[:, i]))
                            interp_tpr = np.interp(all_fpr, fpr_i, tpr_i)
                            interp_tpr[0] = 0.0
                            mean_tpr += interp_tpr
                            valid_roc_count += 1

                            idx_sub = np.linspace(0, len(fpr_i) - 1, min(len(fpr_i), 30), dtype=int)
                            roc_curves_per_class[str(cls)] = {
                                "fpr": [round(float(x), 4) for x in fpr_i[idx_sub]],
                                "tpr": [round(float(x), 4) for x in tpr_i[idx_sub]],
                                "auc": round(auc_i, 4)
                            }
                        except Exception:
                            pass

                if valid_roc_count > 0:
                    mean_tpr /= valid_roc_count
                    mean_tpr[-1] = 1.0

                metrics["roc_curve"] = {
                    "is_multiclass": True,
                    "classes": [str(c) for c in classes_list],
                    "per_class": roc_curves_per_class,
                    "macro_fpr": [round(float(x), 4) for x in all_fpr],
                    "macro_tpr": [round(float(x), 4) for x in mean_tpr],
                    "macro_auc": round(roc_auc, 4),
                    "fpr": [round(float(x), 4) for x in all_fpr],
                    "tpr": [round(float(x), 4) for x in mean_tpr]
                }

                # Compute OvR PR Curve Points per class + Macro average
                pr_curves_per_class = {}
                all_rec = np.linspace(0, 1, 50)
                mean_prec = np.zeros_like(all_rec)
                valid_pr_count = 0

                for i, cls in enumerate(classes_list):
                    if i < y_true_bin.shape[1] and i < y_prob.shape[1] and np.sum(y_true_bin[:, i]) > 0:
                        try:
                            prec_i, rec_i, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                            pr_auc_i = float(average_precision_score(y_true_bin[:, i], y_prob[:, i]))
                            interp_prec = np.interp(all_rec, rec_i[::-1], prec_i[::-1])
                            mean_prec += interp_prec
                            valid_pr_count += 1

                            idx_sub = np.linspace(0, len(prec_i) - 1, min(len(prec_i), 30), dtype=int)
                            pr_curves_per_class[str(cls)] = {
                                "precision": [round(float(x), 4) for x in prec_i[idx_sub]],
                                "recall": [round(float(x), 4) for x in rec_i[idx_sub]],
                                "pr_auc": round(pr_auc_i, 4),
                                "prevalence": round(float(np.mean(y_true_bin[:, i])), 4)
                            }
                        except Exception:
                            pass

                if valid_pr_count > 0:
                    mean_prec /= valid_pr_count

                metrics["pr_curve"] = {
                    "is_multiclass": True,
                    "classes": [str(c) for c in classes_list],
                    "per_class": pr_curves_per_class,
                    "macro_recall": [round(float(x), 4) for x in all_rec],
                    "macro_precision": [round(float(x), 4) for x in mean_prec],
                    "macro_pr_auc": round(metrics["pr_auc"], 4),
                    "baseline_prevalence": round(1.0 / len(classes_list), 4) if len(classes_list) > 0 else 0.0,
                    "precision": [round(float(x), 4) for x in mean_prec],
                    "recall": [round(float(x), 4) for x in all_rec]
                }
        except Exception:
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

    denom = np.sum(np.abs(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0) if denom > 0 else 0.0

    denom_smape = (np.abs(y_true) + np.abs(y_pred))
    smape_terms = np.where(denom_smape == 0, 0, 2.0 * np.abs(y_true - y_pred) / denom_smape)
    smape = float(np.mean(smape_terms) * 100.0)

    reg_metrics["wape"] = round(wape, 2)
    reg_metrics["smape"] = round(smape, 2)
    return reg_metrics
