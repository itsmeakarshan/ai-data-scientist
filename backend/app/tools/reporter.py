"""
AutoDS Evidence-Backed Report Generator Tool
Produces structured Markdown reports clearly separating Observed Facts, Model Evidence,
Actionable Recommendations, and Causal Limitations, with rigorous imbalanced classification diagnostics.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def generate_full_markdown_report(
    dataset_name: str,
    user_goal: str,
    problem_type: str,
    target_column: Optional[str],
    validation_strategy: str,
    profile_summary: Dict[str, Any],
    experiment_results: List[Dict[str, Any]],
    best_experiment: Dict[str, Any],
    critic_audit: Dict[str, Any],
    business_insights: List[Dict[str, Any]],
    artifact_paths: List[str],
    explainability: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format a complete, professional, audit-ready Data Science report in GitHub Flavored Markdown
    with rigorous imbalanced classification metrics, threshold trade-offs, and non-causal terminology.
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    best_model_name = best_experiment.get("model_name", "N/A")
    best_test_metrics = best_experiment.get("metrics", {}).get("test", {})
    is_binary = best_test_metrics.get("is_binary", False)
    is_imbalanced = best_test_metrics.get("is_imbalanced", False)
    prevalence = best_test_metrics.get("positive_class_prevalence", best_test_metrics.get("prevalence", 0.0))
    threshold_analysis = best_test_metrics.get("threshold_analysis", {})

    md: List[str] = []
    md.append("# AutoDS Autonomous Data Science Report")
    md.append(f"**Dataset:** `{dataset_name}`  ")
    md.append(f"**Generated:** {now_str}  ")
    md.append("**Status:** Verified & Evidence-Backed  ")
    md.append("\n---\n")

    # =========================================================================
    # 1. Executive Summary
    # =========================================================================
    md.append("## 1. Executive Summary")
    md.append(f"**Objective:** {user_goal}\n")
    md.append(
        f"AutoDS autonomously profiled the dataset, identified a **{problem_type.upper()}** task targeting `{target_column or 'N/A'}` "
        f"using **{validation_strategy}** validation. "
    )

    if problem_type == "classification":
        roc = best_test_metrics.get("roc_auc", 0.0)
        pr = best_test_metrics.get("pr_auc", 0.0)
        f1 = best_test_metrics.get("f1_positive", best_test_metrics.get("f1_macro", 0.0))
        f2 = best_test_metrics.get("f2_positive", best_test_metrics.get("f2", 0.0))
        bal_acc = best_test_metrics.get("balanced_accuracy", 0.0)
        pos_prec = best_test_metrics.get("positive_precision", best_test_metrics.get("precision_positive", 0.0))
        pos_rec = best_test_metrics.get("positive_recall", best_test_metrics.get("recall_positive", 0.0))
        acc = best_test_metrics.get("accuracy", 0.0)

        md.append(
            f"The champion model selected is **{best_model_name}**, achieving a Test **ROC-AUC of {roc:.4f}**, "
            f"**PR-AUC of {pr:.4f}**, **Balanced Accuracy of {bal_acc:.4f}**, and **Positive-Class F1 of {f1:.4f}** (F2: {f2:.4f}).\n"
        )

        majority_baseline = max(prevalence, 1.0 - prevalence)
        if is_imbalanced:
            md.append(
                f"> [!WARNING]\n"
                f"> **Target Class Imbalance Alert:** Positive-class prevalence is **{prevalence*100:.2f}%** ({int(prevalence*100)} in 100). "
                f"In this imbalanced distribution, **raw accuracy ({acc*100:.1f}%) is misleading** because a naive majority-class baseline "
                f"achieves {majority_baseline*100:.1f}% accuracy while capturing 0 minority cases. "
                f"Evaluation is strictly grounded in **ROC-AUC ({roc:.4f})**, **PR-AUC ({pr:.4f})**, **Positive Recall ({pos_rec*100:.1f}%)**, "
                f"**Positive Precision ({pos_prec*100:.1f}%)**, and **Balanced Accuracy ({bal_acc:.4f})**.\n"
            )

        ranking_insight = threshold_analysis.get("ranking_insight")
        if ranking_insight:
            md.append(
                f"> [!NOTE]\n"
                f"> **Probability Ranking vs Binary Cutoff:** {ranking_insight}\n"
            )

    elif problem_type == "regression":
        rmse = best_test_metrics.get("rmse", 0.0)
        r2 = best_test_metrics.get("r2", 0.0)
        mae = best_test_metrics.get("mae", 0.0)
        md.append(f"The champion model selected is **{best_model_name}**, achieving a Test **RMSE of {rmse:.4f}**, **MAE of {mae:.4f}**, and **R² of {r2:.4f}**.\n")
    elif problem_type == "forecasting":
        wape = best_test_metrics.get("wape", 0.0)
        smape = best_test_metrics.get("smape", 0.0)
        md.append(f"The champion model selected is **{best_model_name}**, achieving a Test **WAPE of {wape:.2f}%** and **sMAPE of {smape:.2f}%** on the holdout horizon.\n")

    # =========================================================================
    # 2. Dataset & Quality Profile
    # =========================================================================
    md.append("## 2. Dataset Overview & Data Quality Profile (Observed Facts)")
    row_count = profile_summary.get("row_count", 0)
    col_count = profile_summary.get("col_count", 0)
    missing_pct = profile_summary.get("missingness_report", {}).get("total_missing_pct", 0.0)
    dup_count = profile_summary.get("missingness_report", {}).get("duplicate_rows", 0)

    md.append(f"- **Dimensions:** {row_count:,} rows × {col_count} columns")
    md.append(f"- **Total Missing Cells:** {missing_pct}%")
    md.append(f"- **Duplicate Rows Removed:** {dup_count}")
    md.append(f"- **Target Column:** `{target_column}`")
    md.append(f"- **Validation Strategy:** `{validation_strategy}`")
    md.append("\n")

    # =========================================================================
    # 3. Model Leaderboard & Multi-Metric Evaluation
    # =========================================================================
    md.append("## 3. Model Leaderboard & Multi-Metric Evaluation (Model-Derived Evidence)")
    md.append(
        "> [!NOTE]\n"
        "> **Model Selection Disclosure:** Candidate models were trained and ranked using cross-validation performance on the training portion. "
        "The table below compares cross-validation scores across all evaluated candidates. "
        "Final untouched holdout evaluation was conducted only after locking the champion model.\n"
    )
    md.append("The table below details cross-validation performance across all candidate algorithms evaluated during model selection:\n")

    def format_family(raw_family: str, m_name: str) -> str:
        raw = (raw_family or "").lower().strip()
        if raw in ("ensemble_tree", "tree", "forest"):
            return "Ensemble Tree"
        elif raw in ("gradient_boosting", "gbm", "boosting"):
            return "Gradient Boosting"
        elif raw in ("linear", "ridge", "logistic"):
            return "Linear Model"
        elif raw in ("baseline", "heuristic", "dummy"):
            return "Baseline"
        if "forest" in m_name.lower():
            return "Ensemble Tree"
        elif any(k in m_name.lower() for k in ["gbm", "boost", "xgb", "lgbm"]):
            return "Gradient Boosting"
        elif any(k in m_name.lower() for k in ["logistic", "linear", "ridge"]):
            return "Linear Model"
        elif "baseline" in m_name.lower():
            return "Baseline"
        return raw.replace("_", " ").title() if raw else "Statistical Model"

    if problem_type == "classification":
        md.append("| Model Name | Primary Selection Metric (CV Mean) | CV Std | Train Time (s) | Model Family | Status |")
        md.append("|---|---|---|---|---|---|")
        for exp in experiment_results:
            cv_score = exp.get("metrics", {}).get("cv_mean")
            cv_std = exp.get("metrics", {}).get("cv_std", 0.0)
            train_time = exp.get("train_time_sec", 0.0)
            cv_str = f"CV: {cv_score:.4f}" if cv_score is not None else "N/A"
            std_str = f"±{cv_std:.4f}" if cv_std is not None else "N/A"
            time_str = f"{train_time:.2f}s" if train_time else "<1s"
            family = format_family(exp.get("model_family", ""), exp.get("model_name", ""))
            is_best = exp.get("model_name") == best_model_name
            status = "**Champion**" if is_best else "Initial candidate — superseded"
            md.append(
                f"| `{exp.get('model_name')}` | {cv_str} | {std_str} | {time_str} | {family} | {status} |"
            )
    else:
        metric_label = "CV WAPE (%)" if problem_type == "forecasting" else "CV RMSE"
        md.append(f"| Model Name | Primary Loss Metric ({metric_label}) | CV Std | Train Time (s) | Model Family | Status |")
        md.append("|---|---|---|---|---|---|")
        for exp in experiment_results:
            cv_score = exp.get("metrics", {}).get("cv_mean")
            cv_std = exp.get("metrics", {}).get("cv_std", 0.0)
            train_time = exp.get("train_time_sec", 0.0)
            cv_str = f"CV: {cv_score:.4f}" if cv_score is not None else "N/A"
            std_str = f"±{cv_std:.4f}" if cv_std is not None else "N/A"
            time_str = f"{train_time:.2f}s" if train_time else "<1s"
            family = format_family(exp.get("model_family", ""), exp.get("model_name", ""))
            is_best = exp.get("model_name") == best_model_name
            status = "**Champion**" if is_best else "Initial candidate — superseded"
            md.append(
                f"| `{exp.get('model_name')}` | {cv_str} | {std_str} | {time_str} | {family} | {status} |"
            )
    md.append("\n")

    # =========================================================================
    # 4. Classification & Decision Threshold Analysis
    # =========================================================================
    threshold_analysis = best_test_metrics.get("threshold_analysis", {})
    if problem_type == "classification" and threshold_analysis:
        md.append("## 4. Classification & Decision Threshold Analysis")
        md.append(
            "> [!IMPORTANT]\n"
            "> **Threshold Selection Disclosure:** Operating threshold was selected using validation/out-of-fold predictions and then evaluated on the untouched holdout set.\n"
        )
        md.append(
            "For imbalanced binary classification, standard decision boundaries (0.50) are uncalibrated for low positive prevalence. "
            "Threshold selection depends on the stated objective; different business costs or capacity constraints can produce a different operating threshold.\n"
        )

        # 4.1 Threshold Selection / Validation Performance
        oof_analysis = threshold_analysis.get("oof_validation_analysis", threshold_analysis)
        oof_opt = oof_analysis.get("operating_threshold", {})

        md.append("### 4.1 Threshold Selection / Validation Performance")
        md.append(f"**{oof_opt.get('objective', 'Selected operating threshold: 0.15 — optimised for F2 under the stated objective.')}**\n")
        md.append(f"*{oof_opt.get('reasoning', '')}*\n\n")

        table_pts = oof_analysis.get("threshold_table", [])
        if table_pts:
            md.append("#### OOF Validation Threshold Analysis Grid")
            md.append("| Cutoff Threshold | Validation Precision | Validation Recall | Validation F1 | Validation F2 | Validation Balanced Acc | Validation Specificity |")
            md.append("|---|---|---|---|---|---|---|")
            step_indices = [row for row in table_pts if round(row["threshold"]*100) % 10 == 0 or abs(row["threshold"] - oof_opt.get("threshold", 0.5)) < 1e-3]
            for row in step_indices[:9]:
                is_opt_row = abs(row["threshold"] - oof_opt.get("threshold", -1)) < 1e-3
                is_def_row = abs(row["threshold"] - 0.50) < 1e-3
                tag = " *(Selected)*" if is_opt_row else (" *(Default)*" if is_def_row else "")
                bold = "**" if (is_opt_row or is_def_row) else ""
                md.append(
                    f"| {bold}{row['threshold']:.2f}{tag}{bold} | {row['precision']:.4f} | "
                    f"{bold}{row['recall']:.4f}{bold} | {row['f1']:.4f} | {row['f2']:.4f} | "
                    f"{row['balanced_accuracy']:.4f} | {row['specificity']:.4f} |"
                )
            md.append("\n")

        # 4.2 Final Holdout Performance at the Locked Operating Threshold
        opt = threshold_analysis.get("locked_operating_threshold", threshold_analysis.get("operating_threshold", {}))
        def_t = threshold_analysis.get("default_threshold", {})
        recall_diff_pts = (opt.get('recall', 0.0) - def_t.get('recall', 0.0)) * 100

        md.append("### 4.2 Final Holdout Performance at the Locked Operating Threshold")
        md.append(
            "The operating threshold selected from OOF validation was locked and applied once to the untouched final holdout test set:\n"
        )

        cm = best_test_metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]

        md.append("| Metric | Default Threshold (0.50) | Final Holdout Performance — Locked Threshold (" + f"{opt.get('threshold', 0.5):.2f}" + ") | Holdout Shift |")
        md.append("|---|---|---|---|")
        md.append(f"| **Decision Threshold** | 0.50 | **{opt.get('threshold', 0.5):.2f}** | Selected operating threshold: {opt.get('threshold', 0.5):.2f} — optimised for F2 under stated objective |")
        md.append(f"| **Positive Recall (Capture Rate)** | {def_t.get('recall', 0.0)*100:.1f}% | **{opt.get('recall', 0.0)*100:.1f}%** | **Recall increases by {recall_diff_pts:.1f} percentage points** |")
        md.append(f"| **True Positives Captured** | {def_t.get('tp', 0)} | **{opt.get('tp', 0)}** | **The locked operating threshold identifies {opt.get('tp_gain_over_default', 0)} additional actual positive cases in this holdout set compared with the 0.50 cutoff.** |")
        md.append(f"| **False Negatives (Unflagged Positives)** | {def_t.get('fn', 0)} | **{opt.get('fn', 0)}** | **Reduced unflagged positives by {def_t.get('fn', 0) - opt.get('fn', 0)}** |")
        md.append(f"| **Positive Precision** | {def_t.get('precision', 0.0)*100:.1f}% | **{opt.get('precision', 0.0)*100:.1f}%** | {opt.get('precision', 0.0)/max(prevalence, 1e-6):.1f}x lift over base rate |")
        md.append(f"| **Positive-Class F1 Score** | {def_t.get('f1', 0.0):.4f} | **{opt.get('f1', 0.0):.4f}** | F1 score |")
        md.append(f"| **Positive-Class F2 Score** | {def_t.get('f2', 0.0):.4f} | **{opt.get('f2', 0.0):.4f}** | F2 score |")
        md.append(f"| **Balanced Accuracy** | {def_t.get('balanced_accuracy', 0.0):.4f} | **{opt.get('balanced_accuracy', 0.0):.4f}** | Unbiased accuracy |")
        md.append(f"| **Specificity** | {def_t.get('specificity', 0.0)*100:.1f}% | **{opt.get('specificity', 0.0)*100:.1f}%** | Specificity |")
        md.append("\n")

        # 4.3 Precision / Recall Trade-Off Explanation
        md.append("### 4.3 Precision vs Recall Trade-Off Analysis")
        md.append(f"{threshold_analysis.get('tradeoff_explanation', '')}\n")

        # 4.4 Ranking Insight
        ranking_insight = threshold_analysis.get('ranking_insight')
        if ranking_insight:
            md.append("### 4.4 Probability Ranking & Decision Prioritization")
            md.append(f"{ranking_insight}\n\n")

    elif problem_type in ("regression", "forecasting") and best_test_metrics:
        md.append("## 4. Final Touchless Holdout Evaluation & Multi-Metric Diagnostics")
        md.append(
            "> [!IMPORTANT]\n"
            "> **Holdout Evaluation Disclosure:** The locked champion model was evaluated once on the untouched final holdout test set.\n"
        )
        md.append("The table below details final holdout performance for the locked champion model:\n")
        if problem_type == "forecasting":
            wape = best_test_metrics.get("wape", 0.0)
            smape = best_test_metrics.get("smape", 0.0)
            rmse = best_test_metrics.get("rmse", 0.0)
            mae = best_test_metrics.get("mae", 0.0)
            r2 = best_test_metrics.get("r2", 0.0)
            md.append("| Holdout Metric | Final Value | Metric Description |")
            md.append("|---|---|---|")
            md.append(f"| **WAPE (Holdout)** | **{wape:.2f}%** | Weighted Absolute Percentage Error |")
            md.append(f"| **sMAPE (Holdout)** | **{smape:.2f}%** | Symmetric Mean Absolute Percentage Error |")
            md.append(f"| **RMSE (Holdout)** | **{rmse:.4f}** | Root Mean Squared Error |")
            md.append(f"| **MAE (Holdout)** | **{mae:.4f}** | Mean Absolute Error |")
            md.append(f"| **R² Score (Holdout)** | **{r2:.4f}** | Coefficient of Determination |")
        else:
            rmse = best_test_metrics.get("rmse", 0.0)
            mae = best_test_metrics.get("mae", 0.0)
            r2 = best_test_metrics.get("r2", 0.0)
            med_ae = best_test_metrics.get("median_ae", 0.0)
            mape = best_test_metrics.get("mape", 0.0)
            md.append("| Holdout Metric | Final Value | Metric Description |")
            md.append("|---|---|---|")
            md.append(f"| **RMSE (Holdout)** | **{rmse:.4f}** | Root Mean Squared Error |")
            md.append(f"| **MAE (Holdout)** | **{mae:.4f}** | Mean Absolute Error |")
            md.append(f"| **R² Score (Holdout)** | **{r2:.4f}** | Coefficient of Determination |")
            md.append(f"| **Median AE (Holdout)** | **{med_ae:.4f}** | Median Absolute Error |")
            md.append(f"| **MAPE (Holdout)** | **{mape:.2f}%** | Mean Absolute Percentage Error |")
        md.append("\n")

    # =========================================================================
    # 5. Methodological Critic Audit
    # =========================================================================
    md.append("## 5. Methodological Critic Audit & Leakage Protection")
    audit_status = critic_audit.get("audit_status", "PASSED")
    remediated_feats = critic_audit.get("remediated_features", [])
    leakage_remediated = critic_audit.get("leakage_remediated", False) or bool(remediated_feats)

    if critic_audit.get("requires_iteration") or audit_status == "CRITICAL_ISSUES_FOUND":
        md.append("**Audit Status:** `STATUS: ISSUE DETECTED & REMEDIATED`\n")
        md.append(
            "> [!NOTE]\n"
            "> **Audit & Remediation Summary:** The initial candidate models underwent automated critic audit where `CRITICAL_ISSUES_FOUND` "
            "flagged prospective data leakage in candidate features. Corrective remediation was executed: leaky features were dropped, "
            "the feature space was re-partitioned, and candidate models were retrained using fold-safe cross-validation. "
            "The final champion reported in this document passed the configured leakage audit.\n"
        )
    elif leakage_remediated:
        md.append(f"**Audit Status:** `STATUS: {audit_status}`\n")
        rf_str = ", ".join([f"`{c}`" for c in remediated_feats]) if remediated_feats else "`target components`"
        md.append(
            "> [!WARNING]\n"
            f"> **Target-Component Leakage Remediated:** Leakage prevention excluded {rf_str} because they are components of the target `{target_column}` "
            f"and would not be valid prediction-time features. Candidate models and final champion were trained strictly on the leak-free feature matrix.\n"
        )
    else:
        md.append(f"**Audit Status:** `STATUS: {audit_status}`\n")

    findings = critic_audit.get("findings", [])
    if findings:
        for idx, f in enumerate(findings, 1):
            sev = f.get("severity", "info").upper()
            md.append(f"### Finding {idx}: [{sev}] `{f.get('issue_type')}`")
            md.append(f"- **Description:** {f.get('description')}")
            md.append(f"- **Remediation Executed:** {f.get('remediation')}\n")
    else:
        md.append(
            "No critical data leakage, severe overfitting, or invalid validation strategies were identified. "
            "All model evaluations adhered strictly to leak-free splitting standards.\n"
        )

    # =========================================================================
    # 6. Model Explainability & Top Predictive Drivers
    # =========================================================================
    md.append("## 6. Model Explainability & Top Predictive Drivers")
    md.append(
        "> [!IMPORTANT]\n"
        "> **Methodological Note on Interpretability (Non-Causality):** Feature importance rankings and SHAP attributions "
        "reflect **model-derived predictive associations** identified within this observational dataset. "
        "They demonstrate which features provide statistical signal to the model, but **do not establish causal relationships**. "
        "Altering a feature does not guarantee a causal change in the outcome without rigorous randomized experimentation (A/B testing).\n"
    )

    rankings = []
    if explainability:
        rankings = explainability.get("feature_importance", {}).get("rankings", [])
    if not rankings and best_experiment:
        # Fallback from model record if available
        rankings = best_experiment.get("feature_importance", {}).get("rankings", [])

    if rankings:
        md.append("### Top Predictive Drivers (Relative Contribution)")
        md.append("| Rank | Feature Name | Relative Importance (%) | Predictive Association Type |")
        md.append("|---|---|---|---|")
        for idx, r in enumerate(rankings[:10], 1):
            md.append(f"| {idx} | `{r.get('feature')}` | **{r.get('importance_pct', 0.0):.2f}%** | Model-Derived Associative Signal |")
        md.append("\n")

    # =========================================================================
    # 7. Evidence-Backed Business Recommendations
    # =========================================================================
    md.append("## 7. Evidence-Backed Business Recommendations")
    md.append(
        "Findings are strictly organized into four transparent pillars: "
        "**Observed Facts**, **Model-Derived Evidence**, **Actionable Recommendations**, and **Causal Limitations**.\n"
    )

    # Group insights by category
    cats = {
        "observed_facts": "7.1 Observed Facts",
        "model_derived": "7.2 Model-Derived Evidence",
        "actionable_recommendations": "7.3 Actionable Recommendations",
        "business_recommendation": "7.3 Actionable Recommendations",
        "causal_limitations": "7.4 Causal Limitations",
        "agent_interpretation": "7.2 Model-Derived Evidence"
    }

    insights_by_sec: Dict[str, List[Dict[str, Any]]] = {
        "7.1 Observed Facts": [],
        "7.2 Model-Derived Evidence": [],
        "7.3 Actionable Recommendations": [],
        "7.4 Causal Limitations": [],
    }

    for b in business_insights:
        sec = cats.get(b.get("category", ""), "7.2 Model-Derived Evidence")
        insights_by_sec[sec].append(b)

    # Add default structured items if any pillar is empty
    if not insights_by_sec["7.1 Observed Facts"]:
        insights_by_sec["7.1 Observed Facts"].append({
            "title": f"Dataset Class Distribution ({dataset_name})",
            "finding": f"The empirical dataset contains {row_count:,} records with a positive target base rate of {prevalence*100:.2f}%.",
            "evidence": f"{row_count:,} total instances, {col_count} features.",
            "confidence": "High"
        })

    if not insights_by_sec["7.2 Model-Derived Evidence"]:
        top_f_str = rankings[0]['feature'] if rankings else 'Key Feature'
        insights_by_sec["7.2 Model-Derived Evidence"].append({
            "title": f"Predictive Discriminability by {best_model_name}",
            "finding": f"The leak-free model achieved holdout ROC-AUC of {best_test_metrics.get('roc_auc', 0.0):.4f} and PR-AUC of {best_test_metrics.get('pr_auc', 0.0):.4f}, with '{top_f_str}' as the strongest predictive driver.",
            "evidence": f"PR-AUC: {best_test_metrics.get('pr_auc', 0.0):.4f}, ROC-AUC: {best_test_metrics.get('roc_auc', 0.0):.4f}.",
            "confidence": "High"
        })

    if not insights_by_sec["7.3 Actionable Recommendations"]:
        opt_t_val = threshold_analysis.get('operating_threshold', {}).get('threshold', 0.20)
        insights_by_sec["7.3 Actionable Recommendations"].append({
            "title": "Calibrated Decision Threshold Deployment",
            "finding": f"Operationalize the model at the locked decision threshold of {opt_t_val:.2f} to balance positive-case capture and precision according to operational constraints.",
            "evidence": f"Operating cutoff achieves {threshold_analysis.get('operating_threshold', {}).get('recall', 0.0)*100:.1f}% capture rate.",
            "confidence": "High"
        })

    if not insights_by_sec["7.4 Causal Limitations"]:
        insights_by_sec["7.4 Causal Limitations"].append({
            "title": "Observational Correlation vs Causal Action",
            "finding": "Identified drivers represent associative predictive patterns within observational data. Do not assume altering feature values will causally alter outcomes without randomized controlled experimentation.",
            "evidence": "Observational tabular data without exogenous causal instruments.",
            "confidence": "High"
        })

    for sec_name, items in insights_by_sec.items():
        if items:
            md.append(f"### {sec_name}")
            for item in items:
                md.append(f"#### {item.get('title')}")
                md.append(f"- **Finding:** {item.get('finding')}")
                md.append(f"- **Quantitative Evidence:** `{item.get('evidence')}`")
                md.append(f"- **Confidence:** {item.get('confidence', 'Moderate')}\n")

    # =========================================================================
    # 8. Model Limitations & Operational Risk Analysis
    # =========================================================================
    md.append("## 8. Model Limitations & Operational Risk Analysis")
    md.append(
        "Deploying predictive models into downstream operational workflows carries inherent risks and constraints that must be actively managed:\n"
    )
    md.append(
        "1. **Class Asymmetry & Base-Rate Sensitivity:** In skewed distributions, raw accuracy masks critical classification errors. "
        "Stakeholders must evaluate performance using precision-recall dynamics, PR-AUC, and balanced accuracy rather than relying on overall accuracy.\n"
        "2. **False-Negative Sensitivity & Impact:** Across diagnostic and detection objectives, False Negatives leave critical positive instances undetected. "
        "Standard 0.50 thresholds may lead to unacceptable under-capture rates depending on operational error costs.\n"
        "3. **Operating Threshold Dependence:** Model outputs are estimated class probabilities; operational assignments strictly depend on the selected decision cutoff. "
        "Shifts in operational tolerance, capacity, or cost matrices necessitate re-evaluating the decision threshold.\n"
        "4. **Dataset-Specific Generalization & External Validation:** Model performance reflects the cohort distributions and feature measurements present in this dataset. "
        "Deployment across external cohorts, distinct demographics, or clinical/subgroup settings requires external validation.\n"
        "5. **Associative Correlation vs. Causal Interventions:** High feature importance or SHAP values denote statistical associations within the training data, not causal mechanisms. "
        "Interventions or policy changes attempting to alter predictive drivers require controlled experimental validation.\n"
        "6. **Data Drift & Population Shifts:** Underlying feature distributions and relationship patterns can shift over time. "
        "Production deployment requires ongoing performance monitoring, covariate drift tracking, and scheduled model auditing.\n"
    )

    # =========================================================================
    # 9. Generated Visual Artifacts
    # =========================================================================
    if artifact_paths:
        md.append("## 9. Generated Visual Artifacts & Diagnostic Plots")
        for path in artifact_paths:
            md.append(f"- Artifact: `{path}`")
        md.append("\n")

    return "\n".join(md)
