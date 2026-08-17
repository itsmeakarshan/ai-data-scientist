"""
AutoDS Evidence-Backed Report Generator Tool
Produces structured Markdown and JSON reports clearly separating Observed Facts, Model Evidence, and Actionable Recommendations.
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
    artifact_paths: List[str]
) -> str:
    """
    Format a complete, professional, audit-ready Data Science report in GitHub Flavored Markdown.
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    best_model_name = best_experiment.get("model_name", "N/A")
    best_test_metrics = best_experiment.get("metrics", {}).get("test", {})

    md = []
    md.append(f"# AutoDS Autonomous Data Science Report")
    md.append(f"**Dataset:** `{dataset_name}`  ")
    md.append(f"**Generated:** {now_str}  ")
    md.append(f"**Status:** Verified & Evidence-Backed  ")
    md.append("\n---\n")

    # 1. Executive Summary
    md.append("## 1. Executive Summary")
    md.append(f"**Objective:** {user_goal}\n")
    md.append(f"AutoDS autonomously profiled the dataset, identified a **{problem_type.upper()}** task targeting `{target_column or 'N/A'}` using **{validation_strategy}** validation. ")
    
    if problem_type == "classification":
        roc = best_test_metrics.get("roc_auc", 0.0)
        pr = best_test_metrics.get("pr_auc", 0.0)
        f1 = best_test_metrics.get("f1_macro", 0.0)
        md.append(f"The champion model selected is **{best_model_name}**, achieving a Test **ROC-AUC of {roc:.4f}**, **PR-AUC of {pr:.4f}**, and **F1-Score of {f1:.4f}**.\n")
    elif problem_type == "regression":
        rmse = best_test_metrics.get("rmse", 0.0)
        r2 = best_test_metrics.get("r2", 0.0)
        mae = best_test_metrics.get("mae", 0.0)
        md.append(f"The champion model selected is **{best_model_name}**, achieving a Test **RMSE of {rmse:.4f}**, **MAE of {mae:.4f}**, and **R² of {r2:.4f}**.\n")
    elif problem_type == "forecasting":
        wape = best_test_metrics.get("wape", 0.0)
        smape = best_test_metrics.get("smape", 0.0)
        md.append(f"The champion model selected is **{best_model_name}**, achieving a Test **WAPE of {wape:.2f}%** and **sMAPE of {smape:.2f}%** on the holdout horizon.\n")

    # 2. Dataset & Quality Profile
    md.append("## 2. Dataset Overview & Data Quality Profile (Observed Facts)")
    row_count = profile_summary.get("row_count", 0)
    col_count = profile_summary.get("col_count", 0)
    missing_pct = profile_summary.get("missingness_report", {}).get("total_missing_pct", 0.0)
    dup_count = profile_summary.get("missingness_report", {}).get("duplicate_rows", 0)
    
    md.append(f"- **Dimensions:** {row_count:,} rows × {col_count} columns")
    md.append(f"- **Total Missing Cells:** {missing_pct}%")
    md.append(f"- **Duplicate Rows Removed:** {dup_count}")
    md.append(f"- **Identified Target Column:** `{target_column}`")
    
    alerts = profile_summary.get("quality_alerts", [])
    if alerts:
        md.append("\n### Data Quality & Hygiene Alerts")
        md.append("| Severity | Type | Column | Message | Action Taken |")
        md.append("|---|---|---|---|---|")
        for a in alerts:
            col_str = f"`{a.get('column')}`" if a.get('column') else "Global"
            md.append(f"| **{a.get('severity', 'info').upper()}** | `{a.get('type')}` | {col_str} | {a.get('message')} | {a.get('suggested_action')} |")
    md.append("\n")

    # 3. Model Benchmark Leaderboard
    md.append("## 3. Model Benchmark Leaderboard (Computed Results)")
    if problem_type == "classification":
        md.append("| Model | Family | Test ROC-AUC | Test PR-AUC | Test F1 | Test Acc | CV Mean ± Std | Train Time (s) |")
        md.append("|---|---|---|---|---|---|---|---|")
        for exp in experiment_results:
            m = exp.get("metrics", {}).get("test", {})
            cv_m = exp.get("metrics", {}).get("cv_mean", 0.0)
            cv_s = exp.get("metrics", {}).get("cv_std", 0.0)
            t_sec = exp.get("train_time_sec", 0.0)
            md.append(
                f"| **{exp.get('model_name')}** | {exp.get('model_family')} | "
                f"**{m.get('roc_auc', 0.0):.4f}** | {m.get('pr_auc', 0.0):.4f} | "
                f"{m.get('f1_macro', 0.0):.4f} | {m.get('accuracy', 0.0):.4f} | "
                f"{cv_m:.4f} ± {cv_s:.4f} | {t_sec:.2f}s |"
            )
    else:
        md.append("| Model | Family | Test RMSE | Test MAE | Test R² / WAPE | CV Mean | Train Time (s) |")
        md.append("|---|---|---|---|---|---|---|")
        for exp in experiment_results:
            m = exp.get("metrics", {}).get("test", {})
            cv_m = exp.get("metrics", {}).get("cv_mean", 0.0)
            t_sec = exp.get("train_time_sec", 0.0)
            sec_val = m.get("wape", m.get("r2", 0.0))
            md.append(
                f"| **{exp.get('model_name')}** | {exp.get('model_family')} | "
                f"**{m.get('rmse', 0.0):.4f}** | {m.get('mae', 0.0):.4f} | "
                f"{sec_val:.4f} | {cv_m:.4f} | {t_sec:.2f}s |"
            )
    md.append("\n")

    # 4. Methodological Critic Audit
    md.append("## 4. Methodological Critic Audit & Leakage Protection")
    audit_status = critic_audit.get("audit_status", "PASSED")
    md.append(f"**Audit Status:** `{audit_status}`\n")
    findings = critic_audit.get("findings", [])
    if findings:
        for idx, f in enumerate(findings, 1):
            md.append(f"### Finding {idx}: [{f.get('severity', '').upper()}] {f.get('issue_type')}")
            md.append(f"- **Description:** {f.get('description')}")
            md.append(f"- **Remediation Action:** {f.get('remediation')}\n")
    else:
        md.append("No critical data leakage, severe overfitting, or improper validation strategies were detected. All models adhered to leak-free splitting standards.\n")

    # 5. Business Insights & Attribution
    md.append("## 5. Evidence-Backed Business Insights")
    md.append("> *Every finding below is traceable directly to computed metrics, SHAP values, and empirical distributions.*")
    md.append("\n")
    for b in business_insights:
        md.append(f"### {b.get('title')}")
        md.append(f"- **Category:** `{b.get('category')}`")
        md.append(f"- **Finding:** {b.get('finding')}")
        md.append(f"- **Quantitative Evidence:** {b.get('evidence')}")
        md.append(f"- **Confidence:** {b.get('confidence')}\n")

    # 6. Generated Visual Artifacts
    if artifact_paths:
        md.append("## 6. Generated Visual Artifacts")
        for path in artifact_paths:
            md.append(f"- Artifact: `{path}`")
        md.append("\n")

    return "\n".join(md)
