"""
AutoDS Analysis Context Builder
Constructs structured, compact, and evidence-grounded context dictionaries from database entities
(Datasets, AnalysisRuns, Experiments, ModelRecords, Reports) for the conversational Chat Agent.
"""

from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from backend.app.models.entities import (
    AnalysisRun,
    Dataset,
    DatasetProfile,
    Experiment,
    ModelRecord,
    Report,
)


class AnalysisContextBuilder:
    """Builds unified, highly-structured context payloads for the AutoDS Chat Agent."""

    @staticmethod
    def build_context(
        sync_db: Session,
        analysis_id: Optional[str] = None,
        report_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        comparison_analysis_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract complete analysis, experiment, threshold, leakage, and report evidence.
        Resolves the most specific entity available (Report -> AnalysisRun -> Dataset).
        """
        context: Dict[str, Any] = {
            "has_context": False,
            "context_type": "none",
            "dataset": None,
            "analysis": None,
            "leaderboard": [],
            "champion_model": None,
            "threshold_analysis": None,
            "critic_audit": None,
            "explainability": None,
            "business_insights": [],
            "operational_risks": [],
            "report_summary": None,
            "latest_prediction": None,
            "comparison": None
        }

        # 1. Resolve AnalysisRun and Report
        analysis_run: Optional[AnalysisRun] = None
        report_obj: Optional[Report] = None
        dataset_obj: Optional[Dataset] = None

        if report_id:
            report_obj = sync_db.query(Report).filter(Report.id == report_id).first()
            if report_obj:
                analysis_run = sync_db.query(AnalysisRun).filter(AnalysisRun.id == report_obj.analysis_id).first()

        if not analysis_run and analysis_id:
            analysis_run = sync_db.query(AnalysisRun).filter(AnalysisRun.id == analysis_id).first()
            if analysis_run:
                report_obj = sync_db.query(Report).filter(Report.analysis_id == analysis_run.id).first()

        if not analysis_run and dataset_id:
            # Find latest completed analysis run for this dataset
            analysis_run = sync_db.query(AnalysisRun).filter(
                AnalysisRun.dataset_id == dataset_id,
                AnalysisRun.status == "COMPLETED"
            ).order_by(AnalysisRun.created_at.desc()).first()
            if analysis_run:
                report_obj = sync_db.query(Report).filter(Report.analysis_id == analysis_run.id).first()

        # 2. Resolve Dataset
        if analysis_run:
            dataset_obj = sync_db.query(Dataset).filter(Dataset.id == analysis_run.dataset_id).first()
        elif dataset_id:
            dataset_obj = sync_db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset_obj and not analysis_run:
            return context

        context["has_context"] = True

        # 3. Populate Dataset Metadata
        if dataset_obj:
            profile: Optional[DatasetProfile] = dataset_obj.profile
            missingness = profile.missingness_report if profile else {}
            context["dataset"] = {
                "id": dataset_obj.id,
                "name": dataset_obj.name,
                "file_path": dataset_obj.file_path,
                "row_count": dataset_obj.row_count,
                "col_count": dataset_obj.col_count,
                "total_missing_pct": missingness.get("total_missing_pct", 0.0) if isinstance(missingness, dict) else 0.0,
                "column_types": profile.column_types if profile else {},
                "quality_alerts": profile.quality_alerts if profile else [],
                "candidate_targets": profile.candidate_targets if profile else [],
            }
            context["context_type"] = "dataset"

        # 4. Populate Analysis Run Details
        if analysis_run:
            context["context_type"] = "analysis"
            context["analysis"] = {
                "id": analysis_run.id,
                "dataset_id": analysis_run.dataset_id,
                "user_goal": analysis_run.user_goal,
                "problem_type": analysis_run.problem_type,
                "target_column": analysis_run.target_column,
                "time_column": analysis_run.time_column,
                "validation_strategy": analysis_run.validation_strategy,
                "status": analysis_run.status,
                "created_at": analysis_run.created_at.isoformat() if analysis_run.created_at else None,
                "completed_at": analysis_run.completed_at.isoformat() if analysis_run.completed_at else None,
            }

            # 5. Populate Experiments & Leaderboard (CV Ranking)
            experiments = sync_db.query(Experiment).filter(Experiment.analysis_id == analysis_run.id).all()
            leaderboard_list = []
            for exp in experiments:
                metrics = exp.metrics_json or {}
                leaderboard_list.append({
                    "model_name": exp.model_name,
                    "model_family": exp.model_family,
                    "cv_mean": metrics.get("cv_mean"),
                    "cv_std": metrics.get("cv_std"),
                    "train_time_sec": exp.train_time_sec,
                    "status": "Champion" if exp.model_name == analysis_run.final_model_id or (report_obj and exp.model_name in (report_obj.title or "")) else "Candidate"
                })

            # Sort leaderboard descending by cv_mean if available
            leaderboard_list.sort(key=lambda x: (x.get("cv_mean") is not None, x.get("cv_mean") or 0.0), reverse=True)
            context["leaderboard"] = leaderboard_list

            # 6. Populate Champion Model & Touchless Holdout Evaluation
            champion_model_rec: Optional[ModelRecord] = None
            if analysis_run.final_model_id:
                champion_model_rec = sync_db.query(ModelRecord).filter(ModelRecord.id == analysis_run.final_model_id).first()
            if not champion_model_rec and experiments:
                # Fallback to model record with is_best == True
                champion_model_rec = sync_db.query(ModelRecord).filter(
                    ModelRecord.experiment_id.in_([e.id for e in experiments]),
                    ModelRecord.is_best == True
                ).first()

            if champion_model_rec:
                test_metrics = champion_model_rec.metrics_json.get("test", {}) if champion_model_rec.metrics_json else {}
                context["champion_model"] = {
                    "id": champion_model_rec.id,
                    "name": champion_model_rec.name,
                    "task_type": champion_model_rec.task_type,
                    "holdout_metrics": {
                        "roc_auc": test_metrics.get("roc_auc"),
                        "pr_auc": test_metrics.get("pr_auc"),
                        "accuracy": test_metrics.get("accuracy"),
                        "balanced_accuracy": test_metrics.get("balanced_accuracy"),
                        "positive_precision": test_metrics.get("positive_precision", test_metrics.get("precision_positive")),
                        "positive_recall": test_metrics.get("positive_recall", test_metrics.get("recall_positive")),
                        "f1_score": test_metrics.get("f1_positive", test_metrics.get("f1_macro")),
                        "f2_score": test_metrics.get("f2_positive", test_metrics.get("f2")),
                        "specificity": test_metrics.get("specificity"),
                        "prevalence": test_metrics.get("positive_class_prevalence", test_metrics.get("prevalence")),
                        "majority_baseline": test_metrics.get("majority_baseline"),
                        "rmse": test_metrics.get("rmse"),
                        "mae": test_metrics.get("mae"),
                        "r2": test_metrics.get("r2"),
                        "wape": test_metrics.get("wape"),
                        "smape": test_metrics.get("smape"),
                        "confusion_matrix": test_metrics.get("confusion_matrix"),
                    }
                }

                # 7. Populate Decision Threshold Analysis (for classification)
                threshold_res = test_metrics.get("threshold_analysis", {})
                if threshold_res:
                    locked_th = threshold_res.get("locked_operating_threshold", threshold_res.get("operating_threshold", {}))
                    default_th = threshold_res.get("default_threshold", {})
                    oof_th = threshold_res.get("oof_validation_analysis", {})

                    context["threshold_analysis"] = {
                        "is_imbalanced": test_metrics.get("is_imbalanced", False),
                        "positive_prevalence": test_metrics.get("positive_class_prevalence", test_metrics.get("prevalence")),
                        "majority_baseline": round(max(test_metrics.get("positive_class_prevalence", 0.0), 1.0 - test_metrics.get("positive_class_prevalence", 0.0)) * 100, 1),
                        "selected_threshold": locked_th.get("threshold", 0.50),
                        "threshold_objective": locked_th.get("objective", "optimised under stated objective"),
                        "oof_validation": oof_th,
                        "locked_holdout": locked_th,
                        "default_holdout": default_th,
                        "recall_gain_pts": round((locked_th.get("recall", 0.0) - default_th.get("recall", 0.0)) * 100, 1),
                        "tp_gain_over_default": locked_th.get("tp_gain_over_default", 0),
                        "fn_reduction": default_th.get("fn", 0) - locked_th.get("fn", 0),
                        "reasoning": locked_th.get("reasoning", "")
                    }

                # 8. Populate Feature Importance / SHAP
                feat_rankings = champion_model_rec.feature_importance_json.get("rankings", []) if champion_model_rec.feature_importance_json else []
                context["explainability"] = {
                    "top_drivers": feat_rankings[:10],
                    "total_features": len(feat_rankings)
                }

            # 9. Populate Critic Audit Findings
            critic_data = analysis_run.critic_findings_json or {}
            context["critic_audit"] = {
                "audit_status": critic_data.get("audit_status", "PASSED"),
                "leakage_remediated": critic_data.get("leakage_remediated", False),
                "remediated_features": critic_data.get("remediated_features", []),
                "findings": critic_data.get("findings", []),
                "remediation_actions": critic_data.get("remediation_actions", [])
            }

        # 10. Populate Report Details
        if report_obj:
            context["context_type"] = "report"
            context["report_summary"] = {
                "id": report_obj.id,
                "title": report_obj.title,
                "summary_markdown": report_obj.summary_markdown,
                "artifact_paths": report_obj.artifact_paths or []
            }
            if report_obj.business_insights_json:
                context["business_insights"] = report_obj.business_insights_json.get("insights", [])

            # Extract Operational Risks from full report markdown if available
            raw_md = report_obj.full_report_markdown or ""
            if "## 8. Model Limitations & Operational Risk Analysis" in raw_md:
                import re
                m = re.search(r'## 8\. Model Limitations & Operational Risk Analysis\s*\n+([\s\S]+?)(?=\n##|\Z)', raw_md)
                if m:
                    lines = [l.strip() for l in m.group(1).split('\n') if re.match(r'^\d+\.\s*\*\*', l.strip())]
                    context["operational_risks"] = lines

        # 11. Handle Comparison Context if requested
        if comparison_analysis_id:
            comp_run = sync_db.query(AnalysisRun).filter(AnalysisRun.id == comparison_analysis_id).first()
            if comp_run:
                comp_ds = sync_db.query(Dataset).filter(Dataset.id == comp_run.dataset_id).first()
                comp_champ = None
                if comp_run.final_model_id:
                    comp_champ = sync_db.query(ModelRecord).filter(ModelRecord.id == comp_run.final_model_id).first()

                comp_test_m = comp_champ.metrics_json.get("test", {}) if comp_champ and comp_champ.metrics_json else {}
                context["comparison"] = {
                    "analysis_id": comp_run.id,
                    "dataset_name": comp_ds.name if comp_ds else "Comparison Dataset",
                    "problem_type": comp_run.problem_type,
                    "target_column": comp_run.target_column,
                    "champion_model": comp_champ.name if comp_champ else "Unknown",
                    "holdout_metrics": comp_test_m
                }

        return context
