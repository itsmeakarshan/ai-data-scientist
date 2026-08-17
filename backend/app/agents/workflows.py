"""
AutoDS Autonomous Workflow Engine
Orchestrates the entire Data Science lifecycle from raw data ingestion to report generation and DB persistence.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from backend.app.agents.gemini_client import gemini_client
from backend.app.agents.state import AgentState
from backend.app.core.config import settings
from backend.app.core.database import SyncSessionLocal
from backend.app.core.logging import logger
from backend.app.models.entities import (
    AnalysisRun,
    Dataset,
    DatasetProfile,
    Experiment,
    ExperimentMetric,
    ModelRecord,
    Report,
)
from backend.app.tools.critic import critique_experiment
from backend.app.tools.data_profiler import profile_dataset
from backend.app.tools.dataset_inspector import load_dataset_as_dataframe
from backend.app.tools.evaluator import (
    evaluate_classification,
    evaluate_forecasting,
    evaluate_regression,
)
from backend.app.tools.explainability import (
    calculate_feature_importance,
    compute_shap_explanations,
)
from backend.app.tools.ml_trainer import train_and_evaluate_model
from backend.app.tools.preprocessor import prepare_train_test_split
from backend.app.tools.problem_classifier import classify_problem_type
from backend.app.tools.quality_detector import detect_data_quality
from backend.app.tools.reporter import generate_full_markdown_report
from backend.app.tools.visualizer import (
    generate_actual_vs_predicted_plot,
    generate_confusion_matrix_plot,
    generate_feature_importance_plot,
    generate_roc_pr_plots,
)


def run_autonomous_datascience_pipeline(
    analysis_id: str,
    dataset_id: str,
    user_goal: str,
    target_column_override: Optional[str] = None,
    time_column_override: Optional[str] = None,
    problem_type_override: Optional[str] = None,
    sync_db_session: Optional[Any] = None
) -> AgentState:
    """
    Execute complete end-to-end autonomous analysis on a dataset.
    """
    db = sync_db_session or SyncSessionLocal()
    
    try:
        # 1. Fetch Dataset Record from DB
        db_dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not db_dataset:
            raise ValueError(f"Dataset '{dataset_id}' not found in database.")

        file_path = db_dataset.file_path
        dataset_name = db_dataset.name

        # Initialize Agent State
        state = AgentState(
            analysis_id=analysis_id,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            file_path=file_path,
            user_goal=user_goal,
            target_column=target_column_override,
            time_column=time_column_override,
            problem_type=problem_type_override or "classification",
        )

        state.current_step = "DATASET_INSPECTION"
        state.log(f"Starting autonomous pipeline on '{dataset_name}' for goal: '{user_goal}'")
        
        # Update AnalysisRun status in DB
        db_run = db.query(AnalysisRun).filter(AnalysisRun.id == analysis_id).first()
        if db_run:
            db_run.status = "RUNNING"
            db.commit()

        # Step 1: Load Data & Profile
        df_raw, meta = load_dataset_as_dataframe(file_path)
        state.current_step = "DATA_PROFILING"
        state.log(f"Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns.")
        
        profile = profile_dataset(df_raw)
        state.profile_summary = profile

        # Step 2: Problem Classification
        state.current_step = "PROBLEM_CLASSIFICATION"
        problem_info = classify_problem_type(
            df=df_raw,
            target_column=target_column_override,
            time_column=time_column_override,
            user_goal=user_goal
        )
        state.problem_type = problem_type_override or problem_info["problem_type"]
        state.sub_type = problem_info["sub_type"]
        state.target_column = target_column_override or problem_info["target_column"]
        state.time_column = time_column_override or problem_info["time_column"]
        state.validation_strategy = problem_info["recommended_validation"]
        state.log(f"Inferred Problem: {state.problem_type} ({state.sub_type}), Target: '{state.target_column}', Time: '{state.time_column}'")

        # Step 3: Analysis Planning
        state.current_step = "PLANNING"
        plan = gemini_client.generate_plan(user_goal, problem_info, profile)
        state.analysis_plan = plan
        state.log(f"Generated plan with {len(plan.get('candidate_models', []))} candidate models.")

        # Step 4: Quality & Leakage Audit (Pre-Train)
        state.current_step = "QUALITY_AUDIT"
        alerts = detect_data_quality(df_raw, profile, state.target_column)
        state.quality_alerts = alerts
        state.log(f"Data quality audit completed: {len(alerts)} alerts generated.")

        # Step 5: Leak-Free Preprocessing & Splitting
        state.current_step = "PREPROCESSING"
        drop_leaks = []
        # Check if duration exists in bank marketing dataset (to demo critic initial run vs leak-free comparison)
        X_train, X_test, y_train, y_test, prep_artifacts = prepare_train_test_split(
            df=df_raw,
            target_column=state.target_column,
            problem_type=state.problem_type,
            time_column=state.time_column,
            drop_leakage_cols=drop_leaks
        )
        state.log(f"Prepared train set ({len(X_train)} samples) and test set ({len(X_test)} samples) across {len(prep_artifacts.feature_names)} features.")

        # Step 6: Candidate Model Training & Comparison
        state.current_step = "MODEL_TRAINING"
        candidate_models = plan.get("candidate_models", [])
        if not candidate_models:
            candidate_models = ["LightGBM", "RandomForest", "Baseline"] if state.problem_type == "classification" else ["LightGBM", "Ridge", "Baseline"]

        raw_experiments = []
        models_trained = {}

        for model_name in candidate_models:
            state.log(f"Training candidate model: {model_name}...")
            exp_res = train_and_evaluate_model(
                model_name=model_name,
                problem_type=state.problem_type,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=prep_artifacts.feature_names,
                cv_folds=3
            )
            raw_experiments.append(exp_res)
            models_trained[model_name] = exp_res

        # Rank models and select initial champion
        if state.problem_type == "classification":
            sorted_exps = sorted(
                raw_experiments,
                key=lambda x: x["metrics"]["test"].get("roc_auc", x["metrics"]["test"].get("accuracy", 0.0)),
                reverse=True
            )
        else:
            sorted_exps = sorted(
                raw_experiments,
                key=lambda x: x["metrics"]["test"].get("rmse", 999999.0)
            )

        initial_best = sorted_exps[0]
        state.experiments = raw_experiments
        state.best_experiment = initial_best
        state.log(f"Initial champion model: {initial_best['model_name']} (Test Metric: {initial_best['metrics']['test']})")

        # Step 7: Methodological Critic Audit
        state.current_step = "CRITIC_AUDIT"
        critic_result = critique_experiment(
            model_name=initial_best["model_name"],
            problem_type=state.problem_type,
            metrics=initial_best["metrics"],
            feature_names=prep_artifacts.feature_names,
            validation_strategy=state.validation_strategy,
            target_column=state.target_column,
            raw_columns=list(df_raw.columns)
        )
        state.critic_findings = critic_result
        state.log(f"Critic Audit Status: {critic_result['audit_status']}, Requires Iteration: {critic_result['requires_iteration']}")

        # Step 8: Corrective Iteration if Critic flags leakage
        champion_exp = initial_best
        if critic_result.get("requires_iteration") and "REMOVE_LEAKY_FEATURES" in critic_result.get("remediation_actions", []):
            state.current_step = "CRITIC_ITERATION"
            state.log("Critic detected domain leakage ('duration'). Retraining corrected leak-free model suite...")
            
            # Retrain with 'duration' purged
            X_tr_c, X_te_c, y_tr_c, y_te_c, prep_art_c = prepare_train_test_split(
                df=df_raw,
                target_column=state.target_column,
                problem_type=state.problem_type,
                time_column=state.time_column,
                drop_leakage_cols=["duration"]
            )
            
            corrected_exp = train_and_evaluate_model(
                model_name=f"{initial_best['model_name']}_LeakFree",
                problem_type=state.problem_type,
                X_train=X_tr_c,
                y_train=y_tr_c,
                X_test=X_te_c,
                y_test=y_te_c,
                feature_names=prep_art_c.feature_names,
                cv_folds=3
            )
            
            state.experiments.append(corrected_exp)
            champion_exp = corrected_exp
            state.best_experiment = corrected_exp
            models_trained[champion_exp["model_name"]] = corrected_exp
            prep_artifacts = prep_art_c
            X_train, X_test, y_train, y_test = X_tr_c, X_te_c, y_tr_c, y_te_c
            state.log(f"Corrected model trained: {champion_exp['model_name']} (Test ROC-AUC: {champion_exp['metrics']['test'].get('roc_auc')})")

        # Step 9: Explainability & Feature Importance
        state.current_step = "EXPLAINABILITY"
        best_model_obj = champion_exp["model"]
        feature_imp = calculate_feature_importance(best_model_obj, prep_artifacts.feature_names)
        shap_res = compute_shap_explanations(best_model_obj, X_test, prep_artifacts.feature_names)
        
        state.explainability = {
            "feature_importance": feature_imp,
            "shap_summary": shap_res,
        }
        state.log("Computed SHAP values and feature importance attributions.")

        # Step 10: Visual Artifact Generation
        state.current_step = "VISUALIZATION"
        visual_paths = []
        test_m = champion_exp["metrics"]["test"]

        if state.problem_type == "classification" and test_m.get("is_binary"):
            roc_data = test_m.get("roc_curve", {})
            pr_data = test_m.get("pr_curve", {})
            curve_paths = generate_roc_pr_plots(
                roc_data=roc_data,
                pr_data=pr_data,
                roc_auc=test_m.get("roc_auc", 0.0),
                pr_auc=test_m.get("pr_auc", 0.0),
                model_name=champion_exp["model_name"],
                run_id=analysis_id
            )
            visual_paths.extend(curve_paths.values())

            cm_path = generate_confusion_matrix_plot(
                cm=test_m.get("confusion_matrix", [[0, 0], [0, 0]]),
                model_name=champion_exp["model_name"],
                run_id=analysis_id
            )
            visual_paths.append(cm_path)
            
        elif state.problem_type in ("regression", "forecasting"):
            act_pred_path = generate_actual_vs_predicted_plot(
                y_true=y_test,
                y_pred=champion_exp["y_test_pred"],
                model_name=champion_exp["model_name"],
                run_id=analysis_id,
                problem_type=state.problem_type
            )
            visual_paths.append(act_pred_path)

        # Feature Importance Plot
        imp_path = generate_feature_importance_plot(
            feature_rankings=feature_imp.get("rankings", []),
            model_name=champion_exp["model_name"],
            run_id=analysis_id
        )
        if imp_path:
            visual_paths.append(imp_path)

        state.visual_artifacts = visual_paths
        state.log(f"Generated {len(visual_paths)} visual artifact plots.")

        # Step 11: Business Insights Synthesis
        state.current_step = "INSIGHT_SYNTHESIS"
        insights = gemini_client.generate_business_insights(
            dataset_name=dataset_name,
            problem_type=state.problem_type,
            best_model_name=champion_exp["model_name"],
            test_metrics=test_m,
            top_features=feature_imp.get("rankings", [])
        )
        state.business_insights = insights

        # Step 12: Full Evidence-Backed Report Generation
        state.current_step = "REPORT_GENERATION"
        report_md = generate_full_markdown_report(
            dataset_name=dataset_name,
            user_goal=user_goal,
            problem_type=state.problem_type,
            target_column=state.target_column,
            validation_strategy=state.validation_strategy,
            profile_summary=profile,
            experiment_results=state.experiments,
            best_experiment=champion_exp,
            critic_audit=state.critic_findings,
            business_insights=state.business_insights,
            artifact_paths=state.visual_artifacts
        )
        state.final_report_markdown = report_md

        # Step 13: Database Persistence
        state.current_step = "PERSISTENCE"
        if db_run:
            db_run.status = "COMPLETED"
            db_run.problem_type = state.problem_type
            db_run.target_column = state.target_column
            db_run.time_column = state.time_column
            db_run.validation_strategy = state.validation_strategy
            db_run.plan_json = state.analysis_plan
            db_run.critic_findings_json = state.critic_findings
            db_run.completed_at = datetime.now(timezone.utc)

            # Save Experiment Records
            for exp in state.experiments:
                db_exp = Experiment(
                    analysis_id=analysis_id,
                    dataset_id=dataset_id,
                    model_name=exp["model_name"],
                    model_family=exp["model_family"],
                    hyperparameters=exp["hyperparameters"],
                    feature_names=prep_artifacts.feature_names,
                    preprocessing_config={"validation_strategy": state.validation_strategy},
                    validation_strategy=state.validation_strategy,
                    cv_folds=3,
                    train_time_sec=exp["train_time_sec"],
                    metrics_json=exp["metrics"],
                    mlflow_run_id=exp.get("mlflow_run_id"),
                    status="COMPLETED"
                )
                db.add(db_exp)
                db.flush()

                # Add granular metrics
                for split_k, m_dict in exp["metrics"].items():
                    if isinstance(m_dict, dict):
                        for m_name, m_val in m_dict.items():
                            if isinstance(m_val, (int, float)) and not np.isnan(m_val):
                                db.add(ExperimentMetric(
                                    experiment_id=db_exp.id,
                                    metric_name=m_name,
                                    metric_value=float(m_val),
                                    split_type=split_k
                                ))

                # If this is champion, save ModelRecord and link final_model_id
                if exp["model_name"] == champion_exp["model_name"]:
                    db_model = ModelRecord(
                        experiment_id=db_exp.id,
                        name=exp["model_name"],
                        task_type=state.problem_type,
                        is_best=True,
                        artifact_path=exp.get("artifacts_path") or f"models/{analysis_id}_{exp['model_name']}.pkl",
                        feature_importance_json=feature_imp,
                        shap_summary_json=shap_res,
                        metrics_json=exp["metrics"]
                    )
                    db.add(db_model)
                    db.flush()
                    db_run.final_model_id = db_model.id

            # Save Report
            db_report = Report(
                analysis_id=analysis_id,
                title=f"AutoDS Analysis Report — {dataset_name}",
                summary_markdown=f"Completed {state.problem_type} pipeline for goal '{user_goal}'. Best Model: {champion_exp['model_name']}.",
                full_report_markdown=report_md,
                business_insights_json={"insights": state.business_insights},
                methodology_json={"plan": state.analysis_plan, "critic": state.critic_findings},
                artifact_paths=state.visual_artifacts
            )
            db.add(db_report)
            db.commit()

        state.status = "COMPLETED"
        state.completed_at = datetime.now(timezone.utc)
        state.current_step = "DONE"
        state.log("Autonomous Data Science workflow successfully finished!")
        return state

    except Exception as e:
        logger.error(f"Error in autonomous pipeline: {e}", exc_info=True)
        if db_run:
            db_run.status = "FAILED"
            db_run.error_message = str(e)
            db.commit()
        raise
    finally:
        if not sync_db_session:
            db.close()
