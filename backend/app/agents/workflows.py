"""
AutoDS Autonomous Workflow Engine
Orchestrates the entire Data Science lifecycle from raw data ingestion to report generation and DB persistence.
"""

import os
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from backend.app.agents.gemini_client import gemini_client
from backend.app.agents.stage_tracker import (
    complete_stage_tracking,
    fail_stage_tracking,
    start_stage_tracking,
    update_stage_progress,
)
from backend.app.agents.state import AgentState
from backend.app.core.database import SyncSessionLocal, with_db_retry
from backend.app.core.logging import logger
from backend.app.models.entities import (
    AnalysisRun,
    Dataset,
    Experiment,
    ExperimentMetric,
    ModelRecord,
    Report,
)
from backend.app.tools.critic import critique_experiment
from backend.app.tools.data_profiler import profile_dataset
from backend.app.tools.dataset_inspector import load_dataset_as_dataframe
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
    generate_residual_plot,
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
    db_run = None
    db = sync_db_session or SyncSessionLocal()

    try:
        # 1. Fetch Dataset & AnalysisRun Records from DB with retry
        def _fetch_initial_data():
            nonlocal db_run
            ds = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not ds:
                raise ValueError(f"Dataset '{dataset_id}' not found in database.")
            run = db.query(AnalysisRun).filter(AnalysisRun.id == analysis_id).first()
            if run:
                run.status = "RUNNING"
                db.commit()
            return ds, run

        db_dataset, db_run = with_db_retry(_fetch_initial_data, max_retries=5, initial_delay=0.1, backoff=1.5)

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

        start_stage_tracking(analysis_id)
        state.current_step = "DATASET_INSPECTION"
        state.log(f"Starting autonomous pipeline on '{dataset_name}' for goal: '{user_goal}'")

        # =====================================================================
        # STAGE 1: Dataset Inspection & Profiling
        # =====================================================================
        update_stage_progress(
            analysis_id,
            stage_number=1,
            stage_details="Inspecting schema, calculating missingness, and profiling distributions"
        )
        df_raw, meta = load_dataset_as_dataframe(file_path)
        state.current_step = "DATA_PROFILING"
        state.log(f"Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns.")

        profile = profile_dataset(df_raw)
        state.profile_summary = profile

        # =====================================================================
        # STAGE 2: Problem Classification & Target Selection
        # =====================================================================
        update_stage_progress(
            analysis_id,
            stage_number=2,
            stage_details="Inferring problem type (classification vs regression), target, and validation strategy"
        )
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

        # =====================================================================
        # STAGE 3: Autonomous Analysis Planning
        # =====================================================================
        update_stage_progress(
            analysis_id,
            stage_number=3,
            stage_details="Generating multi-model candidate plan and quality audit protocol via Gemini 3.1"
        )
        state.current_step = "PLANNING"
        plan = gemini_client.generate_plan(user_goal, problem_info, profile)
        state.analysis_plan = plan
        state.log(f"Generated plan with {len(plan.get('candidate_models', []))} candidate models.")

        # Quality & Leakage Audit (Pre-Train)
        state.current_step = "QUALITY_AUDIT"
        alerts = detect_data_quality(df_raw, profile, state.target_column)
        state.quality_alerts = alerts
        state.log(f"Data quality audit completed: {len(alerts)} alerts generated.")

        # Identify prospective / target-component leakage before model training
        from backend.app.tools.quality_detector import detect_target_component_leakage
        pretrain_leaks, leak_expls = detect_target_component_leakage(df_raw, state.target_column)
        state.remediated_leakage_cols = pretrain_leaks
        state.leakage_explanations = leak_expls
        if pretrain_leaks:
            state.log(f"Pre-train leakage prevention detected and excluded leaky columns: {pretrain_leaks}")

        # =====================================================================
        # STAGE 4: Leak-Free Preprocessing & Splitting
        # =====================================================================
        update_stage_progress(
            analysis_id,
            stage_number=4,
            stage_details="Executing fit-on-train encoding, imputing, and leak-free train/test partition"
        )
        state.current_step = "PREPROCESSING"
        X_train, X_test, y_train, y_test, prep_artifacts = prepare_train_test_split(
            df=df_raw,
            target_column=state.target_column,
            problem_type=state.problem_type,
            time_column=state.time_column,
            drop_leakage_cols=pretrain_leaks
        )
        state.log(f"Prepared train set ({len(X_train)} samples) and test set ({len(X_test)} samples) across {len(prep_artifacts.feature_names)} features.")

        # =====================================================================
        # STAGE 5: Candidate Model Training & CV
        # =====================================================================
        state.current_step = "MODEL_TRAINING"
        candidate_models = plan.get("candidate_models", [])
        if not candidate_models:
            candidate_models = ["LightGBM", "RandomForest", "Baseline"] if state.problem_type == "classification" else ["LightGBM", "Ridge", "Baseline"]

        raw_experiments = []
        models_trained = {}

        for model_name in candidate_models:
            update_stage_progress(
                analysis_id,
                stage_number=5,
                current_model=model_name,
                models_evaluated=list(models_trained.keys()),
                stage_details=f"Training candidate algorithm: {model_name} with cross-validation & MLflow logging"
            )
            state.log(f"Training candidate model: {model_name}...")
            exp_res = train_and_evaluate_model(
                model_name=model_name,
                problem_type=state.problem_type,
                X_train=X_train,
                y_train=y_train,
                feature_names=prep_artifacts.feature_names,
                cv_folds=3,
                user_goal=user_goal,
                raw_X_train=prep_artifacts.raw_X_train,
                num_cols=prep_artifacts.numerical_cols,
                cat_cols=prep_artifacts.categorical_cols
            )
            raw_experiments.append(exp_res)
            models_trained[model_name] = exp_res

        # =====================================================================
        # STAGE 6: Multi-Metric Leaderboard Ranking (Model Selection via CV)
        # =====================================================================
        update_stage_progress(
            analysis_id,
            stage_number=6,
            current_model=None,
            models_evaluated=list(models_trained.keys()),
            stage_details="Ranking candidate models using cross-validation performance on training portion"
        )
        if state.problem_type == "classification":
            sorted_exps = sorted(
                raw_experiments,
                key=lambda x: x["metrics"].get("cv_mean", 0.0),
                reverse=True
            )
        else:
            sorted_exps = sorted(
                raw_experiments,
                key=lambda x: x["metrics"].get("cv_mean", 999999.0)
            )

        state.log("CV COMPLETE")
        state.log("Comparing candidate models using CV metrics (cv_mean)...")
        initial_best = sorted_exps[0]
        state.experiments = raw_experiments
        state.best_experiment = initial_best
        state.log(f"Champion model locked: {initial_best['model_name']} (CV Score: {initial_best['metrics'].get('cv_mean', 0.0)})")
        if initial_best.get("locked_operating_threshold") is not None:
            state.log(f"Operating threshold locked: {initial_best.get('locked_operating_threshold')}")

        # =====================================================================
        # STAGE 7: Methodological Critic Audit
        # =====================================================================
        update_stage_progress(
            analysis_id,
            stage_number=7,
            models_evaluated=list(models_trained.keys()),
            stage_details="Methodological audit for data leakage, severe overfitting, and invalid splits"
        )
        state.current_step = "CRITIC_AUDIT"
        critic_result = critique_experiment(
            model_name=initial_best["model_name"],
            problem_type=state.problem_type,
            metrics=initial_best["metrics"],
            feature_names=prep_artifacts.feature_names,
            validation_strategy=state.validation_strategy,
            target_column=state.target_column,
            raw_columns=list(df_raw.columns),
            remediated_features=getattr(state, "remediated_leakage_cols", []),
            leakage_explanations=getattr(state, "leakage_explanations", {})
        )
        state.critic_findings = critic_result
        state.log(f"Critic Audit Status: {critic_result['audit_status']}, Requires Iteration: {critic_result['requires_iteration']}")

        # Corrective Iteration if Critic flags leakage
        champion_exp = initial_best
        if critic_result.get("requires_iteration") and "REMOVE_LEAKY_FEATURES" in critic_result.get("remediation_actions", []):
            state.current_step = "CRITIC_ITERATION"

            leaky_features_to_drop = []
            for f in critic_result.get("findings", []):
                if f.get("issue_type") in ("domain_target_leakage", "potential_target_leakage", "leaky_feature"):
                    leaky_features_to_drop.extend(f.get("affected_components", []))
            leaky_features_to_drop = list(set(leaky_features_to_drop))

            state.log(f"Critic detected prospective data leakage in features: {leaky_features_to_drop}. Retraining corrected leak-free model suite...")

            X_tr_c, X_te_c, y_tr_c, y_te_c, prep_art_c = prepare_train_test_split(
                df=df_raw,
                target_column=state.target_column,
                problem_type=state.problem_type,
                time_column=state.time_column,
                drop_leakage_cols=leaky_features_to_drop
            )

            leak_free_exps = []
            for m_name in candidate_models:
                lf_name = f"{m_name}_LeakFree"
                update_stage_progress(
                    analysis_id,
                    stage_number=7,
                    current_model=lf_name,
                    models_evaluated=list(models_trained.keys()),
                    stage_details=f"Retraining leak-free candidate: {lf_name}"
                )
                lf_exp = train_and_evaluate_model(
                    model_name=lf_name,
                    problem_type=state.problem_type,
                    X_train=X_tr_c,
                    y_train=y_tr_c,
                    feature_names=prep_art_c.feature_names,
                    cv_folds=3,
                    user_goal=user_goal,
                    raw_X_train=prep_art_c.raw_X_train,
                    num_cols=prep_art_c.numerical_cols,
                    cat_cols=prep_art_c.categorical_cols
                )
                state.experiments.append(lf_exp)
                leak_free_exps.append(lf_exp)
                models_trained[lf_name] = lf_exp

            if state.problem_type == "classification":
                sorted_lf = sorted(
                    leak_free_exps,
                    key=lambda x: x["metrics"].get("cv_mean", 0.0),
                    reverse=True
                )
            else:
                sorted_lf = sorted(
                    leak_free_exps,
                    key=lambda x: x["metrics"].get("cv_mean", 999999.0)
                )

            champion_exp = sorted_lf[0]
            state.best_experiment = champion_exp
            prep_artifacts = prep_art_c
            X_train, X_test, y_train, y_test = X_tr_c, X_te_c, y_tr_c, y_te_c
            state.log(f"Corrected leak-free champion selected via CV: {champion_exp['model_name']} (CV Mean Score: {champion_exp['metrics'].get('cv_mean', 0.0)})")

        # EVALUATE LOCKED CHAMPION ON HOLDOUT EXACTLY ONCE
        state.log(f"Evaluating locked champion {champion_exp['model_name']} on untouched final holdout set.")
        from backend.app.tools.ml_trainer import evaluate_locked_champion_on_holdout
        champion_exp = evaluate_locked_champion_on_holdout(
            champion_exp=champion_exp,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            user_goal=user_goal,
            track_mlflow=True
        )
        state.best_experiment = champion_exp

        # =====================================================================
        # STAGE 8: SHAP Explainability & Feature Attribution
        # =====================================================================
        update_stage_progress(
            analysis_id,
            stage_number=8,
            current_model=None,
            models_evaluated=list(models_trained.keys()),
            stage_details="Extracting TreeSHAP feature attributions and generating diagnostic visualizations"
        )
        state.current_step = "EXPLAINABILITY"
        best_model_obj = champion_exp["model"]
        feature_imp = calculate_feature_importance(best_model_obj, prep_artifacts.feature_names)
        shap_res = compute_shap_explanations(best_model_obj, X_test, prep_artifacts.feature_names)

        state.explainability = {
            "feature_importance": feature_imp,
            "shap_summary": shap_res,
        }
        state.log("Computed SHAP values and feature importance attributions.")

        # Visual Artifact Generation
        state.current_step = "VISUALIZATION"
        visual_paths = []
        test_m = champion_exp["metrics"]["test"]
        model_name = champion_exp["model_name"]

        if state.problem_type == "classification":
            # 1. ROC Curve & Precision-Recall Curve (Binary & Multiclass OvR)
            roc_data = test_m.get("roc_curve")
            pr_data = test_m.get("pr_curve")
            if roc_data and pr_data:
                try:
                    curve_paths = generate_roc_pr_plots(
                        roc_data=roc_data,
                        pr_data=pr_data,
                        roc_auc=test_m.get("roc_auc", 0.0),
                        pr_auc=test_m.get("pr_auc", 0.0),
                        model_name=model_name,
                        run_id=analysis_id
                    )
                    visual_paths.extend(curve_paths.values())
                except Exception as e:
                    state.log(f"Warning: could not generate ROC/PR plots: {e}")

            # 2. Confusion Matrix (Untouched Holdout Test Set)
            cm = test_m.get("confusion_matrix")
            if cm is not None and len(cm) > 0:
                try:
                    class_labels = test_m.get("class_labels")
                    th_val = test_m.get("operating_threshold") if test_m.get("is_binary") else None
                    cm_path = generate_confusion_matrix_plot(
                        cm=cm,
                        model_name=model_name,
                        run_id=analysis_id,
                        class_labels=class_labels,
                        threshold=th_val
                    )
                    visual_paths.append(cm_path)
                except Exception as e:
                    state.log(f"Warning: could not generate confusion matrix plot: {e}")

        elif state.problem_type in ("regression", "forecasting"):
            y_pred = champion_exp.get("y_test_pred")
            if y_pred is None and best_model_obj is not None:
                try:
                    y_pred = best_model_obj.predict(X_test)
                    champion_exp["y_test_pred"] = y_pred
                except Exception as e:
                    state.log(f"Warning: could not compute holdout predictions for visual plot: {e}")

            if y_pred is not None:
                try:
                    act_pred_path = generate_actual_vs_predicted_plot(
                        y_true=y_test,
                        y_pred=y_pred,
                        model_name=model_name,
                        run_id=analysis_id,
                        problem_type=state.problem_type
                    )
                    visual_paths.append(act_pred_path)
                except Exception as e:
                    state.log(f"Warning: could not generate actual vs predicted plot: {e}")

                try:
                    residual_path = generate_residual_plot(
                        y_true=y_test,
                        y_pred=y_pred,
                        model_name=model_name,
                        run_id=analysis_id,
                        problem_type=state.problem_type
                    )
                    visual_paths.append(residual_path)
                except Exception as e:
                    state.log(f"Warning: could not generate residual diagnostics plot: {e}")

        # 3. Top Predictive Drivers / Feature Importance Plot (All problem types where available)
        if feature_imp and feature_imp.get("rankings"):
            try:
                imp_path = generate_feature_importance_plot(
                    feature_rankings=feature_imp.get("rankings", []),
                    model_name=model_name,
                    run_id=analysis_id
                )
                if imp_path:
                    visual_paths.append(imp_path)
            except Exception as e:
                state.log(f"Warning: could not generate feature importance plot: {e}")

        state.visual_artifacts = visual_paths
        state.log(f"Generated {len(visual_paths)} visual artifact plots.")

        # =====================================================================
        # STAGE 9: Evidence-Backed Report Synthesis
        # =====================================================================
        update_stage_progress(
            analysis_id,
            stage_number=9,
            models_evaluated=list(models_trained.keys()),
            stage_details="Synthesizing executive business insights and compiling comprehensive final report"
        )
        state.current_step = "INSIGHT_SYNTHESIS"
        insights = gemini_client.generate_business_insights(
            dataset_name=dataset_name,
            problem_type=state.problem_type,
            best_model_name=champion_exp["model_name"],
            test_metrics=test_m,
            top_features=feature_imp.get("rankings", [])
        )
        state.business_insights = insights

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
            artifact_paths=state.visual_artifacts,
            explainability=state.explainability
        )
        state.final_report_markdown = report_md

        # Database Persistence
        state.current_step = "PERSISTENCE"
        def _persist_pipeline_records():
            nonlocal db_run
            if db_run is None:
                db_run = db.query(AnalysisRun).filter(AnalysisRun.id == analysis_id).first()

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
                        model_family=exp.get("model_family", exp.get("family", "gradient_boosting")),
                        hyperparameters=exp.get("hyperparameters", exp.get("params", {})),
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
                        os.makedirs("artifacts/models", exist_ok=True)
                        saved_model_path = f"artifacts/models/{analysis_id}_{exp['model_name']}.joblib"
                        try:
                            joblib.dump(best_model_obj, saved_model_path)
                        except Exception as dump_err:
                            state.log(f"Warning: could not serialize model object: {dump_err}")

                        db_model = ModelRecord(
                            experiment_id=db_exp.id,
                            name=exp["model_name"],
                            task_type=state.problem_type,
                            is_best=True,
                            artifact_path=saved_model_path,
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

        with_db_retry(_persist_pipeline_records, max_retries=5, initial_delay=0.1, backoff=1.5)

        complete_stage_tracking(analysis_id, models_evaluated=list(models_trained.keys()))
        state.status = "COMPLETED"
        state.completed_at = datetime.now(timezone.utc)
        state.current_step = "DONE"
        state.log("Autonomous Data Science workflow successfully finished!")
        return state

    except Exception as e:
        logger.error(f"Error in autonomous pipeline: {e}", exc_info=True)
        fail_stage_tracking(analysis_id, str(e))
        if db is not None:
            try:
                db.rollback()
                if db_run is None:
                    db_run = db.query(AnalysisRun).filter(AnalysisRun.id == analysis_id).first()
                if db_run is not None:
                    db_run.status = "FAILED"
                    db_run.error_message = str(e)
                    db.commit()
            except Exception as db_err:
                logger.warning(f"Could not update AnalysisRun failure status in DB: {db_err}")
        raise
    finally:
        if not sync_db_session and db is not None:
            try:
                db.close()
            except Exception:
                pass
