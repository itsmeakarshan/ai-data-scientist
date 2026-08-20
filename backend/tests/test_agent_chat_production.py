"""
Production Test Suite for AutoDS Grounded AI Agent Chat
Verifies context extraction, evidence citation, CV vs holdout separation,
remediated leakage explanation, non-causal interpretability, and refusal to hallucinate.
"""


import pytest
from sqlalchemy.orm import Session

from backend.app.agents.chat_agent import answer_chat_query
from backend.app.agents.gemini_client import gemini_client
from backend.app.core.database import SyncSessionLocal
from backend.app.models.entities import (
    AnalysisRun,
    Dataset,
    DatasetProfile,
    Experiment,
    ModelRecord,
    Report,
)
from backend.app.services.analysis_context_builder import AnalysisContextBuilder


@pytest.fixture
def mock_bank_marketing_entities():
    """Seed synthetic database entities reflecting Bank Marketing UCI analysis for chat tests."""
    sync_db: Session = SyncSessionLocal()

    # 1. Dataset
    ds_id = "test_chat_bank_ds_1"
    ds = sync_db.query(Dataset).filter(Dataset.id == ds_id).first()
    if not ds:
        ds = Dataset(
            id=ds_id,
            name="Bank_Marketing_UCI",
            file_path="data/raw/bank_marketing.csv",
            file_type="csv",
            size_bytes=10240,
            row_count=41188,
            col_count=21,
            checksum="checksum_bank_123"
        )
        sync_db.add(ds)
        sync_db.flush()

        profile = DatasetProfile(
            dataset_id=ds_id,
            summary_stats={"numerical_columns": {"age": {"mean": 40.0}}},
            missingness_report={"total_missing_pct": 0.0},
            column_types={"age": "numeric", "job": "categorical", "y": "categorical"},
            candidate_targets=["y"],
            quality_alerts=[]
        )
        sync_db.add(profile)

    # 2. Analysis Run
    run_id = "test_chat_run_bank_1"
    run = sync_db.query(AnalysisRun).filter(AnalysisRun.id == run_id).first()
    if not run:
        run = AnalysisRun(
            id=run_id,
            dataset_id=ds_id,
            user_goal="Predict term deposit subscription",
            problem_type="classification",
            target_column="y",
            validation_strategy="stratified_kfold",
            status="COMPLETED",
            final_model_id="test_chat_model_lgbm_1",
            critic_findings_json={
                "audit_status": "PASSED (Remediated)",
                "leakage_remediated": True,
                "remediated_features": ["duration"],
                "findings": [{
                    "severity": "high",
                    "issue_type": "prospective_leakage",
                    "description": "Contemporaneous call duration feature 'duration' was excluded prior to training.",
                    "remediation": "Excluded from feature matrix."
                }]
            }
        )
        sync_db.add(run)
        sync_db.flush()

    # 3. Experiments (Leaderboard)
    exp_lgbm = sync_db.query(Experiment).filter(Experiment.id == "test_exp_lgbm_1").first()
    if not exp_lgbm:
        exp_lgbm = Experiment(
            id="test_exp_lgbm_1",
            analysis_id=run_id,
            dataset_id=ds_id,
            model_name="LightGBM",
            model_family="Gradient Boosting",
            hyperparameters={"learning_rate": 0.05},
            feature_names=["euribor3m", "age", "campaign"],
            preprocessing_config={"imputer": "median"},
            validation_strategy="stratified_kfold",
            cv_folds=5,
            train_time_sec=1.45,
            metrics_json={"cv_mean": 0.7977, "cv_std": 0.0076},
            status="COMPLETED"
        )
        sync_db.add(exp_lgbm)

    exp_xgb = sync_db.query(Experiment).filter(Experiment.id == "test_exp_xgb_1").first()
    if not exp_xgb:
        exp_xgb = Experiment(
            id="test_exp_xgb_1",
            analysis_id=run_id,
            dataset_id=ds_id,
            model_name="XGBoost",
            model_family="Gradient Boosting",
            hyperparameters={"learning_rate": 0.05},
            feature_names=["euribor3m", "age", "campaign"],
            preprocessing_config={"imputer": "median"},
            validation_strategy="stratified_kfold",
            cv_folds=5,
            train_time_sec=1.82,
            metrics_json={"cv_mean": 0.7912, "cv_std": 0.0081},
            status="COMPLETED"
        )
        sync_db.add(exp_xgb)

    # 4. Champion Model Record with Holdout and Threshold
    model_rec = sync_db.query(ModelRecord).filter(ModelRecord.id == "test_chat_model_lgbm_1").first()
    if not model_rec:
        model_rec = ModelRecord(
            id="test_chat_model_lgbm_1",
            experiment_id="test_exp_lgbm_1",
            name="LightGBM",
            task_type="classification",
            is_best=True,
            artifact_path="artifacts/lgbm_model.joblib",
            feature_importance_json={
                "rankings": [
                    {"feature": "euribor3m", "importance_pct": 17.20},
                    {"feature": "age", "importance_pct": 16.43},
                    {"feature": "campaign", "importance_pct": 9.44}
                ]
            },
            shap_summary_json={"top_shap_features": [{"feature": "euribor3m", "mean_abs_shap": 0.42}]},
            metrics_json={
                "test": {
                    "is_binary": True,
                    "is_imbalanced": True,
                    "roc_auc": 0.7937,
                    "pr_auc": 0.4357,
                    "accuracy": 0.8876,
                    "balanced_accuracy": 0.7380,
                    "positive_precision": 0.3540,
                    "positive_recall": 0.6853,
                    "f1_positive": 0.4667,
                    "f2_positive": 0.5750,
                    "positive_class_prevalence": 0.1127,
                    "majority_baseline": 88.7,
                    "confusion_matrix": {"tp": 636, "fp": 1159, "fn": 292, "tn": 6151},
                    "threshold_analysis": {
                        "locked_operating_threshold": {
                            "threshold": 0.10,
                            "objective": "optimised for F2 under stated objective",
                            "precision": 0.3540,
                            "recall": 0.6853,
                            "f1": 0.4667,
                            "f2": 0.5750,
                            "tp": 636,
                            "fp": 1159,
                            "fn": 292,
                            "tn": 6151,
                            "tp_gain_over_default": 403,
                            "reasoning": "Operating threshold prioritizes positive recall."
                        },
                        "default_threshold": {
                            "threshold": 0.50,
                            "precision": 0.5843,
                            "recall": 0.2511,
                            "f1": 0.3512,
                            "f2": 0.2815,
                            "tp": 233,
                            "fp": 166,
                            "fn": 695,
                            "tn": 7144
                        },
                        "oof_validation_analysis": {
                            "optimal_threshold": 0.10,
                            "optimal_score": 0.5740
                        }
                    }
                }
            }
        )
        sync_db.add(model_rec)

    # 5. Report
    rep = sync_db.query(Report).filter(Report.id == "test_chat_rep_bank_1").first()
    if not rep:
        rep = Report(
            id="test_chat_rep_bank_1",
            analysis_id=run_id,
            title="AutoDS Executive Report: Bank Marketing",
            summary_markdown="LightGBM selected as champion model.",
            full_report_markdown="## 8. Model Limitations & Operational Risk Analysis\n1. **Class Asymmetry & Base-Rate Sensitivity**: Skewed target prevalence.\n2. **False Negative Impact & Capture Economics**: Missing subscribers carries high cost.\n3. **Operating Threshold Sensitivity**: Model requires 0.10 threshold.",
            business_insights_json={"insights": [{"category": "observed_facts", "title": "Class distribution", "finding": "11.27% target prevalence", "evidence": "EDA", "confidence": "HIGH"}]},
            methodology_json={"critic": {"audit_status": "PASSED (Remediated)"}},
            artifact_paths=[]
        )
        sync_db.add(rep)

    sync_db.commit()
    yield {
        "dataset_id": ds_id,
        "analysis_id": run_id,
        "report_id": "test_chat_rep_bank_1"
    }
    sync_db.close()


def test_analysis_context_builder_completeness(mock_bank_marketing_entities):
    """Verify AnalysisContextBuilder compiles all required structured evidence without missing keys."""
    sync_db: Session = SyncSessionLocal()
    try:
        ctx = AnalysisContextBuilder.build_context(
            sync_db=sync_db,
            analysis_id=mock_bank_marketing_entities["analysis_id"]
        )

        assert ctx["has_context"] is True
        assert ctx["dataset"]["name"] == "Bank_Marketing_UCI"
        assert ctx["analysis"]["problem_type"] == "classification"
        assert len(ctx["leaderboard"]) >= 2
        assert ctx["champion_model"]["name"] == "LightGBM"
        assert ctx["champion_model"]["holdout_metrics"]["roc_auc"] == 0.7937
        assert ctx["threshold_analysis"]["selected_threshold"] == 0.10
        assert ctx["threshold_analysis"]["recall_gain_pts"] > 40.0
        assert ctx["critic_audit"]["leakage_remediated"] is True
        assert "duration" in ctx["critic_audit"]["remediated_features"]
        assert len(ctx["explainability"]["top_drivers"]) >= 3
        assert len(ctx["operational_risks"]) >= 2
    finally:
        sync_db.close()


def test_deterministic_engine_why_champion_won(mock_bank_marketing_entities):
    """Verify deterministic engine answers 'Why did LightGBM win?' citing CV leaderboard and evidence tag."""
    sync_db: Session = SyncSessionLocal()
    try:
        ctx = AnalysisContextBuilder.build_context(sync_db=sync_db, analysis_id=mock_bank_marketing_entities["analysis_id"])
        reply = gemini_client._deterministic_chat_response("Why did LightGBM win?", ctx)
        assert "LightGBM" in reply
        assert "0.7977" in reply
        assert "[Evidence: Model Leaderboard]" in reply
        assert "cross-validation" in reply.lower()
    finally:
        sync_db.close()


def test_deterministic_engine_threshold_analysis(mock_bank_marketing_entities):
    """Verify deterministic engine explains threshold selection, prevalence, and holdout shift."""
    sync_db: Session = SyncSessionLocal()
    try:
        ctx = AnalysisContextBuilder.build_context(sync_db=sync_db, analysis_id=mock_bank_marketing_entities["analysis_id"])
        reply = gemini_client._deterministic_chat_response("Why was the threshold set to 0.10?", ctx)
        assert "0.10" in reply
        assert "11.27%" in reply or "11.3%" in reply
        assert "[Evidence: Threshold & Holdout]" in reply
        assert "out-of-fold" in reply.lower() or "oof" in reply.lower()
    finally:
        sync_db.close()


def test_deterministic_engine_leakage_remediation(mock_bank_marketing_entities):
    """Verify deterministic engine accurately explains remediated leakage."""
    sync_db: Session = SyncSessionLocal()
    try:
        ctx = AnalysisContextBuilder.build_context(sync_db=sync_db, analysis_id=mock_bank_marketing_entities["analysis_id"])
        reply = gemini_client._deterministic_chat_response("Is there any data leakage?", ctx)
        assert "duration" in reply.lower()
        assert "remediated" in reply.lower()
        assert "leak-free" in reply.lower()
        assert "[Evidence: Critic Audit]" in reply
    finally:
        sync_db.close()


def test_deterministic_engine_predictive_drivers(mock_bank_marketing_entities):
    """Verify deterministic engine lists top features and non-causal note."""
    sync_db: Session = SyncSessionLocal()
    try:
        ctx = AnalysisContextBuilder.build_context(sync_db=sync_db, analysis_id=mock_bank_marketing_entities["analysis_id"])
        reply = gemini_client._deterministic_chat_response("What are the most important predictive drivers?", ctx)
        assert "euribor3m" in reply.lower()
        assert "age" in reply.lower()
        assert "causal" in reply.lower()
        assert "[Evidence: Predictive Drivers]" in reply
    finally:
        sync_db.close()


def test_deterministic_engine_cv_vs_holdout(mock_bank_marketing_entities):
    """Verify deterministic engine explains CV vs Holdout separation."""
    sync_db: Session = SyncSessionLocal()
    try:
        ctx = AnalysisContextBuilder.build_context(sync_db=sync_db, analysis_id=mock_bank_marketing_entities["analysis_id"])
        reply = gemini_client._deterministic_chat_response("What is the difference between CV performance and holdout performance?", ctx)
        assert "cross-validation" in reply.lower()
        assert "training" in reply.lower()
        assert "untouched holdout" in reply.lower()
        assert "[Evidence: Methodological Protocol]" in reply
    finally:
        sync_db.close()


def test_deterministic_engine_refusal_to_hallucinate(mock_bank_marketing_entities):
    """Verify deterministic engine refuses to hallucinate fake concepts."""
    sync_db: Session = SyncSessionLocal()
    try:
        ctx = AnalysisContextBuilder.build_context(sync_db=sync_db, analysis_id=mock_bank_marketing_entities["analysis_id"])
        reply = gemini_client._deterministic_chat_response("What is the quantum superposition score and telepathy index for this model?", ctx)
        assert "I don't have enough evidence in the current analysis to answer that." in reply
    finally:
        sync_db.close()


def test_deterministic_engine_business_manager_summary(mock_bank_marketing_entities):
    """Verify deterministic engine synthesizes stakeholder summary."""
    sync_db: Session = SyncSessionLocal()
    try:
        ctx = AnalysisContextBuilder.build_context(sync_db=sync_db, analysis_id=mock_bank_marketing_entities["analysis_id"])
        reply = gemini_client._deterministic_chat_response("Explain this report to me like I'm a business manager.", ctx)
        assert "Executive Summary" in reply or "Stakeholder" in reply
        assert "LightGBM" in reply
        assert "0.10" in reply
        assert "[Evidence: 4-Pillar Executive Synthesis]" in reply
    finally:
        sync_db.close()


def test_agent_answers_end_to_end(mock_bank_marketing_entities):
    """Verify end-to-end answer_chat_query responds with rich evidence and context."""
    sync_db: Session = SyncSessionLocal()
    try:
        res = answer_chat_query(
            user_message="Why did LightGBM win?",
            sync_db_session=sync_db,
            analysis_id=mock_bank_marketing_entities["analysis_id"]
        )

        reply = res["reply"]
        assert "LightGBM" in reply
        assert "0.7977" in reply or "0.79" in reply
        assert "cross-validation" in reply.lower() or "cv" in reply.lower()
    finally:
        sync_db.close()


def test_comparison_mode_isolation(mock_bank_marketing_entities):
    """Verify multi-analysis comparison cleanly separates Analysis A from Analysis B."""
    sync_db: Session = SyncSessionLocal()
    try:
        # Create a second dummy analysis (Regression / Bike Sharing)
        ds2 = Dataset(
            id="test_chat_bike_ds_2",
            name="Bike_Sharing_Hour",
            file_path="data/raw/hour.csv",
            file_type="csv",
            size_bytes=5000,
            row_count=17379,
            col_count=17,
            checksum="checksum_bike_456"
        )
        sync_db.merge(ds2)

        run2 = AnalysisRun(
            id="test_chat_run_bike_2",
            dataset_id="test_chat_bike_ds_2",
            user_goal="Forecast hourly rental demand",
            problem_type="forecasting",
            target_column="cnt",
            validation_strategy="time_series_split",
            status="COMPLETED",
            final_model_id="test_chat_model_xgb_2"
        )
        sync_db.merge(run2)

        exp2 = Experiment(
            id="test_exp_bike_2",
            analysis_id="test_chat_run_bike_2",
            dataset_id="test_chat_bike_ds_2",
            model_name="XGBoost",
            model_family="Gradient Boosting",
            validation_strategy="time_series_split",
            status="COMPLETED"
        )
        sync_db.merge(exp2)

        m2 = ModelRecord(
            id="test_chat_model_xgb_2",
            experiment_id="test_exp_bike_2",
            name="XGBoost",
            task_type="forecasting",
            is_best=True,
            artifact_path="artifacts/xgb_model.joblib",
            metrics_json={"test": {"wape": 0.1813, "rmse": 68.61}}
        )
        sync_db.merge(m2)
        sync_db.commit()

        ctx = AnalysisContextBuilder.build_context(
            sync_db=sync_db,
            analysis_id=mock_bank_marketing_entities["analysis_id"],
            comparison_analysis_id="test_chat_run_bike_2"
        )

        assert ctx["has_context"] is True
        assert ctx["comparison"] is not None
        assert ctx["comparison"]["dataset_name"] == "Bike_Sharing_Hour"
        assert ctx["comparison"]["champion_model"] == "XGBoost"

        reply = gemini_client._deterministic_chat_response("Compare this with my Bike Sharing analysis", ctx)
        assert "Analysis A: Bank_Marketing_UCI" in reply
        assert "Analysis B: Bike_Sharing_Hour" in reply
        assert "[Evidence: Multi-Dataset Comparison]" in reply
    finally:
        sync_db.close()


def test_target_component_leakage_explanation():
    """Verify deterministic engine explains target-component leakage when detected."""
    ctx = {
        "dataset": {"name": "hour.csv"},
        "analysis": {"problem_type": "forecasting", "target_column": "cnt"},
        "critic_audit": {
            "audit_status": "PASSED (Remediated)",
            "leakage_remediated": True,
            "remediated_features": ["casual", "registered"]
        }
    }
    reply = gemini_client._deterministic_chat_response("Is there any data leakage?", ctx)
    assert "casual" in reply
    assert "registered" in reply
    assert "remediated" in reply.lower()
    assert "leak-free" in reply.lower()
    assert "[Evidence: Critic Audit]" in reply


@pytest.mark.asyncio
async def test_agent_api_endpoints(mock_bank_marketing_entities):
    """Verify GET /agent/context and POST /agent/chat API endpoints via FastAPI AsyncClient."""
    from httpx import ASGITransport, AsyncClient

    from backend.app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # 1. Test GET /agent/context
        res_ctx = await ac.get(f"/api/v1/agent/context?analysis_id={mock_bank_marketing_entities['analysis_id']}")
        assert res_ctx.status_code == 200
        ctx_data = res_ctx.json()
        assert ctx_data["has_context"] is True
        assert ctx_data["dataset"]["name"] == "Bank_Marketing_UCI"
        assert ctx_data["champion_model"]["name"] == "LightGBM"

        # 2. Test POST /api/v1/agent/chat
        payload = {
            "analysis_id": mock_bank_marketing_entities["analysis_id"],
            "content": "Why did LightGBM win?"
        }
        res_chat = await ac.post("/api/v1/agent/chat", json=payload)
        assert res_chat.status_code == 200
        chat_data = res_chat.json()
        assert chat_data["role"] == "assistant"
        assert len(chat_data["content"]) > 10
        assert "LightGBM" in chat_data["content"]


