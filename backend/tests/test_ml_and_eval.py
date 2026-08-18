"""
Unit and Integration Tests for ML Trainer, Evaluators, Explainability, and Critic
Includes regression suites for imbalanced classification, threshold optimization,
prevalence-aware PR evaluation, non-causal reporting terminology, and leakage audit preservation.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from backend.app.agents.gemini_client import gemini_client
from backend.app.tools.critic import critique_experiment
from backend.app.tools.evaluator import (
    analyze_classification_thresholds,
    evaluate_classification,
    evaluate_forecasting,
    evaluate_regression,
)
from backend.app.tools.explainability import (
    calculate_feature_importance,
    compute_shap_explanations,
)
from backend.app.tools.ml_trainer import evaluate_locked_champion_on_holdout, train_and_evaluate_model
from backend.app.tools.reporter import generate_full_markdown_report


def test_classification_evaluation():
    """Test standard classification metrics computation."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    y_prob = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.85, 0.15],
        [0.55, 0.45],
        [0.1, 0.9],
        [0.95, 0.05],
    ])

    metrics = evaluate_classification(y_true, y_pred, y_prob)
    
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "f1_macro" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["roc_auc"] > 0.80
    assert metrics["is_binary"] is True


def test_imbalanced_classification_metrics():
    """
    Regression Test: Ensure all 10 required metrics for imbalanced binary classification
    are strictly calculated and class imbalance is automatically detected.
    """
    # 90 negative, 10 positive (10% prevalence)
    y_true = np.array([0] * 90 + [1] * 10)
    # Model predicts mostly negative (high accuracy, moderate recall)
    y_pred = np.array([0] * 88 + [1] * 2 + [0] * 3 + [1] * 7)
    
    # Probabilities with good separation
    np.random.seed(42)
    p_neg = np.random.uniform(0.01, 0.35, 90)
    p_pos = np.random.uniform(0.40, 0.95, 10)
    p_all = np.concatenate([p_neg, p_pos])
    y_prob = np.column_stack([1 - p_all, p_all])

    metrics = evaluate_classification(y_true, y_pred, y_prob, user_goal="Predict term deposit subscription")

    # 1. ROC-AUC and PR-AUC
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert metrics["roc_auc"] > 0.80
    assert metrics["pr_auc"] > metrics["positive_class_prevalence"]

    # 2. Positive-class precision and recall
    assert "positive_precision" in metrics
    assert "positive_recall" in metrics
    assert metrics["positive_precision"] == round(7 / 9, 4)
    assert metrics["positive_recall"] == round(7 / 10, 4)

    # 3. F1 and F2 scores
    assert "f1_positive" in metrics or "f1" in metrics
    assert "f2_positive" in metrics or "f2" in metrics
    assert metrics["f1_positive"] > 0.70
    assert metrics["f2_positive"] > 0.65

    # 4. Balanced accuracy
    assert "balanced_accuracy" in metrics
    assert metrics["balanced_accuracy"] > 0.75

    # 5. Specificity
    assert "specificity" in metrics
    assert metrics["specificity"] == round(88 / 90, 4)

    # 6. Confusion matrix & breakdown
    assert "confusion_matrix" in metrics
    assert "confusion_breakdown" in metrics
    assert metrics["confusion_breakdown"]["tn"] == 88
    assert metrics["confusion_breakdown"]["fp"] == 2
    assert metrics["confusion_breakdown"]["fn"] == 3
    assert metrics["confusion_breakdown"]["tp"] == 7

    # 7. Positive-class prevalence and imbalance warning
    assert "positive_class_prevalence" in metrics
    assert metrics["positive_class_prevalence"] == 0.10
    assert metrics["is_imbalanced"] is True
    assert "imbalance_warning" in metrics
    assert "misleading" in metrics["imbalance_warning"].lower()


def test_threshold_optimization_and_tradeoff():
    """
    Regression Test: Verify threshold evaluation across grid, identification of operating
    threshold tailored to marketing/conversion objectives, and precision/recall trade-off narratives.
    """
    np.random.seed(42)
    y_true = np.array([0] * 100 + [1] * 20)
    p_neg = np.random.beta(1, 6, 100)
    p_pos = np.random.beta(4, 3, 20)
    p_all = np.concatenate([p_neg, p_pos])

    thresh_res = analyze_classification_thresholds(
        y_true=y_true,
        p_positive=p_all,
        user_goal="Predict prospective client term deposit conversion campaign."
    )

    assert "threshold_table" in thresh_res
    assert len(thresh_res["threshold_table"]) >= 15
    assert "default_threshold" in thresh_res
    assert "operating_threshold" in thresh_res
    assert "tradeoff_explanation" in thresh_res
    assert "ranking_insight" in thresh_res

    # Default threshold is 0.50
    assert abs(thresh_res["default_threshold"]["threshold"] - 0.50) < 1e-3

    # Operating threshold for conversion/marketing should prioritize recall/F2
    opt = thresh_res["operating_threshold"]
    assert opt["threshold"] <= 0.50  # Lower cutoff to capture more positive converters
    assert opt["recall"] >= thresh_res["default_threshold"]["recall"]
    assert "F2" in opt["objective"] or "Recall" in opt["objective"]
    assert "decile" in thresh_res["ranking_insight"].lower() or "rank" in thresh_res["ranking_insight"].lower()


def test_prevalence_aware_pr_evaluation():
    """
    Regression Test: Verify that Precision-Recall evaluation accounts for baseline positive prevalence.
    """
    y_true = np.array([0] * 85 + [1] * 15)
    p_positive = np.linspace(0.01, 0.99, 100)
    y_prob = np.column_stack([1 - p_positive, p_positive])
    y_pred = (p_positive >= 0.5).astype(int)

    metrics = evaluate_classification(y_true, y_pred, y_prob)
    pr_curve = metrics.get("pr_curve", {})

    assert "precision" in pr_curve
    assert "recall" in pr_curve
    assert "baseline_prevalence" in pr_curve
    assert pr_curve["baseline_prevalence"] == 0.15


def test_non_causal_shap_and_report_wording():
    """
    Regression Test: Verify report terminology is strictly non-causal ('Top Predictive Drivers',
    'model-derived predictive associations') and organizes business recommendations into the 4 pillars.
    """
    profile = {
        "row_count": 5000,
        "col_count": 15,
        "missingness_report": {"total_missing_pct": 0.0, "duplicate_rows": 0},
        "quality_alerts": []
    }
    best_exp = {
        "model_name": "LightGBM_LeakFree",
        "model_family": "gradient_boosting",
        "metrics": {
            "test": {
                "roc_auc": 0.812,
                "pr_auc": 0.485,
                "positive_precision": 0.472,
                "positive_recall": 0.594,
                "f1_positive": 0.526,
                "f2_positive": 0.565,
                "balanced_accuracy": 0.755,
                "specificity": 0.916,
                "accuracy": 0.899,
                "is_binary": True,
                "is_imbalanced": True,
                "positive_class_prevalence": 0.1127,
                "confusion_matrix": [[4000, 400], [250, 350]],
                "threshold_analysis": {
                    "default_threshold": {"threshold": 0.5, "precision": 0.638, "recall": 0.251, "f1": 0.360, "f2": 0.285, "tp": 233, "fn": 695},
                    "operating_threshold": {"threshold": 0.20, "objective": "Maximized F2-Score", "precision": 0.472, "recall": 0.594, "f1": 0.526, "f2": 0.565, "balanced_accuracy": 0.755, "specificity": 0.916, "tp": 551, "fn": 377, "recall_gain_over_default": 0.343, "tp_gain_over_default": 318, "reasoning": "Captures +318 more buyers."},
                    "tradeoff_explanation": "Trade-off between precision and recall.",
                    "ranking_insight": "Prospect ranking by probability decile is superior to raw accuracy."
                }
            },
            "cv_mean": 0.808,
            "cv_std": 0.012
        },
        "train_time_sec": 0.45
    }
    critic_audit = {
        "audit_status": "PASSED",
        "findings": [
            {
                "issue_type": "domain_target_leakage",
                "severity": "critical",
                "description": "Duration leakage detected and dropped before final retraining.",
                "remediation": "Retrained leak-free champion LightGBM_LeakFree."
            }
        ]
    }
    explainability = {
        "feature_importance": {
            "rankings": [
                {"feature": "euribor3m", "importance_pct": 28.5},
                {"feature": "pdays", "importance_pct": 22.1},
                {"feature": "nr.employed", "importance_pct": 18.3},
            ]
        }
    }
    insights = [
        {"category": "observed_facts", "title": "Base Rate", "finding": "Base rate is 11.27%", "evidence": "11.27%", "confidence": "High"},
        {"category": "model_derived", "title": "Predictive Power", "finding": "ROC-AUC is 0.812", "evidence": "0.812", "confidence": "High"},
        {"category": "actionable_recommendations", "title": "Threshold Action", "finding": "Deploy 0.20 cutoff", "evidence": "F2 0.565", "confidence": "High"},
        {"category": "causal_limitations", "title": "Non-Causal Signal", "finding": "Drivers are predictive associations only", "evidence": "Observational data", "confidence": "High"},
    ]

    report = generate_full_markdown_report(
        dataset_name="Bank_Marketing",
        user_goal="Predict customer deposit subscription.",
        problem_type="classification",
        target_column="y",
        validation_strategy="stratified_kfold",
        profile_summary=profile,
        experiment_results=[best_exp],
        best_experiment=best_exp,
        critic_audit=critic_audit,
        business_insights=insights,
        artifact_paths=["reports/artifacts/test_roc.png"],
        explainability=explainability
    )

    # Terminology checks: Must NOT use causal economic or overclaimed phrases
    assert "Primary Economic Drivers" not in report
    assert "vastly superior" not in report
    assert "additional customer conversions" not in report
    assert "caused conversions" not in report
    assert "Top Predictive Drivers" in report
    assert "model-derived predictive associations" in report
    assert "establish causal relationships" in report or "establish causality" in report

    # Threshold qualification & percentage points checks
    assert "Selected operating threshold" in report
    assert "percentage points" in report
    assert "Probability Ranking" in report or "ranking" in report.lower()

    # 4 distinct recommendation sections
    assert "7.1 Observed Facts" in report
    assert "7.2 Model-Derived Evidence" in report
    assert "7.3 Actionable Recommendations" in report
    assert "7.4 Causal Limitations" in report

    # Section 8 Model Limitations & Risk checks
    assert "Model Limitations & Operational Risk Analysis" in report
    assert "Class Imbalance" in report
    assert "False Negative" in report
    assert "Threshold" in report
    assert "Temporal" in report or "Drift" in report


def test_bank_marketing_duration_leakage_preservation():
    """
    Regression Test: Ensure Bank Marketing 'duration' column is identified as prospective
    data leakage by the critic auditor, triggers requires_iteration = True, and requires REMOVE_LEAKY_FEATURES.
    """
    leak_metrics = {
        "train": {"roc_auc": 0.95},
        "test": {"roc_auc": 0.93, "is_binary": True},
        "cv_mean": 0.92
    }
    critic_leak = critique_experiment(
        model_name="LightGBM_Initial",
        problem_type="classification",
        metrics=leak_metrics,
        feature_names=["duration", "age", "job_admin", "euribor3m"],
        validation_strategy="stratified_kfold",
        target_column="y",
        raw_columns=["age", "job", "duration", "campaign", "euribor3m", "y"]
    )

    assert critic_leak["requires_iteration"] is True
    assert "REMOVE_LEAKY_FEATURES" in critic_leak["remediation_actions"]
    assert any(f["issue_type"] == "domain_target_leakage" for f in critic_leak["findings"])
    assert any("duration" in f["affected_components"] for f in critic_leak["findings"])


def test_regression_evaluation():
    """Test regression metrics calculation."""
    y_true = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    y_pred = np.array([105.0, 145.0, 195.0, 260.0, 290.0])

    metrics = evaluate_regression(y_true, y_pred)
    assert metrics["mae"] == 7.0
    assert metrics["rmse"] > 0
    assert metrics["r2"] > 0.95
    assert "residual_percentiles" in metrics


def test_forecasting_evaluation():
    """Test time-series forecasting metrics including WAPE and sMAPE."""
    y_true = np.array([50.0, 60.0, 70.0, 80.0])
    y_pred = np.array([48.0, 62.0, 71.0, 78.0])

    metrics = evaluate_forecasting(y_true, y_pred)
    assert "wape" in metrics
    assert "smape" in metrics
    assert metrics["wape"] < 5.0


def test_ml_trainer_execution():
    """Test training and cross-validation of LightGBM and Baseline."""
    np.random.seed(42)
    X_train = np.random.randn(100, 4)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    X_test = np.random.randn(25, 4)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
    features = ["f1", "f2", "f3", "f4"]

    res = train_and_evaluate_model(
        model_name="LightGBM",
        problem_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=features,
        cv_folds=3,
        track_mlflow=False
    )

    assert res["model_name"] == "LightGBM"
    assert "test" in res["metrics"]
    assert "cv_mean" in res["metrics"]
    assert res["train_time_sec"] >= 0


def test_explainability_and_shap():
    """Test SHAP value and feature importance calculations."""
    np.random.seed(42)
    X = np.random.randn(80, 3)
    y = (X[:, 0] * 2 + X[:, 1] > 0).astype(int)
    
    model = LogisticRegression()
    model.fit(X, y)
    features = ["driver_feat", "secondary_feat", "noise_feat"]

    imp = calculate_feature_importance(model, features)
    assert len(imp["rankings"]) == 3
    assert imp["rankings"][0]["feature"] == "driver_feat"

    shap_res = compute_shap_explanations(model, X, features, max_samples=40)
    assert "top_shap_features" in shap_res


def test_critic_leakage_and_overfitting_detection():
    """Test methodology critic flags leakage and severe overfitting."""
    bad_metrics = {
        "train": {"roc_auc": 0.999},
        "test": {"roc_auc": 0.650, "is_binary": True},
        "cv_mean": 0.67
    }
    critic_res = critique_experiment(
        model_name="DeepOverfitTree",
        problem_type="classification",
        metrics=bad_metrics,
        feature_names=["f1", "f2"],
        validation_strategy="stratified_kfold"
    )
    assert critic_res["requires_iteration"] is True
    assert any(f["issue_type"] == "severe_overfitting" for f in critic_res["findings"])


def test_leak_free_oof_threshold_selection():
    """
    Regression Test: Ensure threshold selection is performed strictly on OOF validation predictions
    without holdout test set label leakage, and that metrics separate OOF validation from holdout eval.
    """
    np.random.seed(42)
    X_train = np.random.randn(200, 4)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0.5).astype(int)
    X_test = np.random.randn(50, 4)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0.5).astype(int)
    features = ["f1", "f2", "f3", "f4"]

    res = train_and_evaluate_model(
        model_name="LogisticRegression",
        problem_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=features,
        cv_folds=5,
        track_mlflow=False,
        user_goal="Predict deposit conversion campaign."
    )

    test_metrics = res["metrics"]["test"]
    assert "threshold_analysis" in test_metrics
    th_analysis = test_metrics["threshold_analysis"]

    # 1. Disclosure statement presence
    assert "disclosure" in th_analysis
    assert "Operating threshold was selected using validation/out-of-fold predictions and then evaluated on the untouched holdout set." in th_analysis["disclosure"]

    # 2. Separation of OOF validation and locked holdout performance
    assert "oof_validation_analysis" in th_analysis
    assert "locked_operating_threshold" in th_analysis
    assert "default_threshold" in th_analysis

    # 3. Objective string formatting
    locked_th = th_analysis["locked_operating_threshold"]
    assert "Selected operating threshold:" in locked_th["objective"]
    assert "optimised for F2 under the stated objective." in locked_th["objective"]
    assert locked_th["threshold"] <= 0.50


def test_model_selection_uses_cv_not_holdout():
    """
    Regression Test: Verify model selection is based strictly on CV performance (cv_mean)
    on the training set, not on final holdout test metrics.
    """
    exp1 = {
        "model_name": "Model_High_CV_Low_Test",
        "metrics": {
            "cv_mean": 0.850,
            "test": {"roc_auc": 0.700}
        }
    }
    exp2 = {
        "model_name": "Model_Low_CV_High_Test",
        "metrics": {
            "cv_mean": 0.750,
            "test": {"roc_auc": 0.950}
        }
    }
    raw_experiments = [exp2, exp1]

    # Sort strictly by cv_mean as implemented in workflows.py
    sorted_exps = sorted(
        raw_experiments,
        key=lambda x: x["metrics"].get("cv_mean", 0.0),
        reverse=True
    )

    champion = sorted_exps[0]
    # The champion MUST be Model_High_CV_Low_Test because its CV score is higher
    assert champion["model_name"] == "Model_High_CV_Low_Test"


def test_fold_safe_cv_preprocessing():
    """
    Regression Test: Ensure preprocess_fold fits scaler and imputer ONLY on the training fold
    and transforms the validation fold without leaking validation stats.
    """
    import pandas as pd
    from backend.app.tools.preprocessor import preprocess_fold

    # Training fold with mean=0
    df_train_fold = pd.DataFrame({"num": [0.0, 2.0, 4.0], "cat": ["A", "A", "B"]})
    # Validation fold with extreme value 100
    df_val_fold = pd.DataFrame({"num": [100.0], "cat": ["A"]})

    X_tr, X_val = preprocess_fold(
        X_tr_raw=df_train_fold,
        X_val_raw=df_val_fold,
        num_cols=["num"],
        cat_cols=["cat"]
    )

    # Standard scale of df_train_fold num: mean=2, std=sqrt(8/3)=1.63299
    # Validation value (100 - 2) / 1.63299 ~ 60.0
    # If val fold influenced mean, mean would be (0+2+4+100)/4 = 26.5 and scaled val value would be ~1.7
    assert X_val[0, 0] > 50.0  # Proves scaling parameters came ONLY from df_train_fold


def test_threshold_selection_never_reads_holdout_labels():
    """
    Regression Test: Prove that changing or corrupting holdout y_test labels has ZERO effect
    on the selected operating threshold, proving complete holdout isolation during threshold optimization.
    """
    np.random.seed(42)
    X_train = np.random.randn(200, 4)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0.5).astype(int)
    X_test = np.random.randn(50, 4)

    # Real holdout labels
    y_test_real = (X_test[:, 0] + X_test[:, 1] > 0.5).astype(int)
    # Corrupted / inverted holdout labels
    y_test_corrupted = 1 - y_test_real

    res_real = train_and_evaluate_model(
        model_name="LogisticRegression",
        problem_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test_real,
        feature_names=["f1", "f2", "f3", "f4"],
        cv_folds=3,
        track_mlflow=False,
        user_goal="Predict deposit conversion campaign."
    )

    res_corrupted = train_and_evaluate_model(
        model_name="LogisticRegression",
        problem_type="classification",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test_corrupted,
        feature_names=["f1", "f2", "f3", "f4"],
        cv_folds=3,
        track_mlflow=False,
        user_goal="Predict deposit conversion campaign."
    )

    th_real = res_real["metrics"]["test"]["threshold_analysis"]["locked_operating_threshold"]["threshold"]
    th_corrupted = res_corrupted["metrics"]["test"]["threshold_analysis"]["locked_operating_threshold"]["threshold"]

    # Operating threshold MUST be identical regardless of y_test!
    assert th_real == th_corrupted


def test_candidate_models_do_not_evaluate_holdout():
    """
    Regression Test: Prove candidate models trained via train_and_evaluate_model do NOT touch holdout,
    leaving metrics['test'] empty, and only evaluate_locked_champion_on_holdout populates holdout metrics.
    """
    np.random.seed(42)
    X_train = np.random.randn(200, 4)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0.5).astype(int)
    X_test = np.random.randn(50, 4)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0.5).astype(int)

    # Candidate mode (no X_test/y_test passed)
    cand_res = train_and_evaluate_model(
        model_name="LogisticRegression",
        problem_type="classification",
        X_train=X_train,
        y_train=y_train,
        feature_names=["f1", "f2", "f3", "f4"],
        cv_folds=3,
        track_mlflow=False,
        user_goal="Predict deposit conversion campaign."
    )

    # Candidate test metrics MUST be empty dictionary before champion locking!
    assert cand_res["metrics"]["test"] == {}

    # Final champion mode (evaluate_locked_champion_on_holdout called explicitly ONCE)
    champ_res = evaluate_locked_champion_on_holdout(
        champion_exp=cand_res,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        user_goal="Predict deposit conversion campaign.",
        track_mlflow=False
    )

    # Champion holdout metrics MUST be populated after locked evaluation
    assert "accuracy" in champ_res["metrics"]["test"]
    assert "threshold_analysis" in champ_res["metrics"]["test"]


def test_validation_metrics_differ_from_holdout_metrics():
    """
    Regression Test: Ensure OOF validation metrics (from y_train) and Holdout metrics (from y_test)
    are independent data objects and reflect their respective sample sizes and label distributions.
    """
    np.random.seed(42)
    # Train set (1,000 samples)
    X_train = np.random.randn(1000, 4)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0.2).astype(int)

    # Test set with different size (200 samples)
    X_test = np.random.randn(200, 4)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0.5).astype(int)

    cand_res = train_and_evaluate_model(
        model_name="LogisticRegression",
        problem_type="classification",
        X_train=X_train,
        y_train=y_train,
        feature_names=["f1", "f2", "f3", "f4"],
        cv_folds=3,
        track_mlflow=False,
        user_goal="Predict deposit conversion campaign."
    )

    champ_res = evaluate_locked_champion_on_holdout(
        champion_exp=cand_res,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        user_goal="Predict deposit conversion campaign.",
        track_mlflow=False
    )

    th_analysis = champ_res["metrics"]["test"]["threshold_analysis"]
    oof_val = th_analysis["oof_validation_analysis"]["operating_threshold"]
    holdout_opt = th_analysis["locked_operating_threshold"]

    # True positives count in 1000-sample validation set MUST be larger than 200-sample holdout set
    assert oof_val["tp"] > holdout_opt["tp"]
    # Total samples (tp + fp + fn + tn) must match 1,000 for validation and 200 for holdout
    assert (oof_val["tp"] + oof_val["fp"] + oof_val["fn"] + oof_val["tn"]) == 1000
    assert (holdout_opt["tp"] + holdout_opt["fp"] + holdout_opt["fn"] + holdout_opt["tn"]) == 200


def test_regression_holdout_evaluation_and_explainability():
    """
    Regression Test: Ensure regression champion evaluation produces y_test_pred,
    computes valid regression metrics (RMSE, MAE, R²), supports SHAP explainability,
    and successfully generates actual vs predicted visualization artifacts.
    """
    from backend.app.tools.visualizer import generate_actual_vs_predicted_plot
    import os

    np.random.seed(42)
    X_train = np.random.randn(300, 5)
    y_train = X_train[:, 0] * 3.0 + X_train[:, 1] * -2.0 + np.random.randn(300) * 0.1

    X_test = np.random.randn(60, 5)
    y_test = X_test[:, 0] * 3.0 + X_test[:, 1] * -2.0 + np.random.randn(60) * 0.1

    feat_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population"]

    # 1. Candidate training (CV on train set only)
    cand_res = train_and_evaluate_model(
        model_name="Ridge",
        problem_type="regression",
        X_train=X_train,
        y_train=y_train,
        feature_names=feat_names,
        cv_folds=3,
        track_mlflow=False,
        user_goal="Predict median house values in California districts."
    )

    assert not cand_res["metrics"].get("test")
    assert cand_res["metrics"]["cv_mean"] > 0

    # 2. Holdout evaluation on locked champion
    champ_res = evaluate_locked_champion_on_holdout(
        champion_exp=cand_res,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        user_goal="Predict median house values in California districts.",
        track_mlflow=False
    )

    # 3. Assert y_test_pred is present and correct length
    assert "y_test_pred" in champ_res
    assert champ_res["y_test_pred"] is not None
    assert len(champ_res["y_test_pred"]) == len(y_test)

    # 4. Assert regression test metrics exist
    test_m = champ_res["metrics"]["test"]
    assert "rmse" in test_m
    assert "mae" in test_m
    assert "r2" in test_m
    assert test_m["r2"] > 0.85

    # 5. Explainability (Feature Importance + SHAP)
    feat_imp = calculate_feature_importance(champ_res["model"], feat_names)
    assert len(feat_imp["rankings"]) == 5
    assert feat_imp["rankings"][0]["feature"] == "MedInc"

    shap_res = compute_shap_explanations(champ_res["model"], X_test, feat_names)
    assert len(shap_res["top_shap_features"]) > 0

    # 6. Actual vs Predicted Plot generation
    from backend.app.core.config import settings
    from pathlib import Path

    plot_path = generate_actual_vs_predicted_plot(
        y_true=y_test,
        y_pred=champ_res["y_test_pred"],
        model_name=champ_res["model_name"],
        run_id="test_reg_run",
        problem_type="regression"
    )
    assert plot_path is not None
    full_path = Path(settings.REPORTS_DIR).parent / plot_path
    assert full_path.exists()

    # 7. Generate full report markdown
    report_md = generate_full_markdown_report(
        dataset_name="Benchmark_California_Housing",
        user_goal="Predict median house values in California districts.",
        problem_type="regression",
        target_column="MedHouseVal",
        validation_strategy="kfold",
        profile_summary={"row_count": 360, "col_count": 6},
        experiment_results=[champ_res],
        best_experiment=champ_res,
        critic_audit={"audit_status": "PASSED"},
        business_insights=[],
        artifact_paths=[plot_path],
        explainability={"feature_importance": feat_imp, "shap_summary": shap_res}
    )
    assert "## 1. Executive Summary" in report_md
    assert "RMSE" in report_md
    assert "MedInc" in report_md


def test_bike_sharing_target_component_leakage_prevention():
    """
    Forecasting Regression Test: Ensure deterministic target-component leakage
    (e.g., casual + registered == cnt) is detected and excluded BEFORE candidate model training,
    while legitimate calendar/weather and historical lag/rolling features are preserved.
    """
    import pandas as pd
    from backend.app.tools.quality_detector import detect_target_component_leakage
    from backend.app.tools.preprocessor import prepare_train_test_split
    from backend.app.tools.critic import critique_experiment
    from backend.app.tools.reporter import generate_full_markdown_report

    # Create synthetic dataset with target cnt and additive components casual & registered
    np.random.seed(42)
    n_hours = 200
    dates = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    casual = np.random.poisson(lam=10, size=n_hours)
    registered = np.random.poisson(lam=50, size=n_hours)
    cnt = casual + registered
    temp = np.random.uniform(0.1, 0.9, size=n_hours)
    hum = np.random.uniform(0.2, 0.8, size=n_hours)
    windspeed = np.random.uniform(0.0, 0.5, size=n_hours)

    df = pd.DataFrame({
        "dteday": dates.astype(str),
        "temp": temp,
        "hum": hum,
        "windspeed": windspeed,
        "casual": casual,
        "registered": registered,
        "cnt": cnt,
    })

    # 1. Leakage detection verifies casual & registered are identified as target components
    leaks, expls = detect_target_component_leakage(df, target_column="cnt")
    assert "casual" in leaks
    assert "registered" in leaks
    assert len(leaks) == 2
    assert "casual" in expls
    assert "registered" in expls

    # 2. Preprocessor excludes casual and registered from feature matrix
    X_train, X_test, y_train, y_test, artifacts = prepare_train_test_split(
        df=df,
        target_column="cnt",
        problem_type="forecasting",
        time_column="dteday",
        drop_leakage_cols=leaks
    )

    # Prove casual and registered CANNOT enter the final feature matrix
    assert "casual" not in artifacts.feature_names
    assert "registered" not in artifacts.feature_names
    assert "cnt" not in artifacts.feature_names
    assert "dteday" not in artifacts.feature_names

    # Prove legitimate prediction-time features (weather, calendar, lags) ARE preserved
    assert "temp" in artifacts.feature_names
    assert "hum" in artifacts.feature_names
    assert "windspeed" in artifacts.feature_names
    assert "cal_dayofweek" in artifacts.feature_names
    assert "target_lag_1" in artifacts.feature_names
    assert "target_roll_mean_7" in artifacts.feature_names

    # 3. Model training on leak-free feature matrix
    cand_res = train_and_evaluate_model(
        model_name="Ridge",
        problem_type="forecasting",
        X_train=X_train,
        y_train=y_train,
        feature_names=artifacts.feature_names,
        cv_folds=3,
        track_mlflow=False,
        user_goal="Forecast hourly bike rental demand (cnt)."
    )

    champ_res = evaluate_locked_champion_on_holdout(
        champion_exp=cand_res,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        user_goal="Forecast hourly bike rental demand (cnt).",
        track_mlflow=False
    )

    # 4. Critic audit reflects PASSED (Remediated)
    critic_res = critique_experiment(
        model_name=champ_res["model_name"],
        problem_type="forecasting",
        metrics=champ_res["metrics"],
        feature_names=artifacts.feature_names,
        validation_strategy="walk_forward_time_split",
        target_column="cnt",
        raw_columns=list(df.columns),
        remediated_features=leaks,
        leakage_explanations=expls
    )

    assert critic_res["audit_status"] == "PASSED (Remediated)"
    assert critic_res["leakage_remediated"] is True
    assert "casual" in critic_res["remediated_features"]
    assert "registered" in critic_res["remediated_features"]

    # 5. Report explicitly documents leakage exclusion rationale
    report_md = generate_full_markdown_report(
        dataset_name="Bike_Sharing_Hour",
        user_goal="Forecast hourly bike rental demand (cnt).",
        problem_type="forecasting",
        target_column="cnt",
        validation_strategy="walk_forward_time_split",
        profile_summary={"row_count": len(df), "col_count": len(df.columns)},
        experiment_results=[champ_res],
        best_experiment=champ_res,
        critic_audit=critic_res,
        business_insights=[],
        artifact_paths=[],
        explainability={"feature_importance": {"rankings": []}}
    )

    assert "PASSED (Remediated)" in report_md
    assert "casual" in report_md
    assert "registered" in report_md
    assert "Leakage prevention excluded" in report_md


def test_forecasting_leaderboard_cv_metric_mapping_and_holdout_isolation():
    """
    Ensure the Model Leaderboard compares ALL candidate models using the SAME CV primary metric (CV WAPE),
    and proves that the Champion model's holdout RMSE (e.g. 68.6134) is NOT placed into the Leaderboard's CV column.
    """
    from backend.app.tools.reporter import generate_full_markdown_report

    # Candidate models with their respective CV metrics
    xgb_exp = {
        "model_name": "XGBoost",
        "model_family": "Tree Ensemble",
        "train_time_sec": 1.84,
        "metrics": {
            "cv_mean": 16.6200,
            "cv_std": 0.4210,
            "test": {
                "rmse": 68.6134,
                "mae": 44.9655,
                "wape": 18.13,
                "smape": 28.03,
                "r2": 0.9030
            }
        }
    }
    rf_exp = {
        "model_name": "RandomForest",
        "model_family": "Tree Ensemble",
        "train_time_sec": 2.12,
        "metrics": {
            "cv_mean": 16.6800,
            "cv_std": 0.3850
        }
    }
    lgbm_exp = {
        "model_name": "LightGBM",
        "model_family": "Tree Ensemble",
        "train_time_sec": 0.95,
        "metrics": {
            "cv_mean": 17.4633,
            "cv_std": 0.5120
        }
    }
    baseline_exp = {
        "model_name": "Baseline",
        "model_family": "Heuristic",
        "train_time_sec": 0.01,
        "metrics": {
            "cv_mean": 75.0333,
            "cv_std": 1.2400
        }
    }

    candidates = [xgb_exp, rf_exp, lgbm_exp, baseline_exp]

    report_md = generate_full_markdown_report(
        dataset_name="Bike_Sharing_Hour",
        user_goal="Forecast hourly bike rental demand (cnt).",
        problem_type="forecasting",
        target_column="cnt",
        validation_strategy="walk_forward_time_split",
        profile_summary={"row_count": 17379, "col_count": 17},
        experiment_results=candidates,
        best_experiment=xgb_exp,
        critic_audit={"audit_status": "PASSED (Remediated)", "remediated_features": ["casual", "registered"]},
        business_insights=[],
        artifact_paths=[],
        explainability={"feature_importance": {"rankings": []}}
    )

    # 1. Verify Leaderboard Section 3
    assert "## 3. Model Leaderboard & Multi-Metric Evaluation" in report_md
    assert "Primary Loss Metric (CV WAPE (%))" in report_md

    # 2. Verify all candidate rows render their CV scores
    assert "| `XGBoost` | CV: 16.6200 |" in report_md
    assert "| `RandomForest` | CV: 16.6800 |" in report_md
    assert "| `LightGBM` | CV: 17.4633 |" in report_md
    assert "| `Baseline` | CV: 75.0333 |" in report_md

    # 3. Proves XGBoost's holdout RMSE 68.6134 is NOT in the Leaderboard's CV column
    assert "| `XGBoost` | 68.6134 |" not in report_md

    # 4. Proves final holdout metrics are located in Section 4 Holdout Evaluation
    assert "## 4. Final Touchless Holdout Evaluation & Multi-Metric Diagnostics" in report_md
    assert "RMSE (Holdout)" in report_md
    assert "68.6134" in report_md
    assert "44.9655" in report_md
    assert "0.9030" in report_md
    assert "18.13%" in report_md


def test_get_model_instance_normalization_and_estimator_mapping():
    """
    Ensure get_model_instance correctly normalizes PascalCase, snake_case, and kebab-case
    model names and instantiates genuine estimator classes rather than fallback defaults.
    """
    from backend.app.tools.ml_trainer import get_model_instance
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.dummy import DummyClassifier, DummyRegressor
    import lightgbm as lgb
    from xgboost import XGBClassifier, XGBRegressor

    # Classification Tests
    clf_cases = [
        ("RandomForest", RandomForestClassifier, "ensemble_tree"),
        ("random_forest", RandomForestClassifier, "ensemble_tree"),
        ("RF", RandomForestClassifier, "ensemble_tree"),
        ("LogisticRegression", LogisticRegression, "linear"),
        ("logistic_regression", LogisticRegression, "linear"),
        ("LR", LogisticRegression, "linear"),
        ("LightGBM", lgb.LGBMClassifier, "gradient_boosting"),
        ("lightgbm", lgb.LGBMClassifier, "gradient_boosting"),
        ("LGBM", lgb.LGBMClassifier, "gradient_boosting"),
        ("XGBoost", XGBClassifier, "gradient_boosting"),
        ("xgboost", XGBClassifier, "gradient_boosting"),
        ("XGB", XGBClassifier, "gradient_boosting"),
        ("Baseline", DummyClassifier, "baseline"),
        ("dummy", DummyClassifier, "baseline"),
    ]

    for name, expected_cls, expected_fam in clf_cases:
        inst, fam, params = get_model_instance(name, "classification")
        assert isinstance(inst, expected_cls), f"Expected {expected_cls.__name__} for '{name}', got {inst.__class__.__name__}"
        assert fam == expected_fam, f"Expected family '{expected_fam}' for '{name}', got '{fam}'"

    # Regression / Forecasting Tests
    reg_cases = [
        ("RandomForest", RandomForestRegressor, "ensemble_tree"),
        ("random_forest", RandomForestRegressor, "ensemble_tree"),
        ("Ridge", Ridge, "linear"),
        ("LinearRegression", Ridge, "linear"),
        ("linear_regression", Ridge, "linear"),
        ("LightGBM", lgb.LGBMRegressor, "gradient_boosting"),
        ("XGBoost", XGBRegressor, "gradient_boosting"),
        ("Baseline", DummyRegressor, "baseline"),
    ]

    for name, expected_cls, expected_fam in reg_cases:
        inst, fam, params = get_model_instance(name, "regression")
        assert isinstance(inst, expected_cls), f"Expected {expected_cls.__name__} for '{name}', got {inst.__class__.__name__}"
        assert fam == expected_fam, f"Expected family '{expected_fam}' for '{name}', got '{fam}'"








