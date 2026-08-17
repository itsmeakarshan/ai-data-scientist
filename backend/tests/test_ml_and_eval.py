"""
Unit and Integration Tests for ML Trainer, Evaluators, Explainability, and Critic
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from backend.app.tools.critic import critique_experiment
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


def test_classification_evaluation():
    """Test classification metrics computation."""
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
    # Test Overfitting Alert
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

    # Test Domain Leakage (Bank Marketing duration)
    leak_metrics = {
        "train": {"roc_auc": 0.95},
        "test": {"roc_auc": 0.93, "is_binary": True},
        "cv_mean": 0.92
    }
    critic_leak = critique_experiment(
        model_name="LightGBM",
        problem_type="classification",
        metrics=leak_metrics,
        feature_names=["duration", "age", "campaign"],
        validation_strategy="stratified_kfold",
        target_column="y",
        raw_columns=["age", "job", "duration", "y"]
    )
    assert critic_leak["requires_iteration"] is True
    assert "REMOVE_LEAKY_FEATURES" in critic_leak["remediation_actions"]
