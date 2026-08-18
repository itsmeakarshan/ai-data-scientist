"""
AutoDS Visual Diagnostics Regression Tests
Tests data-driven, dataset-agnostic generation of ROC, Precision-Recall, Confusion Matrix,
Actual vs Predicted, Residuals, and Feature Importance plots across binary, multiclass,
regression, and forecasting scenarios.
"""

from pathlib import Path
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from backend.app.core.config import settings
from backend.app.tools.evaluator import (
    evaluate_classification,
    evaluate_regression,
    evaluate_forecasting,
)
from backend.app.tools.visualizer import (
    generate_roc_pr_plots,
    generate_confusion_matrix_plot,
    generate_feature_importance_plot,
    generate_actual_vs_predicted_plot,
    generate_residual_plot,
)
from backend.app.tools.explainability import calculate_feature_importance


def test_binary_classification_visual_diagnostics():
    """Verify binary classification produces all 4 valid visual diagnostic plots."""
    np.random.seed(42)
    N = 200
    X = np.random.randn(N, 5)
    y_true = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(N) * 0.5 > 0).astype(int)
    
    clf = LogisticRegression()
    clf.fit(X, y_true)
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)
    
    # 1. Evaluation
    metrics = evaluate_classification(y_true, y_pred, y_prob, user_goal="Predict binary outcome")
    assert metrics["is_binary"] is True
    assert "roc_curve" in metrics
    assert "pr_curve" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["roc_auc"] > 0.5
    assert metrics["pr_auc"] > 0.0

    run_id = "test_bin_diag"
    # 2. ROC & PR plots
    roc_pr_paths = generate_roc_pr_plots(
        roc_data=metrics["roc_curve"],
        pr_data=metrics["pr_curve"],
        roc_auc=metrics["roc_auc"],
        pr_auc=metrics["pr_auc"],
        model_name="LogisticRegression",
        run_id=run_id
    )
    assert "roc_curve_path" in roc_pr_paths
    assert "pr_curve_path" in roc_pr_paths
    for p in roc_pr_paths.values():
        full_path = Path(settings.REPORTS_DIR).parent / p
        assert full_path.exists()
        assert full_path.stat().st_size > 500

    # 3. Confusion Matrix plot
    cm_path = generate_confusion_matrix_plot(
        cm=metrics["confusion_matrix"],
        model_name="LogisticRegression",
        run_id=run_id,
        class_labels=metrics["class_labels"]
    )
    assert (Path(settings.REPORTS_DIR).parent / cm_path).exists()

    # 4. Feature Importance plot
    feat_names = [f"feat_{i}" for i in range(5)]
    feat_imp = calculate_feature_importance(clf, feat_names)
    imp_path = generate_feature_importance_plot(
        feature_rankings=feat_imp["rankings"],
        model_name="LogisticRegression",
        run_id=run_id
    )
    assert (Path(settings.REPORTS_DIR).parent / imp_path).exists()


def test_multiclass_classification_visual_diagnostics():
    """Verify multiclass classification produces valid OvR ROC, OvR PR, multiclass CM, and Feature Importance."""
    np.random.seed(42)
    N = 300
    K = 4  # 4 classes
    X = np.random.randn(N, 6)
    y_true = np.random.choice(K, size=N, p=[0.4, 0.3, 0.2, 0.1])
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y_true)
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)
    
    # 1. Evaluation
    metrics = evaluate_classification(y_true, y_pred, y_prob, user_goal="Multiclass prediction")
    assert metrics["is_binary"] is False
    assert "roc_curve" in metrics
    assert metrics["roc_curve"]["is_multiclass"] is True
    assert "pr_curve" in metrics
    assert metrics["pr_curve"]["is_multiclass"] is True
    assert len(metrics["confusion_matrix"]) == K
    assert metrics["class_labels"] == [0, 1, 2, 3]

    run_id = "test_mc_diag"
    # 2. OvR ROC & PR plots
    roc_pr_paths = generate_roc_pr_plots(
        roc_data=metrics["roc_curve"],
        pr_data=metrics["pr_curve"],
        roc_auc=metrics["roc_auc"],
        pr_auc=metrics["pr_auc"],
        model_name="RandomForest",
        run_id=run_id
    )
    assert "roc_curve_path" in roc_pr_paths
    assert "pr_curve_path" in roc_pr_paths
    for p in roc_pr_paths.values():
        full_path = Path(settings.REPORTS_DIR).parent / p
        assert full_path.exists()
        assert full_path.stat().st_size > 500

    # 3. Multiclass Confusion Matrix plot
    cm_path = generate_confusion_matrix_plot(
        cm=metrics["confusion_matrix"],
        model_name="RandomForest",
        run_id=run_id,
        class_labels=metrics["class_labels"]
    )
    assert (Path(settings.REPORTS_DIR).parent / cm_path).exists()

    # 4. Feature Importance plot
    feat_names = [f"signal_{i}" for i in range(6)]
    feat_imp = calculate_feature_importance(clf, feat_names)
    imp_path = generate_feature_importance_plot(
        feature_rankings=feat_imp["rankings"],
        model_name="RandomForest",
        run_id=run_id
    )
    assert (Path(settings.REPORTS_DIR).parent / imp_path).exists()


def test_regression_visual_diagnostics():
    """Verify regression produces Actual vs Predicted, Residual Diagnostics, and Feature Importance."""
    np.random.seed(42)
    N = 200
    X = np.random.randn(N, 4)
    y_true = X[:, 0] * 3.0 + X[:, 1] * -2.0 + np.random.randn(N)
    
    reg = Ridge()
    reg.fit(X, y_true)
    y_pred = reg.predict(X)
    
    metrics = evaluate_regression(y_true, y_pred)
    assert "rmse" in metrics
    assert "r2" in metrics

    run_id = "test_reg_diag"
    # 1. Actual vs Predicted
    act_pred_path = generate_actual_vs_predicted_plot(
        y_true=y_true,
        y_pred=y_pred,
        model_name="Ridge",
        run_id=run_id,
        problem_type="regression"
    )
    assert (Path(settings.REPORTS_DIR).parent / act_pred_path).exists()

    # 2. Residual Diagnostics
    res_path = generate_residual_plot(
        y_true=y_true,
        y_pred=y_pred,
        model_name="Ridge",
        run_id=run_id,
        problem_type="regression"
    )
    assert (Path(settings.REPORTS_DIR).parent / res_path).exists()

    # 3. Feature Importance
    feat_names = [f"reg_feat_{i}" for i in range(4)]
    feat_imp = calculate_feature_importance(reg, feat_names)
    imp_path = generate_feature_importance_plot(
        feature_rankings=feat_imp["rankings"],
        model_name="Ridge",
        run_id=run_id
    )
    assert (Path(settings.REPORTS_DIR).parent / imp_path).exists()


def test_forecasting_visual_diagnostics():
    """Verify forecasting produces time-series trajectory and residual plots."""
    np.random.seed(42)
    steps = 100
    y_true = np.sin(np.linspace(0, 10, steps)) + np.random.randn(steps) * 0.1
    y_pred = np.sin(np.linspace(0, 10, steps)) + np.random.randn(steps) * 0.15

    metrics = evaluate_forecasting(y_true, y_pred)
    assert "wape" in metrics
    assert "smape" in metrics

    run_id = "test_forecast_diag"
    # 1. Trajectory plot
    traj_path = generate_actual_vs_predicted_plot(
        y_true=y_true,
        y_pred=y_pred,
        model_name="LightGBM_Forecast",
        run_id=run_id,
        problem_type="forecasting"
    )
    assert (Path(settings.REPORTS_DIR).parent / traj_path).exists()

    # 2. Residual plot
    res_path = generate_residual_plot(
        y_true=y_true,
        y_pred=y_pred,
        model_name="LightGBM_Forecast",
        run_id=run_id,
        problem_type="forecasting"
    )
    assert (Path(settings.REPORTS_DIR).parent / res_path).exists()


def test_graceful_skipping_when_probabilities_unavailable():
    """Verify that when predict_proba is not available, ROC/PR are skipped gracefully without error."""
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 0])
    
    # Passing y_prob=None
    metrics = evaluate_classification(y_true, y_pred, y_prob=None)
    assert metrics["is_binary"] is True
    assert "roc_curve" not in metrics
    assert "pr_curve" not in metrics
    assert "confusion_matrix" in metrics
    assert metrics.get("roc_curve") is None
