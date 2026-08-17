"""
AutoDS Machine Learning Trainer Tool
Trains genuine, reproducible ML models, executes cross-validation, and tracks runs via MLflow.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import lightgbm as lgb
import mlflow
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.tools.evaluator import evaluate_classification, evaluate_forecasting, evaluate_regression


def initialize_mlflow():
    """Ensure MLflow tracking URI and default experiment are initialized."""
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    except Exception as e:
        logger.warning(f"MLflow initialization warning: {e}")


def get_model_instance(model_name: str, problem_type: str, random_state: int = 42) -> Tuple[Any, str, Dict[str, Any]]:
    """Instantiate model, family, and hyperparameter dict."""
    name = model_name.lower().strip()

    if problem_type == "classification":
        if name in ("dummy", "baseline", "dummy_classifier"):
            return DummyClassifier(strategy="prior"), "baseline", {"strategy": "prior"}
        elif name in ("logistic_regression", "lr", "linear"):
            params = {"max_iter": 1000, "C": 1.0, "random_state": random_state}
            return LogisticRegression(**params), "linear", params
        elif name in ("random_forest", "rf", "random_forest_classifier"):
            params = {"n_estimators": 100, "max_depth": 10, "random_state": random_state, "n_jobs": -1}
            return RandomForestClassifier(**params), "ensemble_tree", params
        elif name in ("lightgbm", "lgbm", "lightgbm_classifier"):
            params = {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31, "verbose": -1, "random_state": random_state, "n_jobs": -1}
            return lgb.LGBMClassifier(**params), "gradient_boosting", params
        elif name in ("xgboost", "xgb", "xgboost_classifier"):
            params = {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 6, "eval_metric": "logloss", "random_state": random_state, "n_jobs": -1}
            return XGBClassifier(**params), "gradient_boosting", params
        elif name in ("gradient_boosting", "gbm"):
            params = {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 5, "random_state": random_state}
            return GradientBoostingClassifier(**params), "gradient_boosting", params
        else:
            # Default to LightGBM
            params = {"n_estimators": 100, "learning_rate": 0.05, "verbose": -1, "random_state": random_state}
            return lgb.LGBMClassifier(**params), "gradient_boosting", params

    elif problem_type in ("regression", "forecasting"):
        if name in ("dummy", "baseline", "dummy_regressor"):
            return DummyRegressor(strategy="mean"), "baseline", {"strategy": "mean"}
        elif name in ("ridge", "linear_regression", "linear"):
            params = {"alpha": 1.0}
            return Ridge(**params), "linear", params
        elif name in ("random_forest", "rf", "random_forest_regressor"):
            params = {"n_estimators": 100, "max_depth": 10, "random_state": random_state, "n_jobs": -1}
            return RandomForestRegressor(**params), "ensemble_tree", params
        elif name in ("lightgbm", "lgbm", "lightgbm_regressor"):
            params = {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31, "verbose": -1, "random_state": random_state, "n_jobs": -1}
            return lgb.LGBMRegressor(**params), "gradient_boosting", params
        elif name in ("xgboost", "xgb", "xgboost_regressor"):
            params = {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 6, "random_state": random_state, "n_jobs": -1}
            return XGBRegressor(**params), "gradient_boosting", params
        elif name in ("gradient_boosting", "gbm"):
            params = {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 5, "random_state": random_state}
            return GradientBoostingRegressor(**params), "gradient_boosting", params
        else:
            params = {"n_estimators": 100, "learning_rate": 0.05, "verbose": -1, "random_state": random_state}
            return lgb.LGBMRegressor(**params), "gradient_boosting", params

    raise ValueError(f"Unsupported problem_type: {problem_type}")


def train_and_evaluate_model(
    model_name: str,
    problem_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    cv_folds: int = 5,
    random_state: int = 42,
    track_mlflow: bool = True
) -> Dict[str, Any]:
    """
    Train a model, compute cross-validation and test set metrics, and log to MLflow.
    Returns model instance, metrics, training time, and parameters.
    """
    initialize_mlflow()
    model, family, params = get_model_instance(model_name, problem_type, random_state)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time_sec = round(time.time() - start_time, 3)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Probabilities for classification
    y_train_prob = None
    y_test_prob = None
    if problem_type == "classification" and hasattr(model, "predict_proba"):
        try:
            y_train_prob = model.predict_proba(X_train)
            y_test_prob = model.predict_proba(X_test)
        except Exception:
            pass

    # Metric Evaluation
    if problem_type == "classification":
        train_metrics = evaluate_classification(y_train, y_train_pred, y_train_prob)
        test_metrics = evaluate_classification(y_test, y_test_pred, y_test_prob)
    elif problem_type == "regression":
        train_metrics = evaluate_regression(y_train, y_train_pred)
        test_metrics = evaluate_regression(y_test, y_test_pred)
    elif problem_type == "forecasting":
        train_metrics = evaluate_forecasting(y_train, y_train_pred)
        test_metrics = evaluate_forecasting(y_test, y_test_pred)
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")

    # Cross-validation on X_train for validation stability
    cv_scores = []
    if cv_folds > 1 and len(X_train) >= cv_folds * 2:
        if problem_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        for tr_idx, val_idx in cv.split(X_train, y_train):
            X_tr_f, X_val_f = X_train[tr_idx], X_train[val_idx]
            y_tr_f, y_val_f = y_train[tr_idx], y_train[val_idx]
            
            fold_model, _, _ = get_model_instance(model_name, problem_type, random_state)
            fold_model.fit(X_tr_f, y_tr_f)
            val_pred = fold_model.predict(X_val_f)
            
            if problem_type == "classification":
                val_prob = fold_model.predict_proba(X_val_f) if hasattr(fold_model, "predict_proba") else None
                m = evaluate_classification(y_val_f, val_pred, val_prob)
                cv_scores.append(m.get("roc_auc", m.get("accuracy", 0.0)))
            elif problem_type == "regression":
                m = evaluate_regression(y_val_f, val_pred)
                cv_scores.append(m.get("rmse", 0.0))
            elif problem_type == "forecasting":
                m = evaluate_forecasting(y_val_f, val_pred)
                cv_scores.append(m.get("wape", 0.0))

    cv_mean = round(float(np.mean(cv_scores)), 4) if cv_scores else 0.0
    cv_std = round(float(np.std(cv_scores)), 4) if cv_scores else 0.0

    combined_metrics = {
        "test": test_metrics,
        "train": train_metrics,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_scores": cv_scores,
    }

    # MLflow Tracking
    mlflow_run_id = None
    if track_mlflow:
        try:
            with mlflow.start_run(run_name=f"{model_name}_{problem_type}") as run:
                mlflow_run_id = run.info.run_id
                mlflow.log_params(params)
                mlflow.log_param("model_family", family)
                mlflow.log_param("num_features", len(feature_names))
                mlflow.log_metric("train_time_sec", train_time_sec)
                mlflow.log_metric("cv_mean", cv_mean)
                
                for k, v in test_metrics.items():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        mlflow.log_metric(f"test_{k}", v)
        except Exception as e:
            logger.debug(f"MLflow logging bypassed: {e}")

    return {
        "model": model,
        "model_name": model_name,
        "model_family": family,
        "hyperparameters": params,
        "train_time_sec": train_time_sec,
        "metrics": combined_metrics,
        "mlflow_run_id": mlflow_run_id,
        "y_test_pred": y_test_pred,
        "y_test_prob": y_test_prob,
    }
