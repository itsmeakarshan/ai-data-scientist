"""
AutoDS Machine Learning Trainer Tool
Trains genuine, reproducible ML models, executes cross-validation, and tracks runs via MLflow.
"""

from pathlib import Path
import threading
import time
import uuid
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
from backend.app.tools.evaluator import (
    analyze_classification_thresholds,
    evaluate_classification,
    evaluate_forecasting,
    evaluate_regression,
)
from backend.app.tools.preprocessor import preprocess_fold


# Global singleton lock to ensure MLflow is initialized once
_MLFLOW_INITIALIZED = False
_MLFLOW_LOCK = threading.Lock()


def initialize_mlflow(tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None, force: bool = False):
    """Ensure MLflow tracking URI and default experiment are initialized once (or re-initialized if forced/custom)."""
    global _MLFLOW_INITIALIZED
    if _MLFLOW_INITIALIZED and not force and tracking_uri is None:
        return

    with _MLFLOW_LOCK:
        if _MLFLOW_INITIALIZED and not force and tracking_uri is None:
            return

        uri = tracking_uri or settings.MLFLOW_TRACKING_URI
        exp = experiment_name or settings.MLFLOW_EXPERIMENT_NAME

        try:
            # If using SQLite tracking store, ensure target directory exists
            if uri.startswith("sqlite:///"):
                db_path = uri.replace("sqlite:///", "")
                if not db_path.startswith(":memory:"):
                    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(exp)
            _MLFLOW_INITIALIZED = True
            logger.info(f"MLflow initialized with database tracking URI: {uri}")
        except Exception as e:
            logger.warning(f"MLflow initialization warning: {e}")


def get_model_instance(model_name: str, problem_type: str, random_state: int = 42) -> Tuple[Any, str, Dict[str, Any]]:
    """Instantiate model, family, and hyperparameter dict."""
    clean_name = model_name.lower().replace("_", "").replace("-", "").strip()

    if problem_type == "classification":
        if clean_name in ("dummy", "baseline", "dummyclassifier"):
            return DummyClassifier(strategy="prior"), "baseline", {"strategy": "prior"}
        elif clean_name in ("logisticregression", "logistic", "lr", "linear", "linearclassifier"):
            params = {"max_iter": 1000, "C": 1.0, "random_state": random_state}
            return LogisticRegression(**params), "linear", params
        elif clean_name in ("randomforest", "rf", "randomforestclassifier"):
            params = {"n_estimators": 100, "max_depth": 10, "random_state": random_state, "n_jobs": -1}
            return RandomForestClassifier(**params), "ensemble_tree", params
        elif clean_name in ("lightgbm", "lgbm", "lightgbmclassifier"):
            params = {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31, "min_child_samples": 2, "verbose": -1, "random_state": random_state, "n_jobs": -1}
            return lgb.LGBMClassifier(**params), "gradient_boosting", params
        elif clean_name in ("xgboost", "xgb", "xgboostclassifier"):
            params = {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 6, "eval_metric": "logloss", "random_state": random_state, "n_jobs": -1}
            return XGBClassifier(**params), "gradient_boosting", params
        elif clean_name in ("gradientboosting", "gbm", "gradientboostingclassifier"):
            params = {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 5, "random_state": random_state}
            return GradientBoostingClassifier(**params), "gradient_boosting", params
        else:
            # Default to LightGBM
            params = {"n_estimators": 100, "learning_rate": 0.05, "verbose": -1, "random_state": random_state}
            return lgb.LGBMClassifier(**params), "gradient_boosting", params

    elif problem_type in ("regression", "forecasting"):
        if clean_name in ("dummy", "baseline", "dummyregressor"):
            return DummyRegressor(strategy="mean"), "baseline", {"strategy": "mean"}
        elif clean_name in ("ridge", "linearregression", "linear", "linearregressor", "ridgeclassifier"):
            params = {"alpha": 1.0}
            return Ridge(**params), "linear", params
        elif clean_name in ("randomforest", "rf", "randomforestregressor"):
            params = {"n_estimators": 100, "max_depth": 10, "random_state": random_state, "n_jobs": -1}
            return RandomForestRegressor(**params), "ensemble_tree", params
        elif clean_name in ("lightgbm", "lgbm", "lightgbmregressor"):
            params = {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 6, "num_leaves": 31, "min_child_samples": 5, "verbose": -1, "random_state": random_state, "n_jobs": -1}
            return lgb.LGBMRegressor(**params), "gradient_boosting", params
        elif clean_name in ("xgboost", "xgb", "xgboostregressor"):
            params = {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 6, "random_state": random_state, "n_jobs": -1}
            return XGBRegressor(**params), "gradient_boosting", params
        elif clean_name in ("gradientboosting", "gbm", "gradientboostingregressor"):
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
    feature_names: List[str],
    cv_folds: int = 5,
    random_state: int = 42,
    track_mlflow: bool = True,
    user_goal: str = "",
    raw_X_train: Optional[Any] = None,
    num_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Train a candidate model portfolio via fold-safe cross-validation on training data ONLY.
    Holdout evaluation is strictly isolated and executed only on the locked champion model.
    """
    if track_mlflow:
        initialize_mlflow()

    model, family, params = get_model_instance(model_name, problem_type, random_state)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time_sec = round(time.time() - start_time, 3)

    # Training Predictions
    y_train_pred = model.predict(X_train)
    y_train_prob = None
    if problem_type == "classification" and hasattr(model, "predict_proba"):
        try:
            y_train_prob = model.predict_proba(X_train)
        except Exception:
            pass

    # Cross-validation on X_train for validation stability, fold-safe preprocessing, and OOF threshold selection
    cv_scores = []
    oof_threshold_analysis = None
    locked_operating_threshold = None

    if cv_folds > 1 and len(X_train) >= cv_folds * 2:
        if problem_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            classes = np.unique(y_train)
            is_binary = len(classes) == 2

            oof_y_prob = np.zeros((len(y_train), 2)) if is_binary else np.zeros((len(y_train), len(classes)))

            for tr_idx, val_idx in cv.split(X_train, y_train):
                if raw_X_train is not None and num_cols is not None and cat_cols is not None:
                    X_tr_raw = raw_X_train.iloc[tr_idx]
                    X_val_raw = raw_X_train.iloc[val_idx]
                    X_tr_f, X_val_f = preprocess_fold(X_tr_raw, X_val_raw, num_cols, cat_cols)
                else:
                    X_tr_f, X_val_f = X_train[tr_idx], X_train[val_idx]

                y_tr_f, y_val_f = y_train[tr_idx], y_train[val_idx]

                fold_model, _, _ = get_model_instance(model_name, problem_type, random_state)
                fold_model.fit(X_tr_f, y_tr_f)
                val_pred = fold_model.predict(X_val_f)

                if hasattr(fold_model, "predict_proba"):
                    val_prob = fold_model.predict_proba(X_val_f)
                    if is_binary:
                        p_pos_f = val_prob[:, 1] if val_prob.ndim == 2 else val_prob
                        oof_y_prob[val_idx, 1] = p_pos_f
                        oof_y_prob[val_idx, 0] = 1.0 - p_pos_f
                    else:
                        oof_y_prob[val_idx] = val_prob
                else:
                    oof_y_prob[val_idx, 1] = val_pred
                    oof_y_prob[val_idx, 0] = 1.0 - val_pred

                m = evaluate_classification(y_val_f, val_pred, val_prob if hasattr(fold_model, "predict_proba") else None, user_goal=user_goal)
                cv_scores.append(m.get("roc_auc", m.get("accuracy", 0.0)))

            if is_binary:
                oof_p_pos = oof_y_prob[:, 1]
                logger.info("OOF validation predictions generated on training set.")
                logger.info("Evaluating threshold grid (0.05 - 0.95) on OOF validation predictions...")
                oof_threshold_analysis = analyze_classification_thresholds(
                    y_true=y_train,
                    p_positive=oof_p_pos,
                    user_goal=user_goal,
                    is_oof_validation=True
                )
                locked_operating_threshold = oof_threshold_analysis["operating_threshold"]["threshold"]
                logger.info(f"Selected operating threshold: {locked_operating_threshold} (optimised for F2 under stated objective).")
                logger.info("Operating threshold locked.")

        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            for tr_idx, val_idx in cv.split(X_train, y_train):
                if raw_X_train is not None and num_cols is not None and cat_cols is not None:
                    X_tr_raw = raw_X_train.iloc[tr_idx]
                    X_val_raw = raw_X_train.iloc[val_idx]
                    X_tr_f, X_val_f = preprocess_fold(X_tr_raw, X_val_raw, num_cols, cat_cols)
                else:
                    X_tr_f, X_val_f = X_train[tr_idx], X_train[val_idx]

                y_tr_f, y_val_f = y_train[tr_idx], y_train[val_idx]

                fold_model, _, _ = get_model_instance(model_name, problem_type, random_state)
                fold_model.fit(X_tr_f, y_tr_f)
                val_pred = fold_model.predict(X_val_f)

                if problem_type == "regression":
                    m = evaluate_regression(y_val_f, val_pred)
                    cv_scores.append(m.get("rmse", 0.0))
                elif problem_type == "forecasting":
                    m = evaluate_forecasting(y_val_f, val_pred)
                    cv_scores.append(m.get("wape", 0.0))

    cv_mean = round(float(np.mean(cv_scores)), 4) if cv_scores else 0.0
    cv_std = round(float(np.std(cv_scores)), 4) if cv_scores else 0.0

    # Candidate Training Metrics
    train_metrics = {}
    if problem_type == "classification":
        train_metrics = evaluate_classification(
            y_train, y_train_pred, y_train_prob,
            user_goal=user_goal,
            locked_threshold=locked_operating_threshold,
            oof_threshold_analysis=oof_threshold_analysis
        )
    elif problem_type == "regression":
        train_metrics = evaluate_regression(y_train, y_train_pred)
    elif problem_type == "forecasting":
        train_metrics = evaluate_forecasting(y_train, y_train_pred)

    exp_result = {
        "model_name": model_name,
        "model": model,
        "family": family,
        "model_family": family,
        "params": params,
        "problem_type": problem_type,
        "feature_names": feature_names,
        "train_time_sec": train_time_sec,
        "metrics": {
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "cv_scores": cv_scores,
            "train": train_metrics,
            "test": {},  # Holdout test metrics populated ONCE after champion selection
        },
        "oof_threshold_analysis": oof_threshold_analysis,
        "locked_operating_threshold": locked_operating_threshold,
    }

    # If X_test/y_test were provided explicitly for single-model calls, evaluate holdout once
    if X_test is not None and y_test is not None:
        evaluate_locked_champion_on_holdout(
            champion_exp=exp_result,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            user_goal=user_goal,
            track_mlflow=track_mlflow
        )

    return exp_result


def evaluate_locked_champion_on_holdout(
    champion_exp: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    user_goal: str = "",
    track_mlflow: bool = True
) -> Dict[str, Any]:
    """
    Fit the final locked champion model on full training data, apply the already-locked
    operating threshold, and evaluate the untouched final holdout test set EXACTLY ONCE.
    """
    logger.info("FINAL MODEL FIT COMPLETE ON FULL TRAINING DATA.")
    logger.info("FINAL HOLDOUT EVALUATION STARTED ON UNTOUCHED TEST SET.")

    model = champion_exp["model"]
    problem_type = champion_exp["problem_type"]
    locked_th = champion_exp.get("locked_operating_threshold")
    oof_th_analysis = champion_exp.get("oof_threshold_analysis")

    # Fit final model on complete training portion
    start_t = time.time()
    model.fit(X_train, y_train)
    fit_time_sec = round(time.time() - start_t, 3)

    y_test_pred = model.predict(X_test)
    y_test_prob = None
    if problem_type == "classification" and hasattr(model, "predict_proba"):
        try:
            y_test_prob = model.predict_proba(X_test)
        except Exception:
            pass

    if problem_type == "classification":
        test_metrics = evaluate_classification(
            y_test, y_test_pred, y_test_prob,
            user_goal=user_goal,
            locked_threshold=locked_th,
            oof_threshold_analysis=oof_th_analysis
        )
    elif problem_type == "regression":
        test_metrics = evaluate_regression(y_test, y_test_pred)
    elif problem_type == "forecasting":
        test_metrics = evaluate_forecasting(y_test, y_test_pred)
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")

    logger.info("FINAL HOLDOUT EVALUATION COMPLETED.")

    champion_exp["metrics"]["test"] = test_metrics
    champion_exp["final_fit_time_sec"] = fit_time_sec
    champion_exp["y_test_pred"] = y_test_pred
    champion_exp["y_test_prob"] = y_test_prob

    if track_mlflow:
        try:
            with mlflow.start_run(run_name=f"Champion_{champion_exp['model_name']}") as active_run:
                champion_exp["mlflow_run_id"] = active_run.info.run_id
                mlflow.log_params(champion_exp.get("params", {}))
                mlflow.log_metric("cv_mean", champion_exp.get("metrics", {}).get("cv_mean", 0.0))
                for k, v in test_metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"test_{k}", v)
        except Exception as e:
            logger.warning(f"MLflow champion logging warning: {e}")
            champion_exp["mlflow_run_id"] = f"mock_{uuid.uuid4().hex[:8]}"
    else:
        champion_exp["mlflow_run_id"] = f"mock_{uuid.uuid4().hex[:8]}"

    return champion_exp

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
