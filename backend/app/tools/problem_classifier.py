"""
AutoDS Problem Classifier Tool
Infers task type (classification, regression, forecasting, eda) using deterministic heuristics.
"""

from typing import Any, Dict, Optional, Tuple
import pandas as pd
from backend.app.core.logging import logger
from backend.app.tools.data_profiler import is_candidate_datetime


def classify_problem_type(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    time_column: Optional[str] = None,
    user_goal: str = ""
) -> Dict[str, Any]:
    """
    Deterministically analyze the DataFrame, target, time dimensions, and user objective
    to determine the appropriate Data Science workflow.
    """
    goal_lower = user_goal.lower()
    
    # 1. Look for explicit user goal keywords
    wants_forecast = any(w in goal_lower for w in ("forecast", "time series", "future sales", "demand", "predict next", "horizon"))
    wants_classify = any(w in goal_lower for w in ("classify", "classification", "churn", "subscribe", "fraud", "default", "categorical", "binary"))
    wants_regression = any(w in goal_lower for w in ("regression", "price", "predict price", "continuous", "amount", "revenue", "cost", "value"))
    wants_eda = any(w in goal_lower for w in ("eda", "explore", "profile", "overview", "visualize", "summary only"))

    # 2. Time Column & Forecasting Detection
    detected_time_col = time_column
    if not detected_time_col:
        for col in df.columns:
            if is_candidate_datetime(df[col]):
                detected_time_col = col
                break

    # 3. Target Detection
    detected_target_col = target_column
    if not detected_target_col:
        # Candidate priorities
        priority_names = ["y", "target", "label", "class", "churn", "deposit", "sales", "price", "revenue", "median_house_value", "demand", "status"]
        for p in priority_names:
            for col in df.columns:
                if col.lower() == p:
                    detected_target_col = col
                    break
            if detected_target_col:
                break
                
        if not detected_target_col:
            # Fallback to last non-id, non-time column
            for col in reversed(df.columns):
                if col != detected_time_col and "id" not in col.lower():
                    detected_target_col = col
                    break

    # 4. Infer Problem Type
    problem_type = "classification"
    sub_type = "binary_classification"
    confidence = 0.90
    reasoning = []

    if wants_eda and not (wants_classify or wants_regression or wants_forecast):
        problem_type = "eda"
        sub_type = "exploratory_data_analysis"
        reasoning.append("User requested Exploratory Data Analysis without model training.")
        return {
            "problem_type": problem_type,
            "sub_type": sub_type,
            "target_column": detected_target_col,
            "time_column": detected_time_col,
            "confidence": confidence,
            "reasoning": " ".join(reasoning),
            "recommended_metric": "none",
            "recommended_validation": "none"
        }

    if (detected_time_col and wants_forecast) or (detected_time_col and detected_target_col in ("sales", "demand", "volume", "revenue")):
        problem_type = "forecasting"
        sub_type = "time_series_forecasting"
        reasoning.append(f"Detected temporal index column '{detected_time_col}' and sequential target '{detected_target_col}'.")
        return {
            "problem_type": problem_type,
            "sub_type": sub_type,
            "target_column": detected_target_col,
            "time_column": detected_time_col,
            "confidence": 0.95,
            "reasoning": " ".join(reasoning),
            "recommended_metric": "wape",
            "recommended_validation": "walk_forward_time_split"
        }

    if detected_target_col and detected_target_col in df.columns:
        target_series = df[detected_target_col].dropna()
        nunique = target_series.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(target_series)
        
        if nunique == 2 or (nunique <= 5 and not is_numeric):
            problem_type = "classification"
            sub_type = "binary_classification" if nunique == 2 else "multiclass_classification"
            val_counts = target_series.value_counts(normalize=True)
            min_ratio = float(val_counts.min()) if len(val_counts) > 0 else 0.5
            
            if nunique == 2 and min_ratio < 0.25:
                reasoning.append(f"Target '{detected_target_col}' has binary distribution with class imbalance (minority prevalence: {min_ratio*100:.1f}%). Primary evaluation relies on PR-AUC, ROC-AUC, and F1/F2 rather than raw accuracy.")
                recommended_metric = "pr_auc" if min_ratio < 0.15 else "roc_auc"
            else:
                reasoning.append(f"Target '{detected_target_col}' has {nunique} discrete categories.")
                recommended_metric = "roc_auc" if nunique == 2 else "f1_macro"
            recommended_validation = "stratified_kfold"
        elif 2 < nunique <= 20 and (not is_numeric or wants_classify):
            problem_type = "classification"
            sub_type = "multiclass_classification"
            reasoning.append(f"Target '{detected_target_col}' has {nunique} discrete classes.")
            recommended_metric = "f1_macro"
            recommended_validation = "stratified_kfold"
        elif is_numeric and nunique > 20:
            problem_type = "regression"
            sub_type = "tabular_regression"
            reasoning.append(f"Target '{detected_target_col}' is continuous numeric with {nunique} unique values.")
            recommended_metric = "rmse"
            recommended_validation = "kfold"
        else:
            problem_type = "classification"
            sub_type = "binary_classification"
            recommended_metric = "roc_auc"
            recommended_validation = "stratified_kfold"
    else:
        problem_type = "eda"
        sub_type = "exploratory_data_analysis"
        recommended_metric = "none"
        recommended_validation = "none"
        reasoning.append("No valid target column identified in dataset.")

    return {
        "problem_type": problem_type,
        "sub_type": sub_type,
        "target_column": detected_target_col,
        "time_column": detected_time_col,
        "confidence": confidence,
        "reasoning": " ".join(reasoning),
        "recommended_metric": recommended_metric,
        "recommended_validation": recommended_validation
    }
