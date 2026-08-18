"""
AutoDS Data Quality Detector Tool
Identifies missingness, high cardinality, class imbalance, zero-variance columns, extreme outliers, and potential leakage.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def detect_data_quality(
    df: pd.DataFrame,
    profile: Optional[Dict[str, Any]] = None,
    target_column: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Audit dataset for data quality flaws, potential methodology hazards, and statistical anomalies.
    Returns a list of structured alerts with severity and suggested actions.
    """
    alerts: List[Dict[str, Any]] = []
    total_rows = len(df)
    if total_rows == 0:
        return alerts

    # 1. Check Missing Values
    missing_counts = df.isnull().sum()
    for col, count in missing_counts.items():
        if count > 0:
            pct = (count / total_rows) * 100
            if pct > 40.0:
                alerts.append({
                    "type": "high_missingness",
                    "column": col,
                    "severity": "critical",
                    "message": f"Column '{col}' has {pct:.1f}% missing values ({count}/{total_rows}).",
                    "suggested_action": "Consider dropping this column or creating an explicit 'is_missing' indicator."
                })
            elif pct > 5.0:
                alerts.append({
                    "type": "moderate_missingness",
                    "column": col,
                    "severity": "warning",
                    "message": f"Column '{col}' has {pct:.1f}% missing values ({count}/{total_rows}).",
                    "suggested_action": "Impute with median for numeric or most frequent / 'unknown' category."
                })

    # 2. Check Constant / Zero-Variance Features
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1:
            alerts.append({
                "type": "constant_feature",
                "column": col,
                "severity": "warning",
                "message": f"Column '{col}' is constant with {nunique} unique value.",
                "suggested_action": "Remove feature before model training as it contains zero predictive variance."
            })

    # 3. Check High Cardinality Categoricals
    for col in df.select_dtypes(include=["object", "category"]).columns:
        nunique = df[col].nunique(dropna=True)
        if nunique > 100 and nunique < total_rows:
            alerts.append({
                "type": "high_cardinality",
                "column": col,
                "severity": "warning",
                "message": f"Categorical column '{col}' has {nunique} unique categories.",
                "suggested_action": "Use frequency encoding, target encoding, or top-N grouping to prevent dimensionality explosion."
            })

    # 4. Check Target Quality (if target is known/detected)
    if target_column and target_column in df.columns:
        target_series = df[target_column].dropna()
        nunique_target = target_series.nunique()

        # Binary or Multiclass Imbalance check
        if nunique_target <= 10:
            val_counts = target_series.value_counts(normalize=True)
            min_class_ratio = val_counts.min()
            min_class_name = val_counts.idxmin()

            if min_class_ratio < 0.05:
                alerts.append({
                    "type": "extreme_target_imbalance",
                    "column": target_column,
                    "severity": "critical",
                    "message": f"Severe class imbalance: minority class '{min_class_name}' represents only {min_class_ratio*100:.2f}% of data. Raw accuracy is completely misleading.",
                    "suggested_action": "Use StratifiedKFold, class weights, and evaluate using PR-AUC / ROC-AUC / F1 / F2 instead of accuracy."
                })
            elif min_class_ratio < 0.25:
                alerts.append({
                    "type": "moderate_target_imbalance",
                    "column": target_column,
                    "severity": "warning",
                    "message": f"Class imbalance detected: minority class '{min_class_name}' is {min_class_ratio*100:.1f}%. Raw accuracy may present a false sense of model quality.",
                    "suggested_action": "Use StratifiedKFold validation, calibrate decision threshold, and evaluate PR-AUC, ROC-AUC, F1, and F2 scores."
                })

        # 5. Potential Leakage Detection & Deterministic Target Subcomponents
        leaky_cols, leak_expls = detect_target_component_leakage(df, target_column)
        for l_col in leaky_cols:
            alerts.append({
                "type": "target_component_leakage",
                "column": l_col,
                "severity": "critical",
                "message": leak_expls.get(l_col, f"Feature '{l_col}' represents target leakage or post-outcome information."),
                "suggested_action": f"Exclude '{l_col}' from feature matrix before model training to eliminate prospective target leakage."
            })

    # 6. Duplicates Check
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        dup_pct = (duplicate_count / total_rows) * 100
        severity = "critical" if dup_pct > 10.0 else "info"
        alerts.append({
            "type": "duplicate_rows",
            "column": None,
            "severity": severity,
            "message": f"Dataset contains {duplicate_count} exact duplicate rows ({dup_pct:.1f}%).",
            "suggested_action": "Deduplicate dataset before validation split to prevent test set contamination."
        })

    return alerts


def detect_target_component_leakage(
    df: pd.DataFrame,
    target_column: str,
    problem_type: Optional[str] = None
) -> Tuple[List[str], Dict[str, str]]:
    """
    Detect deterministic target-component leakage, exact identity proxies,
    and contemporaneous post-outcome features.

    Returns:
        leaky_columns: List of column names that must be excluded.
        explanations: Mapping from column name to scientific rationale.
    """
    leaky_cols: set = set()
    explanations: Dict[str, str] = {}

    if target_column not in df.columns:
        return list(leaky_cols), explanations

    # 1. Exact Identity / Duplicate Check
    for col in df.columns:
        if col == target_column:
            continue
        try:
            if (df[col] == df[target_column]).mean() > 0.999:
                leaky_cols.add(col)
                explanations[col] = f"Column '{col}' is an exact duplicate/identity of the target '{target_column}'."
        except Exception:
            pass

    # 2. Deterministic Additive / Subtractive Subcomponent Decomposition (Numeric)
    if pd.api.types.is_numeric_dtype(df[target_column]):
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]

        # Check pairwise additive: c1 + c2 == target
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                try:
                    diff = (df[c1] + df[c2] - df[target_column]).abs()
                    if diff.max() < 1e-4 or (diff == 0).mean() > 0.999:
                        leaky_cols.add(c1)
                        leaky_cols.add(c2)
                        explanations[c1] = f"Column '{c1}' is a deterministic additive component of target '{target_column}' ('{c1}' + '{c2}' == '{target_column}')."
                        explanations[c2] = f"Column '{c2}' is a deterministic additive component of target '{target_column}' ('{c1}' + '{c2}' == '{target_column}')."
                except Exception:
                    pass

        # Check pairwise difference: c1 - c2 == target or c2 - c1 == target
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                if c1 in leaky_cols and c2 in leaky_cols:
                    continue
                try:
                    diff1 = (df[c1] - df[c2] - df[target_column]).abs()
                    if diff1.max() < 1e-4 or (diff1 == 0).mean() > 0.999:
                        leaky_cols.add(c1)
                        leaky_cols.add(c2)
                        explanations[c1] = f"Column '{c1}' is a deterministic component of target '{target_column}' ('{c1}' - '{c2}' == '{target_column}')."
                        explanations[c2] = f"Column '{c2}' is a deterministic component of target '{target_column}' ('{c1}' - '{c2}' == '{target_column}')."
                except Exception:
                    pass

        # Check individual extreme correlation
        for col in num_cols:
            if col not in leaky_cols:
                try:
                    corr = df[[col, target_column]].dropna().corr().iloc[0, 1]
                    if abs(corr) >= 0.999:
                        leaky_cols.add(col)
                        explanations[col] = f"Feature '{col}' has an extreme correlation ({corr:.4f}) with the target '{target_column}'."
                except Exception:
                    pass

    # 3. Known Contemporaneous / Post-Outcome Domain Feature Keywords
    prospective_keywords = [
        "casual", "registered", "duration", "post_event", "post_call",
        "after_outcome", "future_val", "future_outcome", "post_sale", "post_conversion"
    ]
    for col in df.columns:
        if col == target_column:
            continue
        c_low = col.lower()
        # Ensure we do NOT flag legitimate historical lags / rolling features
        if any(leg in c_low for leg in ["lag_", "roll_", "rolling_", "cal_", "hist_"]):
            continue

        if any(k == c_low or (k in c_low and not any(leg in c_low for leg in ["lag", "roll", "cal"])) for k in prospective_keywords):
            leaky_cols.add(col)
            if col not in explanations:
                explanations[col] = f"Feature '{col}' represents contemporaneous/post-outcome information that would not be available at prediction time."

    return sorted(list(leaky_cols)), explanations
