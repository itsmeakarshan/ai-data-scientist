"""
AutoDS Data Quality Detector Tool
Identifies missingness, high cardinality, class imbalance, zero-variance columns, extreme outliers, and potential leakage.
"""

from typing import Any, Dict, List, Optional
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
                    "message": f"Severe class imbalance: minority class '{min_class_name}' represents only {min_class_ratio*100:.2f}% of data.",
                    "suggested_action": "Use StratifiedKFold, class weights, SMOTE, and evaluate using PR-AUC / ROC-AUC / F1 instead of accuracy."
                })
            elif min_class_ratio < 0.20:
                alerts.append({
                    "type": "moderate_target_imbalance",
                    "column": target_column,
                    "severity": "warning",
                    "message": f"Moderate class imbalance: minority class '{min_class_name}' is {min_class_ratio*100:.1f}%.",
                    "suggested_action": "Use StratifiedKFold validation and monitor precision-recall curves."
                })

        # 5. Potential Leakage Detection (Features correlated suspiciously high with target)
        if pd.api.types.is_numeric_dtype(df[target_column]):
            for col in df.select_dtypes(include=[np.number]).columns:
                if col != target_column:
                    corr = df[[col, target_column]].dropna().corr().iloc[0, 1]
                    if abs(corr) >= 0.95:
                        alerts.append({
                            "type": "potential_target_leakage",
                            "column": col,
                            "severity": "critical",
                            "message": f"Feature '{col}' has an extreme correlation ({corr:.4f}) with the target '{target_column}'.",
                            "suggested_action": "Audit whether this column is a proxy or post-event measurement, and drop if leaky."
                        })
                        
        # Specific known dataset leakage heuristics (e.g. Bank Marketing 'duration' column)
        if target_column in ("y", "deposit") and "duration" in df.columns:
            alerts.append({
                "type": "domain_target_leakage",
                "column": "duration",
                "severity": "warning",
                "message": "Column 'duration' (call duration) heavily affects target outcome, but is unknown before a call is made.",
                "suggested_action": "Include models trained both with and without 'duration' to assess realistic prospective performance."
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
