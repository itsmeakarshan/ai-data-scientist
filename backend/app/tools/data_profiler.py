"""
AutoDS Data Profiler Tool
Performs deterministic, comprehensive exploratory data analysis, summary statistics, and type profiling.
"""

import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backend.app.core.logging import logger


def is_candidate_datetime(series: pd.Series) -> bool:
    """Check if a series is datetime or parsable as datetime strings without triggering unhandled warnings."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (series.dtype == object or pd.api.types.is_string_dtype(series)):
        return False

    sample = series.dropna().head(30)
    if len(sample) == 0:
        return False

    # Regex patterns for common genuine date/time strings
    date_patterns = [
        re.compile(r'^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}'),        # YYYY-MM-DD, YYYY/MM/DD
        re.compile(r'^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}'),       # MM/DD/YYYY, DD-MM-YYYY
        re.compile(r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}'),      # ISO datetime: 2023-01-01 12:00
        re.compile(r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}', re.IGNORECASE), # Month name format
    ]

    # Check what proportion of samples match standard date patterns
    match_count = sum(
        1 for val in sample
        if isinstance(val, str) and any(p.match(val.strip()) for p in date_patterns)
    )

    # If fewer than 70% of non-null samples match date syntax, reject immediately
    if match_count / len(sample) < 0.7:
        return False

    try:
        converted = pd.to_datetime(sample, errors="coerce", format="mixed")
        return (converted.notna().sum() / len(sample)) >= 0.8
    except Exception:
        return False


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a full deterministic profile of a pandas DataFrame.
    Returns structured summary stats, column types, missingness, duplicates, correlations, and candidate targets.
    """
    total_rows = len(df)
    total_cols = len(df.columns)

    if total_rows == 0:
        return {
            "row_count": 0,
            "col_count": total_cols,
            "summary_stats": {},
            "missingness_report": {},
            "column_types": {},
            "correlations": {},
            "candidate_targets": [],
            "candidate_datetimes": []
        }

    # 1. Duplicates and Missingness
    duplicate_rows = int(df.duplicated().sum())
    duplicate_pct = round((duplicate_rows / total_rows) * 100, 2)

    missing_by_col = df.isnull().sum().to_dict()
    missing_pct_by_col = {
        col: round((count / total_rows) * 100, 2)
        for col, count in missing_by_col.items()
    }
    total_missing_cells = int(df.isnull().sum().sum())
    total_cells = total_rows * total_cols
    total_missing_pct = round((total_missing_cells / total_cells) * 100, 2)

    missingness_report = {
        "total_missing_cells": total_missing_cells,
        "total_missing_pct": total_missing_pct,
        "missing_by_column": missing_by_col,
        "missing_pct_by_column": missing_pct_by_col,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": duplicate_pct,
    }

    # 2. Type Classification & Detailed Column Stats
    column_types: Dict[str, str] = {}
    numerical_stats: Dict[str, Dict[str, Any]] = {}
    categorical_stats: Dict[str, Dict[str, Any]] = {}
    datetime_stats: Dict[str, Dict[str, Any]] = {}
    id_columns: List[str] = []
    constant_columns: List[str] = []
    candidate_datetimes: List[str] = []

    for col in df.columns:
        series = df[col]
        nunique = series.nunique(dropna=True)

        # Check for constant
        if nunique <= 1:
            constant_columns.append(col)
            column_types[col] = "constant"
            continue

        # Check for ID columns (100% unique or high unique ratio with 'id' in name)
        if (nunique == total_rows or nunique / total_rows > 0.95) and ("id" in col.lower() or "key" in col.lower() or col.lower() == "index"):
            id_columns.append(col)
            column_types[col] = "id"
            continue

        # Check datetime
        if is_candidate_datetime(series):
            column_types[col] = "datetime"
            candidate_datetimes.append(col)
            try:
                dt_series = pd.to_datetime(series, errors="coerce", format="mixed")
                min_dt = dt_series.min()
                max_dt = dt_series.max()
                datetime_stats[col] = {
                    "min_date": min_dt.isoformat() if pd.notna(min_dt) else None,
                    "max_date": max_dt.isoformat() if pd.notna(max_dt) else None,
                    "unique_dates": nunique,
                    "missing_count": int(series.isnull().sum()),
                }
            except Exception:
                datetime_stats[col] = {"unique_dates": nunique}
            continue

        # Check Numeric vs Categorical
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            # If numeric with very few unique values (e.g. binary 0/1), record as numeric or flag
            column_types[col] = "numeric"
            clean_s = series.dropna()

            mean_val = float(clean_s.mean()) if len(clean_s) > 0 else 0.0
            std_val = float(clean_s.std()) if len(clean_s) > 1 else 0.0
            min_val = float(clean_s.min()) if len(clean_s) > 0 else 0.0
            q25_val = float(clean_s.quantile(0.25)) if len(clean_s) > 0 else 0.0
            median_val = float(clean_s.median()) if len(clean_s) > 0 else 0.0
            q75_val = float(clean_s.quantile(0.75)) if len(clean_s) > 0 else 0.0
            max_val = float(clean_s.max()) if len(clean_s) > 0 else 0.0
            skew_val = float(clean_s.skew()) if len(clean_s) > 2 else 0.0
            kurt_val = float(clean_s.kurt()) if len(clean_s) > 3 else 0.0
            zero_count = int((clean_s == 0).sum())

            # Outlier detection (IQR method)
            iqr = q75_val - q25_val
            lower_bound = q25_val - (1.5 * iqr)
            upper_bound = q75_val + (1.5 * iqr)
            outlier_count = int(((clean_s < lower_bound) | (clean_s > upper_bound)).sum())

            numerical_stats[col] = {
                "count": int(clean_s.count()),
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "min": round(min_val, 4),
                "q25": round(q25_val, 4),
                "median": round(median_val, 4),
                "q75": round(q75_val, 4),
                "max": round(max_val, 4),
                "skewness": round(skew_val, 4) if not np.isnan(skew_val) else 0.0,
                "kurtosis": round(kurt_val, 4) if not np.isnan(kurt_val) else 0.0,
                "zero_count": zero_count,
                "outlier_count": outlier_count,
                "outlier_pct": round((outlier_count / total_rows) * 100, 2),
                "missing_count": int(series.isnull().sum()),
                "missing_pct": missing_pct_by_col[col],
            }
        else:
            column_types[col] = "categorical"
            clean_s = series.dropna().astype(str)
            top_counts = clean_s.value_counts().head(10).to_dict()
            top_val = clean_s.mode().iloc[0] if len(clean_s) > 0 else None
            top_freq = int(clean_s.value_counts().iloc[0]) if len(clean_s) > 0 else 0

            categorical_stats[col] = {
                "count": int(clean_s.count()),
                "unique_count": nunique,
                "cardinality_ratio": round(nunique / total_rows, 4),
                "top_value": top_val,
                "top_freq": top_freq,
                "top_freq_pct": round((top_freq / total_rows) * 100, 2) if total_rows > 0 else 0.0,
                "top_categories": {k: int(v) for k, v in top_counts.items()},
                "missing_count": int(series.isnull().sum()),
                "missing_pct": missing_pct_by_col[col],
            }

    # 3. Correlation Matrix (Numeric Features)
    numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
    correlations: Dict[str, Any] = {"matrix": {}, "top_positive": [], "top_negative": []}

    if len(numeric_cols) >= 2:
        try:
            num_df = df[numeric_cols].dropna()
            if len(num_df) > 1:
                corr_matrix = num_df.corr(method="pearson").round(4).to_dict()
                correlations["matrix"] = corr_matrix

                pairs = []
                for i in range(len(numeric_cols)):
                    for j in range(i + 1, len(numeric_cols)):
                        c1, c2 = numeric_cols[i], numeric_cols[j]
                        val = corr_matrix.get(c1, {}).get(c2, 0.0)
                        if not np.isnan(val):
                            pairs.append({"feature_1": c1, "feature_2": c2, "correlation": val})

                pairs_sorted = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)
                correlations["top_positive"] = [p for p in pairs_sorted if p["correlation"] > 0][:10]
                correlations["top_negative"] = [p for p in pairs_sorted if p["correlation"] < 0][:10]
        except Exception as e:
            logger.debug(f"Correlation calculation error: {e}")

    # 4. Candidate Target Detection
    candidate_targets = []
    target_names_priority = [
        "y", "target", "label", "class", "churn", "status", "outcome",
        "default", "sales", "price", "revenue", "demand", "median_house_value", "deposit"
    ]

    # Priority 1: Exact matches in priority list
    for name in target_names_priority:
        for col in df.columns:
            if col.lower() == name and col not in candidate_targets:
                candidate_targets.append(col)

    # Priority 2: Substring matches
    for col in df.columns:
        col_lower = col.lower()
        if any(p in col_lower for p in ("target", "label", "outcome", "churn", "is_")) and col not in candidate_targets:
            candidate_targets.append(col)

    # Priority 3: Last column in dataset (standard convention in many ML datasets)
    last_col = df.columns[-1]
    if last_col not in candidate_targets and column_types.get(last_col) != "id":
        candidate_targets.append(last_col)

    # Inferred problem type preview
    inferred_problem_type = None
    if len(candidate_datetimes) > 0:
        inferred_problem_type = "forecasting"
    elif len(candidate_targets) > 0:
        primary_target = candidate_targets[0]
        t_type = column_types.get(primary_target)
        if t_type == "categorical" or df[primary_target].nunique() <= 10:
            inferred_problem_type = "classification"
        else:
            inferred_problem_type = "regression"
    else:
        inferred_problem_type = "eda"

    return {
        "row_count": total_rows,
        "col_count": total_cols,
        "summary_stats": {
            "numerical_columns": numerical_stats,
            "categorical_columns": categorical_stats,
            "datetime_columns": datetime_stats,
            "id_columns": id_columns,
            "constant_columns": constant_columns,
        },
        "missingness_report": missingness_report,
        "column_types": column_types,
        "correlations": correlations,
        "candidate_targets": candidate_targets,
        "candidate_datetimes": candidate_datetimes,
        "inferred_problem_type": inferred_problem_type,
    }
