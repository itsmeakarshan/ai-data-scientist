"""
AutoDS Leak-Free Preprocessor & Feature Engineering Tool
Applies rigorous split-before-fit preprocessing to ensure zero data leakage across train, validation, and test sets.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


@dataclass
class PreprocessingArtifacts:
    feature_names: List[str]
    target_name: str
    target_encoder: Optional[Any]
    scaler: Optional[Any]
    one_hot_encoder: Optional[Any]
    imputation_values: Dict[str, Any]
    problem_type: str
    categorical_cols: List[str]
    numerical_cols: List[str]
    dropped_columns: List[str]
    raw_X_train: Optional[pd.DataFrame] = None


def preprocess_fold(
    X_tr_raw: pd.DataFrame,
    X_val_raw: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit imputer, scaler, and one-hot encoder strictly on X_tr_raw (CV training fold),
    and transform X_tr_raw and X_val_raw (CV validation fold).
    Ensures zero data leakage across cross-validation folds.
    """
    X_tr_imp = X_tr_raw.copy()
    X_val_imp = X_val_raw.copy()

    # 1. Imputation (Calculated strictly on X_tr_raw)
    for col in num_cols:
        med = float(X_tr_raw[col].median()) if pd.notna(X_tr_raw[col].median()) else 0.0
        X_tr_imp[col] = X_tr_imp[col].fillna(med)
        X_val_imp[col] = X_val_imp[col].fillna(med)

    for col in cat_cols:
        mode_val = str(X_tr_raw[col].mode().iloc[0]) if len(X_tr_raw[col].dropna()) > 0 else "missing"
        X_tr_imp[col] = X_tr_imp[col].fillna(mode_val).astype(str)
        X_val_imp[col] = X_val_imp[col].fillna(mode_val).astype(str)

    # 2. Scaling (Fit on X_tr_raw only)
    if len(num_cols) > 0:
        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(X_tr_imp[num_cols])
        X_val_num = scaler.transform(X_val_imp[num_cols])
    else:
        X_tr_num = np.empty((len(X_tr_imp), 0))
        X_val_num = np.empty((len(X_val_imp), 0))

    # 3. Categorical Encoding (Fit on X_tr_raw only)
    if len(cat_cols) > 0:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_tr_cat = ohe.fit_transform(X_tr_imp[cat_cols])
        X_val_cat = ohe.transform(X_val_imp[cat_cols])
        X_tr = np.hstack([X_tr_num, X_tr_cat])
        X_val = np.hstack([X_val_num, X_val_cat])
    else:
        X_tr = X_tr_num
        X_val = X_val_num

    return X_tr, X_val


def clean_dataframe(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    drop_leakage_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Clean raw dataset: drop exact duplicate rows, drop specified leakage features, drop high-cardinality raw IDs."""
    cleaned = df.copy()
    initial_rows = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    dropped_cols = []

    # Auto-detect target component / prospective leakage
    from backend.app.tools.quality_detector import detect_target_component_leakage
    if target_column:
        auto_leaks, _ = detect_target_component_leakage(cleaned, target_column=target_column)
        target_leak_cols = sorted(list(set((drop_leakage_columns or []) + auto_leaks)))
    else:
        target_leak_cols = sorted(list(set(drop_leakage_columns or [])))

    for col in target_leak_cols:
        if col in cleaned.columns and col != target_column:
            cleaned = cleaned.drop(columns=[col])
            dropped_cols.append(col)

    # Drop high-cardinality ID/Key columns (excluding target)
    for col in list(cleaned.columns):
        if col == target_column:
            continue
        nunique = cleaned[col].nunique(dropna=True)
        if nunique == len(cleaned) and ("id" in col.lower() or "key" in col.lower() or col.lower() in ("index", "instant", "row_id", "row_number")):
            cleaned = cleaned.drop(columns=[col])
            dropped_cols.append(col)

    return cleaned, dropped_cols


def create_forecasting_features(
    df: pd.DataFrame,
    time_column: str,
    target_column: str,
    group_columns: Optional[List[str]] = None,
    lags: List[int] = [1, 2, 3, 7, 14, 28],
    rolling_windows: List[int] = [7, 14, 28]
) -> pd.DataFrame:
    """
    Generate chronological time-series lag and rolling window features without future lookahead.
    """
    feat_df = df.copy()
    feat_df[time_column] = pd.to_datetime(feat_df[time_column])

    # Sort chronologically
    sort_cols = (group_columns or []) + [time_column]
    feat_df = feat_df.sort_values(by=sort_cols).reset_index(drop=True)

    # Calendar features
    dt = feat_df[time_column].dt
    feat_df["cal_dayofweek"] = dt.dayofweek
    feat_df["cal_day"] = dt.day
    feat_df["cal_month"] = dt.month
    feat_df["cal_year"] = dt.year
    feat_df["cal_is_weekend"] = (dt.dayofweek >= 5).astype(int)

    # Lag features
    if group_columns:
        grouped = feat_df.groupby(group_columns)[target_column]
        for lag in lags:
            feat_df[f"target_lag_{lag}"] = grouped.shift(lag)
        for window in rolling_windows:
            feat_df[f"target_roll_mean_{window}"] = grouped.shift(1).rolling(window=window, min_periods=1).mean()
            feat_df[f"target_roll_std_{window}"] = grouped.shift(1).rolling(window=window, min_periods=1).std().fillna(0)
    else:
        for lag in lags:
            feat_df[f"target_lag_{lag}"] = feat_df[target_column].shift(lag)
        for window in rolling_windows:
            feat_df[f"target_roll_mean_{window}"] = feat_df[target_column].shift(1).rolling(window=window, min_periods=1).mean()
            feat_df[f"target_roll_std_{window}"] = feat_df[target_column].shift(1).rolling(window=window, min_periods=1).std().fillna(0)

    # Drop rows where initial lag is NaN
    feat_df = feat_df.dropna(subset=[f"target_lag_{lags[0]}"]).reset_index(drop=True)
    return feat_df


def prepare_train_test_split(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
    time_column: Optional[str] = None,
    drop_leakage_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PreprocessingArtifacts]:
    """
    Fit-on-train only preprocessing pipeline.
    Returns X_train, X_test, y_train, y_test, and PreprocessingArtifacts.
    """
    clean_df, dropped = clean_dataframe(df, target_column=target_column, drop_leakage_columns=drop_leakage_cols)

    if target_column not in clean_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # Handle Forecasting split (strictly chronological)
    if problem_type == "forecasting" and time_column and time_column in clean_df.columns:
        feat_df = create_forecasting_features(clean_df, time_column=time_column, target_column=target_column)
        split_idx = int(len(feat_df) * (1.0 - test_size))
        train_df = feat_df.iloc[:split_idx].copy()
        test_df = feat_df.iloc[split_idx:].copy()

        feature_cols = [c for c in feat_df.columns if c not in [target_column, time_column]]
        # Categorical encoding for non-numeric columns
        cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(feat_df[c])]
        num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(feat_df[c])]

        # Simple frequency or label encoding for forecast categorical grouping
        for col in cat_cols:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col].astype(str))
            # Handle unseen labels in test
            known = set(le.classes_)
            test_df[col] = test_df[col].astype(str).map(lambda s: s if s in known else "<unknown>")
            if "<unknown>" not in le.classes_:
                le.classes_ = np.append(le.classes_, "<unknown>")
            test_df[col] = le.transform(test_df[col])

        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df[target_column].values
        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df[target_column].values

        artifacts = PreprocessingArtifacts(
            feature_names=feature_cols,
            target_name=target_column,
            target_encoder=None,
            scaler=None,
            one_hot_encoder=None,
            imputation_values={},
            problem_type=problem_type,
            categorical_cols=cat_cols,
            numerical_cols=num_cols,
            dropped_columns=dropped
        )
        return X_train, X_test, y_train, y_test, artifacts

    # Tabular Classification or Regression
    X_raw = clean_df.drop(columns=[target_column])
    y_raw = clean_df[target_column]

    # Target Encoding for classification
    target_encoder = None
    if problem_type == "classification":
        if not pd.api.types.is_numeric_dtype(y_raw) or y_raw.nunique() > 2:
            target_encoder = LabelEncoder()
            y_clean = target_encoder.fit_transform(y_raw.astype(str))
        else:
            y_clean = y_raw.values.astype(int)
    else:
        y_clean = y_raw.values.astype(float)

    # Initial Split
    if problem_type == "classification" and len(np.unique(y_clean)) > 1:
        # Check if any class has only 1 sample
        _, class_counts = np.unique(y_clean, return_counts=True)
        if (class_counts >= 2).all():
            stratify = y_clean
        else:
            stratify = None
    else:
        stratify = None

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_clean, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Separate Column Types
    num_cols = [c for c in X_train_raw.columns if pd.api.types.is_numeric_dtype(X_train_raw[c])]
    cat_cols = [c for c in X_train_raw.columns if c not in num_cols]

    # 1. Imputation (Calculated strictly on X_train)
    imputation_values = {}
    X_train_imp = X_train_raw.copy()
    X_test_imp = X_test_raw.copy()

    for col in num_cols:
        med = float(X_train_raw[col].median()) if pd.notna(X_train_raw[col].median()) else 0.0
        imputation_values[col] = med
        X_train_imp[col] = X_train_imp[col].fillna(med)
        X_test_imp[col] = X_test_imp[col].fillna(med)

    for col in cat_cols:
        mode_val = str(X_train_raw[col].mode().iloc[0]) if len(X_train_raw[col].dropna()) > 0 else "missing"
        imputation_values[col] = mode_val
        X_train_imp[col] = X_train_imp[col].fillna(mode_val).astype(str)
        X_test_imp[col] = X_test_imp[col].fillna(mode_val).astype(str)

    # 2. Scaling Numeric Features
    scaler = None
    if len(num_cols) > 0:
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train_imp[num_cols])
        X_test_num = scaler.transform(X_test_imp[num_cols])
    else:
        X_train_num = np.empty((len(X_train_imp), 0))
        X_test_num = np.empty((len(X_test_imp), 0))

    # 3. Categorical Encoding (OneHot for categoricals with max 50 categories)
    ohe = None
    encoded_feature_names = list(num_cols)

    if len(cat_cols) > 0:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_train_cat = ohe.fit_transform(X_train_imp[cat_cols])
        X_test_cat = ohe.transform(X_test_imp[cat_cols])
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
        encoded_feature_names.extend(cat_feature_names)

        X_train = np.hstack([X_train_num, X_train_cat])
        X_test = np.hstack([X_test_num, X_test_cat])
    else:
        X_train = X_train_num
        X_test = X_test_num

    artifacts = PreprocessingArtifacts(
        feature_names=encoded_feature_names,
        target_name=target_column,
        target_encoder=target_encoder,
        scaler=scaler,
        one_hot_encoder=ohe,
        imputation_values=imputation_values,
        problem_type=problem_type,
        categorical_cols=cat_cols,
        numerical_cols=num_cols,
        dropped_columns=dropped,
        raw_X_train=X_train_raw
    )

    return X_train, X_test, y_train, y_test, artifacts
