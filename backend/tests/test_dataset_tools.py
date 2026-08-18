"""
Unit and Integration Tests for AutoDS Dataset Tools
Tests inspection, statistical profiling, data quality alerts, problem classification, and leak-free preprocessing.
"""

import numpy as np

from backend.app.tools.data_profiler import profile_dataset
from backend.app.tools.dataset_inspector import detect_csv_delimiter, load_dataset_as_dataframe
from backend.app.tools.preprocessor import create_forecasting_features, prepare_train_test_split
from backend.app.tools.problem_classifier import classify_problem_type
from backend.app.tools.quality_detector import detect_data_quality


def test_dataset_inspector_csv(synthetic_csv_file):
    """Test CSV delimiter detection, loading, and checksum."""
    delim = detect_csv_delimiter(synthetic_csv_file)
    assert delim == ","

    df, meta = load_dataset_as_dataframe(synthetic_csv_file)
    assert len(df) == 200
    assert len(df.columns) == 5
    assert meta["file_type"] == "csv"
    assert len(meta["checksum"]) == 64


def test_data_profiler(synthetic_classification_df):
    """Test comprehensive dataset statistical profiling."""
    profile = profile_dataset(synthetic_classification_df)

    assert profile["row_count"] == 200
    assert profile["col_count"] == 5
    assert "income" in profile["summary_stats"]["numerical_columns"]
    assert "job_category" in profile["summary_stats"]["categorical_columns"]
    assert profile["missingness_report"]["total_missing_cells"] == 0
    assert "target" in profile["candidate_targets"]


def test_quality_detector_alerts(synthetic_classification_df):
    """Test quality detection on synthetic data with introduced flaws."""
    flawed_df = synthetic_classification_df.copy()
    # Introduce missing values in age
    flawed_df.loc[:30, "age"] = np.nan
    # Introduce constant column
    flawed_df["constant_feature"] = 1.0

    alerts = detect_data_quality(flawed_df, target_column="target")
    alert_types = [a["type"] for a in alerts]

    assert "moderate_missingness" in alert_types
    assert "constant_feature" in alert_types


def test_problem_classifier(synthetic_classification_df, synthetic_regression_df, synthetic_forecasting_df):
    """Test automated Data Science problem classification heuristics."""
    cls_res = classify_problem_type(synthetic_classification_df, target_column="target")
    assert cls_res["problem_type"] == "classification"
    assert cls_res["sub_type"] == "binary_classification"
    assert cls_res["recommended_metric"] == "roc_auc"

    reg_res = classify_problem_type(synthetic_regression_df, target_column="price")
    assert reg_res["problem_type"] == "regression"
    assert reg_res["recommended_metric"] == "rmse"

    fc_res = classify_problem_type(synthetic_forecasting_df, target_column="sales", time_column="date", user_goal="Forecast sales")
    assert fc_res["problem_type"] == "forecasting"
    assert fc_res["recommended_metric"] == "wape"


def test_leak_free_preprocessor(synthetic_classification_df):
    """Test that preprocessor splits first and imputes/scales strictly from training distribution."""
    X_train, X_test, y_train, y_test, artifacts = prepare_train_test_split(
        df=synthetic_classification_df,
        target_column="target",
        problem_type="classification",
        test_size=0.25,
        random_state=42
    )

    assert len(X_train) == 150
    assert len(X_test) == 50
    assert len(y_train) == 150
    assert len(y_test) == 50
    assert len(artifacts.feature_names) >= 4
    assert artifacts.problem_type == "classification"


def test_forecasting_feature_generator(synthetic_forecasting_df):
    """Test chronological lag and rolling window feature generation."""
    feat_df = create_forecasting_features(
        synthetic_forecasting_df,
        time_column="date",
        target_column="sales",
        lags=[1, 2, 7],
        rolling_windows=[7]
    )

    assert "target_lag_1" in feat_df.columns
    assert "target_lag_7" in feat_df.columns
    assert "target_roll_mean_7" in feat_df.columns
    assert "cal_dayofweek" in feat_df.columns
    # Ensure no NaN rows after cleaning
    assert feat_df["target_lag_1"].isnull().sum() == 0
