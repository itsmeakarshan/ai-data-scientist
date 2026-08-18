"""
Unit Tests for Intelligent Date Detection & Profiling
Tests genuine dates, non-date categorical strings, pure numeric columns, and mixed data.
"""

import warnings
import pandas as pd
import pytest
from backend.app.tools.data_profiler import is_candidate_datetime, profile_dataset
from backend.app.tools.problem_classifier import classify_problem_type


def test_date_detection_genuine_and_categorical():
    """Verify is_candidate_datetime distinguishes real dates from text without emitting warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # 1. ISO Dates
        iso_series = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])
        assert bool(is_candidate_datetime(iso_series)) is True
        
        # 2. US Slash Dates
        us_series = pd.Series(["01/15/2023", "02/20/2023", "03/25/2023", "04/30/2023"])
        assert bool(is_candidate_datetime(us_series)) is True
        
        # 3. Categorical Strings (Job titles, loan status)
        cat_series = pd.Series(["admin.", "blue-collar", "technician", "services", "management"])
        assert bool(is_candidate_datetime(cat_series)) is False
        
        # 4. Pure numeric series
        num_series = pd.Series([10, 25, 30, 45, 50])
        assert bool(is_candidate_datetime(num_series)) is False
        
        # 5. Mixed non-date text
        text_series = pd.Series(["approved", "pending", "rejected", None, "approved"])
        assert bool(is_candidate_datetime(text_series)) is False
        
        # Ensure zero pandas dateutil fallback UserWarnings were triggered
        date_warnings = [item for item in w if "Could not infer format" in str(item.message)]
        assert len(date_warnings) == 0


def test_profiler_and_classifier_with_dates():
    """Verify profile_dataset and classify_problem_type infer time series correctly on date columns."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
        "sales": [10 + i * 0.5 for i in range(100)],
        "category": ["A", "B"] * 50
    })
    
    profile = profile_dataset(df)
    assert "timestamp" in profile.get("candidate_datetimes", [])
    
    p_info = classify_problem_type(df, user_goal="Forecast sales for the next month")
    assert p_info["problem_type"] == "forecasting"
    assert p_info["time_column"] == "timestamp"
