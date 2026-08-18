"""
Pytest configuration and synthetic fixtures for AutoDS backend tests.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.core.config import settings
from backend.app.core.database import init_db
from backend.app.main import app


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configure test environment and sqlite in-memory db."""
    settings.ENVIRONMENT = "test"
    settings.STORAGE_DIR = tempfile.mkdtemp(prefix="autods_test_data_")
    settings.REPORTS_DIR = tempfile.mkdtemp(prefix="autods_test_reports_")
    settings.EXPERIMENTS_DIR = tempfile.mkdtemp(prefix="autods_test_exp_")


@pytest.fixture
def synthetic_classification_df() -> pd.DataFrame:
    """Generate synthetic binary classification dataset."""
    np.random.seed(42)
    n = 200
    age = np.random.randint(18, 70, n)
    income = np.random.normal(50000, 15000, n).clip(15000, 150000)
    category = np.random.choice(["admin", "services", "technician", "management"], n)
    credit_score = np.random.normal(650, 80, n).clip(300, 850)

    # Target probability depends on income and credit score
    prob = 1.0 / (1.0 + np.exp(-( (income - 50000)/30000 + (credit_score - 650)/100 )))
    target = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income.round(2),
        "job_category": category,
        "credit_score": credit_score.round(1),
        "target": target
    })
    return df


@pytest.fixture
def synthetic_regression_df() -> pd.DataFrame:
    """Generate synthetic tabular regression dataset."""
    np.random.seed(42)
    n = 200
    sqft = np.random.normal(2000, 400, n).clip(600, 4000)
    bedrooms = np.random.choice([1, 2, 3, 4], n)
    city = np.random.choice(["Austin", "Seattle", "Denver"], n)
    price = 100000 + (sqft * 150) + (bedrooms * 20000) + np.random.normal(0, 15000, n)

    return pd.DataFrame({
        "sqft": sqft.round(),
        "bedrooms": bedrooms,
        "city": city,
        "price": price.round(2)
    })


@pytest.fixture
def synthetic_forecasting_df() -> pd.DataFrame:
    """Generate synthetic time-series forecasting dataset."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=180, freq="D")
    trend = np.linspace(10, 30, len(dates))
    season = 5 * np.sin(2 * np.pi * dates.dayofweek.to_numpy() / 7)
    noise = np.random.normal(0, 2, len(dates))
    sales = np.maximum(0, trend + season + noise).round(1)

    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "store": "Store_A",
        "sales": sales
    })


@pytest.fixture
def synthetic_csv_file(synthetic_classification_df, tmp_path) -> Path:
    """Save synthetic classification dataframe to a temporary CSV file."""
    csv_path = tmp_path / "synthetic_data.csv"
    synthetic_classification_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
async def async_client():
    """Async HTTP client for testing FastAPI API routes."""
    await init_db()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
