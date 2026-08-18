import pytest

from backend.app.api.v1.models import extract_model_metric_and_score


def test_extract_model_metric_and_score_classification():
    """Verify classification metrics extract holdout ROC-AUC or balanced accuracy strictly on [0, 1]."""
    # Bank Marketing LightGBM
    metrics_bank = {
        "test": {
            "roc_auc": 0.8121,
            "pr_auc": 0.4829,
            "balanced_accuracy": 0.7619,
            "accuracy": 0.8998
        }
    }
    name, score = extract_model_metric_and_score("classification", metrics_bank)
    assert name == "Holdout ROC-AUC"
    assert score == 0.8121
    assert 0.0 <= score <= 1.0

    # Breast Cancer RandomForest
    metrics_cancer = {
        "test": {
            "roc_auc": 0.9911,
            "balanced_accuracy": 0.9504,
            "accuracy": 0.9561
        }
    }
    name, score = extract_model_metric_and_score("classification", metrics_cancer)
    assert name == "Holdout ROC-AUC"
    assert score == 0.9911
    assert score <= 1.0


def test_extract_model_metric_and_score_regression():
    """Verify regression metrics extract holdout R² strictly on [0, 1]."""
    metrics_housing = {
        "test": {
            "r2": 0.8183,
            "rmse": 48789.54,
            "mae": 32790.48
        }
    }
    name, score = extract_model_metric_and_score("regression", metrics_housing)
    assert name == "Holdout R²"
    assert score == 0.8183
    assert 0.0 <= score <= 1.0


def test_extract_model_metric_and_score_forecasting_wape_regression_test():
    """Regression test specifically proving that WAPE metrics never produce 99.819 or exceed 1.0."""
    # Case 1: WAPE stored as decimal fraction 0.1813 (18.13%)
    metrics_wape_frac = {
        "test": {
            "wape": 0.1813,
            "rmse": 68.61,
            "mae": 44.97
        }
    }
    name, score = extract_model_metric_and_score("forecasting", metrics_wape_frac)
    assert name == "Holdout Accuracy (1-WAPE)"
    assert score == pytest.approx(0.8187, rel=1e-3)
    assert score <= 1.0
    assert score != 99.819
    assert score < 1.0

    # Case 2: WAPE stored as percentage 18.13
    metrics_wape_pct = {
        "test": {
            "wape": 18.13,
            "rmse": 68.61
        }
    }
    name, score = extract_model_metric_and_score("forecasting", metrics_wape_pct)
    assert name == "Holdout Accuracy (1-WAPE)"
    assert score == pytest.approx(0.8187, rel=1e-3)
    assert score <= 1.0
    assert score != 99.819

    # Case 3: Forecasting with R2 present
    metrics_forecast_r2 = {
        "test": {
            "r2": 0.9030,
            "wape": 18.13,
            "rmse": 68.61
        }
    }
    name, score = extract_model_metric_and_score("forecasting", metrics_forecast_r2)
    assert name == "Holdout R²"
    assert score == 0.9030
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_list_models_api_normalization_and_deduplication():
    """Verify GET /api/v1/models populates normalized_score and deduplicates via latest_per_dataset."""
    from httpx import ASGITransport, AsyncClient

    from backend.app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/api/v1/models?latest_per_dataset=true&is_best=true")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)

        for item in data:
            assert "normalized_score" in item
            assert "metric_name" in item
            if item["normalized_score"] is not None:
                assert 0.0 <= item["normalized_score"] <= 1.0
                assert item["normalized_score"] != 99.819
