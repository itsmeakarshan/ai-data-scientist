"""
Unit Tests for SHAP Explainability Compatibility
Tests TreeExplainer with LightGBM, Random Forest, Linear models across 2D, 3D, and list of ndarrays.
"""

import warnings

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

from backend.app.tools.explainability import _extract_mean_abs_shap, compute_shap_explanations


def test_shap_output_dimension_extractor():
    """Verify _extract_mean_abs_shap reduces 1D, 2D, 3D, and lists into 1D feature attributions."""
    # 2D ndarray
    v2 = np.array([[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]])
    res2 = _extract_mean_abs_shap(v2)
    assert res2.shape == (3,)
    assert np.allclose(res2, [1.0, 2.0, 3.0])

    # 3D ndarray (N, F, C)
    v3 = np.zeros((10, 4, 3))
    v3[:, 0, :] = 2.0
    res3 = _extract_mean_abs_shap(v3)
    assert res3.shape == (4,)
    assert res3[0] == 2.0

    # List of ndarrays
    v_list = [np.ones((10, 5)), np.full((10, 5), 2.0)]
    res_list = _extract_mean_abs_shap(v_list)
    assert res_list.shape == (5,)
    assert res_list[0] == 2.0  # binary index 1 positive class


def test_shap_with_lightgbm_binary_without_warnings():
    """Verify TreeExplainer with LightGBM binary classifier operates cleanly with zero deprecation warnings."""
    X = np.random.randn(80, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
    clf.fit(X, y)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        shap_res = compute_shap_explanations(
            model=clf,
            X_sample=X,
            feature_names=["f1", "f2", "f3", "f4"]
        )

        assert shap_res["shap_available"] is True
        assert len(shap_res["top_shap_features"]) == 4

        # Verify zero LightGBM SHAP UserWarnings
        lgb_warnings = [item for item in w if "LightGBM binary classifier" in str(item.message)]
        assert len(lgb_warnings) == 0


def test_shap_with_multiclass_and_regression():
    """Verify SHAP computes attributions on multiclass RF and Ridge regression."""
    X = np.random.randn(60, 4)
    y_multi = np.random.choice([0, 1, 2], size=60)
    y_reg = X[:, 0] * 3.0 + X[:, 1] * 2.0

    # Multiclass RF
    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y_multi)
    res_multi = compute_shap_explanations(rf, X, ["f1", "f2", "f3", "f4"])
    assert res_multi["shap_available"] is True

    # Ridge Regression
    ridge = Ridge().fit(X, y_reg)
    res_reg = compute_shap_explanations(ridge, X, ["f1", "f2", "f3", "f4"])
    assert res_reg["shap_available"] is True
    assert res_reg["top_shap_features"][0]["feature"] in ("f1", "f2")
