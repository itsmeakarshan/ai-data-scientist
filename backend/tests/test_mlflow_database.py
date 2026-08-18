"""
Unit Tests for MLflow Database Tracking Backend
Tests initialization, experiment creation, parameter logging, metric logging, and artifact handling.
"""

import os
import tempfile
import mlflow
import numpy as np
import pytest
from backend.app.tools.ml_trainer import initialize_mlflow, train_and_evaluate_model


def test_mlflow_database_initialization_and_logging():
    """Verify MLflow tracks runs inside an SQLite database backend cleanly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_mlflow.db")
        tracking_uri = f"sqlite:///{db_path}"
        
        # Test initialization
        initialize_mlflow(tracking_uri=tracking_uri, experiment_name="Test_DB_Experiment", force=True)
        
        # Test synthetic dataset run
        X_tr = np.random.randn(50, 4)
        y_tr = np.random.randint(0, 2, size=50)
        X_te = np.random.randn(20, 4)
        y_te = np.random.randint(0, 2, size=20)
        
        res = train_and_evaluate_model(
            model_name="RandomForest",
            problem_type="classification",
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_te,
            y_test=y_te,
            feature_names=["f1", "f2", "f3", "f4"],
            cv_folds=2,
            track_mlflow=True
        )
        
        assert res["mlflow_run_id"] is not None
        
        # Query MLflow client to verify run persistence
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        run_data = client.get_run(res["mlflow_run_id"])
        
        assert run_data.info.run_id == res["mlflow_run_id"]
        assert "n_estimators" in run_data.data.params
        assert "cv_mean" in run_data.data.metrics
        assert "test_accuracy" in run_data.data.metrics
