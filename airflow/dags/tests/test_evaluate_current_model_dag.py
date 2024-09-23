from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from evaluate_current_model_dag import evaluate_current_model

import mlflow


@pytest.fixture
def mock_mlflow_tracking_uri():
    original_uri = mlflow.get_tracking_uri()
    test_uri = ""
    mlflow.set_tracking_uri(test_uri)
    yield test_uri
    mlflow.set_tracking_uri(original_uri)


@patch("evaluate_current_model_dag.create_engine")
@patch("evaluate_current_model_dag.pd.read_sql")
@patch("evaluate_current_model_dag.load_model")
@patch("evaluate_current_model_dag.mlflow.start_run")
@patch("evaluate_current_model_dag.mlflow.log_metrics")
@patch("evaluate_current_model_dag.mlflow.set_experiment")
def test_evaluate_current_model(
    mock_set_experiment,
    mock_log_metrics,
    mock_start_run,
    mock_load_model,
    mock_read_sql,
    mock_create_engine,
    mock_mlflow_tracking_uri,
):
    mock_set_experiment.return_value = None
    mock_start_run.return_value.__enter__.return_value = None
    mock_start_run.return_value.__exit__.return_value = None
    mock_log_metrics.return_value = None

    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    test_data = pd.DataFrame(
        {
            "truck_id": range(1, 11),
            "day_of_week": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            "month": [1] * 10,
            "day_of_year": range(1, 11),
            "total_weight": np.arange(1000, 2000, 100),
        }
    )

    mock_read_sql.return_value = test_data

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1100, 1200])
    mock_load_model.return_value = mock_model

    evaluate_current_model()

    mock_set_experiment.assert_called_once_with("Model Evaluation")
    mock_start_run.assert_called_once()
    mock_log_metrics.assert_called_once()

    metrics = mock_log_metrics.call_args[0][0]
    assert "daily_mse" in metrics
    assert "daily_rmse" in metrics
    assert "daily_r2" in metrics
    assert "daily_mae" in metrics
