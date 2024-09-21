from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from dags.evaluate_current_model_dag import evaluate_current_model


@pytest.fixture
def mock_mlflow():
    with patch("dags.evaluate_current_model_dag.mlflow") as mock_mlflow:
        yield mock_mlflow


@pytest.fixture
def mock_s3_helpers():
    with patch("dags.evaluate_current_model_dag.load_data_from_s3") as mock_load:
        yield mock_load


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=300),
            "total_weight": np.random.rand(300) * 1000,
            "truck_id": np.random.randint(1, 6, 300),
            "day_of_week": np.random.randint(0, 7, 300),
            "month": np.random.randint(1, 13, 300),
        }
    )


def test_evaluate_current_model(mock_mlflow, mock_s3_helpers, sample_data):
    mock_s3_helpers.return_value = sample_data.to_parquet()

    mock_model = MagicMock()
    mock_model.predict.return_value = np.random.rand(300) * 1000
    mock_mlflow.xgboost.load_model.return_value = mock_model

    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

    evaluate_current_model()

    mock_mlflow.set_experiment.assert_called_once_with("Model Evaluation")
    mock_mlflow.xgboost.load_model.assert_called_once()
    mock_mlflow.log_metrics.assert_called_once()
    args, kwargs = mock_mlflow.log_metrics.call_args
    assert "daily_mse" in args[0]
    assert "daily_r2" in args[0]
