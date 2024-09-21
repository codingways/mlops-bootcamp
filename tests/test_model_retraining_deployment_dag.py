from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from dags.model_retraining_deployment_dag import retrain_and_deploy


@pytest.fixture
def mock_mlflow():
    with patch("dags.model_retraining_deployment_dag.mlflow") as mock_mlflow:
        yield mock_mlflow


@pytest.fixture
def mock_s3_helpers():
    with patch("dags.model_retraining_deployment_dag.load_data_from_s3") as mock_load:
        yield mock_load


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=100),
            "total_weight": np.random.rand(100) * 1000,
            "truck_id": np.random.randint(1, 6, 100),
        }
    )


def test_retrain_and_deploy(mock_mlflow, mock_s3_helpers, sample_data):
    mock_s3_helpers.return_value = sample_data.to_parquet()

    mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()
    mock_mlflow.tracking.MlflowClient.return_value = MagicMock()

    retrain_and_deploy()

    mock_mlflow.set_experiment.assert_called_once()
    mock_mlflow.log_metrics.assert_called_once()
    mock_mlflow.prophet.log_model.assert_called_once()
