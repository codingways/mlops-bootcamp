from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from config import MODEL_NAME
from model_retraining_deployment_dag import retrain_and_deploy


@pytest.fixture
def mock_mlflow():
    with patch("model_retraining_deployment_dag.mlflow") as mock:
        yield mock


@pytest.fixture
def mock_engine():
    with patch("model_retraining_deployment_dag.create_engine") as mock:
        yield mock


@pytest.fixture
def mock_xgb():
    with patch("model_retraining_deployment_dag.xgb") as mock:
        yield mock


@pytest.fixture
def mock_client():
    with patch("model_retraining_deployment_dag.mlflow.tracking.MlflowClient") as mock:
        yield mock


def test_retrain_and_deploy(mock_mlflow, mock_engine, mock_xgb, mock_client):
    mock_engine_instance = MagicMock()
    mock_engine.return_value = mock_engine_instance

    test_data = pd.DataFrame(
        {
            "truck_id": range(1, 11),
            "day_of_week": range(0, 10),
            "month": [1] * 10,
            "day_of_year": range(1, 11),
            "total_weight": [1000 + i * 500 for i in range(10)],
        }
    )

    with patch("pandas.read_sql", return_value=test_data):
        # Mock the XGBoost model
        mock_model = MagicMock()
        mock_xgb.XGBRegressor.return_value = mock_model

        mock_model.predict.return_value = [950, 1450]

        mock_mlflow_client_instance = MagicMock()
        mock_client.return_value = mock_mlflow_client_instance

        mock_mlflow_run = MagicMock()
        mock_mlflow.active_run.return_value = mock_mlflow_run
        mock_mlflow_run.info.run_id = "test_run_id"

        retrain_and_deploy()

    mock_mlflow.set_experiment.assert_called_once_with(
        "Model Retraining and Deployment"
    )
    mock_xgb.XGBRegressor.assert_called_once_with(
        objective="reg:squarederror", random_state=42
    )
    mock_model.fit.assert_called_once()
    mock_model.predict.assert_called_once()
    mock_mlflow.log_metrics.assert_called_once()
    mock_mlflow.xgboost.log_model.assert_called_once_with(
        mock_model, "model", registered_model_name=MODEL_NAME
    )
    mock_mlflow_client_instance.transition_model_version_stage.assert_called_once()
