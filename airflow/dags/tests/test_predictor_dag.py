from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from predictor_dag import generate_future_predictions

import mlflow


@pytest.fixture
def mock_mlflow_tracking_uri():
    original_uri = mlflow.get_tracking_uri()
    test_uri = ""
    mlflow.set_tracking_uri(test_uri)
    yield test_uri
    mlflow.set_tracking_uri(original_uri)


@pytest.fixture
def mock_engine():
    with patch("predictor_dag.create_engine") as mock:
        yield mock


@patch("predictor_dag.load_model")
@patch("predictor_dag.pd.read_sql")
@patch("predictor_dag.pd.DataFrame.to_sql")
def test_generate_future_predictions(
    mock_to_sql, mock_read_sql, mock_load_model, mock_engine, mock_mlflow_tracking_uri
):
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model

    test_data = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=3),
            "truck_id": [1, 2, 3],
            "total_weight": [1000, 1500, 2000],
        }
    )
    mock_read_sql.return_value = test_data

    num_future_dates = 30
    num_truck_ids = test_data["truck_id"].nunique()
    expected_num_predictions = num_future_dates * num_truck_ids

    mock_model.predict.return_value = [1000] * expected_num_predictions

    generate_future_predictions()

    mock_load_model.assert_called_once()
    mock_read_sql.assert_called_once()
    mock_to_sql.assert_called_once()
