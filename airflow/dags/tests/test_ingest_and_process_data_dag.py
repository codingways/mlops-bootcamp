from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from config import DATA_DATABASE_URL
from ingest_and_process_data_dag import (
    check_model_exists_and_branch,
    ingest_and_process_data,
)
from mlflow.exceptions import MlflowException

import mlflow


@pytest.fixture
def mock_mlflow_tracking_uri():
    original_uri = mlflow.get_tracking_uri()
    test_uri = ""
    mlflow.set_tracking_uri(test_uri)
    yield test_uri
    mlflow.set_tracking_uri(original_uri)


def test_check_model_exists_and_branch_model_not_exists(mock_mlflow_tracking_uri):
    with patch("ingest_and_process_data_dag.load_model") as mock_load_model:
        mock_load_model.side_effect = MlflowException("Model not found")

        result = check_model_exists_and_branch()

        assert result == "trigger_model_training"


def test_check_model_exists_and_branch_model_exists(mock_mlflow_tracking_uri):
    with patch("ingest_and_process_data_dag.load_model") as mock_load_model:
        mock_load_model.return_value = MagicMock()

        result = check_model_exists_and_branch()

        assert result == "skip_model_training"


def test_ingest_and_process_data():
    with patch(
        "ingest_and_process_data_dag.create_engine"
    ) as mock_create_engine, patch(
        "ingest_and_process_data_dag.generate_sample_data"
    ) as mock_generate_sample_data, patch(
        "ingest_and_process_data_dag.check_data_drift"
    ) as mock_check_data_drift:
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        with patch("ingest_and_process_data_dag.pd.read_sql") as mock_read_sql, patch(
            "ingest_and_process_data_dag.pd.DataFrame.to_sql"
        ) as mock_to_sql:
            mock_read_sql.return_value = pd.DataFrame(
                {
                    "id": [1, 2, 3, 4, 5],
                    "date": pd.date_range(start="2023-01-01", periods=5),
                    "truck_id": [1, 2, 3, 4, 5],
                    "total_weight": [1000, 1500, 2000, 2500, 3000],
                    "day_of_week": [0, 1, 2, 3, 4],
                    "month": [1, 1, 1, 1, 1],
                    "day_of_year": [1, 2, 3, 4, 5],
                }
            )

            mock_generate_sample_data.return_value = pd.DataFrame(
                {
                    "date": [pd.Timestamp("2023-01-06")],
                    "truck_id": [6],
                    "total_weight": [3500],
                    "day_of_week": [5],
                    "month": [1],
                    "day_of_year": [6],
                }
            )

            mock_check_data_drift.return_value = (False, {})

            ingest_and_process_data()

            mock_create_engine.assert_called_once_with(DATA_DATABASE_URL)
            mock_engine.connect.assert_called()
            mock_read_sql.assert_called_once_with("SELECT * FROM raw_data", mock_conn)
            mock_generate_sample_data.assert_called_once()
            mock_check_data_drift.assert_called_once()
            mock_to_sql.assert_called_once()
