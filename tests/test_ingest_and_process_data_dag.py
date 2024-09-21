from unittest.mock import MagicMock, patch

import pytest

from dags.ingest_and_process_data_dag import (
    check_model_exists_and_branch,
    ingest_and_process_data,
)


@pytest.fixture
def mock_s3_helpers():
    with patch(
        "dags.ingest_and_process_data_dag.load_data_from_s3"
    ) as mock_load, patch(
        "dags.ingest_and_process_data_dag.save_data_to_s3"
    ) as mock_save:
        yield mock_load, mock_save


@pytest.fixture
def mock_ml_helpers():
    with patch(
        "dags.ingest_and_process_data_dag.generate_sample_data"
    ) as mock_generate, patch(
        "dags.ingest_and_process_data_dag.check_data_drift"
    ) as mock_drift:
        yield mock_generate, mock_drift


def test_ingest_and_process_data(mock_s3_helpers, mock_ml_helpers):
    mock_load, mock_save = mock_s3_helpers
    mock_generate, mock_drift = mock_ml_helpers

    mock_generate.return_value = MagicMock()
    mock_load.return_value = b"mock_parquet_data"
    mock_drift.return_value = (False, {})

    ingest_and_process_data()

    mock_generate.assert_called_once()
    mock_load.assert_called_once()
    mock_save.assert_called_once()
    mock_drift.assert_called_once()


@patch("dags.ingest_and_process_data_dag.check_model_exists")
def test_check_model_exists_and_branch(mock_check_model):
    mock_check_model.return_value = True
    assert check_model_exists_and_branch() == "skip_model_training"

    mock_check_model.return_value = False
    assert check_model_exists_and_branch() == "trigger_model_training"
