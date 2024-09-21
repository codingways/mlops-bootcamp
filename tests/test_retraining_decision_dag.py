from unittest.mock import MagicMock, patch

import pytest

from dags.retraining_decision_dag import decide_retraining


@pytest.fixture
def mock_mlflow():
    with patch("dags.retraining_decision_dag.mlflow") as mock_mlflow:
        yield mock_mlflow


def test_decide_retraining_needs_retraining(mock_mlflow):
    mock_client = MagicMock()
    mock_client.search_runs.return_value = [MagicMock()] * 4  # 4 bad runs
    mock_mlflow.tracking.MlflowClient.return_value = mock_client

    result = decide_retraining()
    assert result == "trigger_retraining"


def test_decide_retraining_no_retraining_needed(mock_mlflow):
    mock_client = MagicMock()
    mock_client.search_runs.return_value = [MagicMock()] * 2  # 2 bad runs
    mock_mlflow.tracking.MlflowClient.return_value = mock_client

    result = decide_retraining()
    assert result == "skip_retraining"


def test_decide_retraining_exception_handling(mock_mlflow):
    mock_client = MagicMock()
    mock_client.search_runs.side_effect = Exception("Test exception")
    mock_mlflow.tracking.MlflowClient.return_value = mock_client

    with pytest.raises(Exception):
        decide_retraining()
