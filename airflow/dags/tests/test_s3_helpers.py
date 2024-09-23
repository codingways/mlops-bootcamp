from unittest.mock import MagicMock, patch

import pytest
from utils.s3_helpers import load_data_from_s3, save_data_to_s3


@pytest.fixture
def mock_s3_client():
    with patch("utils.s3_helpers.s3") as mock:
        yield mock


def test_load_data_from_s3(mock_s3_client):
    mock_s3_client.get_object.return_value = {
        "Body": MagicMock(read=lambda: b"test data")
    }
    data = load_data_from_s3("test-bucket", "test-key")
    assert data == b"test data"
    mock_s3_client.get_object.assert_called_once_with(
        Bucket="test-bucket", Key="test-key"
    )


def test_save_data_to_s3(mock_s3_client):
    save_data_to_s3("test data", "test-bucket", "test-key")
    mock_s3_client.put_object.assert_called_once_with(
        Bucket="test-bucket", Key="test-key", Body="test data"
    )
