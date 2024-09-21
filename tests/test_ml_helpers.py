import numpy as np
import pandas as pd
import pytest

from dags.utils.ml_helpers import check_data_drift, generate_sample_data


@pytest.fixture
def sample_data():
    return generate_sample_data(num_days=10, trucks=5)


def test_generate_sample_data():
    data = generate_sample_data(num_days=5, trucks=3)
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 15  # 5 days * 3 trucks
    assert set(data.columns) == {
        "date",
        "truck_id",
        "day_of_week",
        "month",
        "day_of_year",
        "total_weight",
    }
    assert data["truck_id"].nunique() == 3
    assert data["total_weight"].between(500, 4000).all()


def test_check_data_drift():
    np.random.seed(42)
    data1 = pd.DataFrame(
        {"A": np.random.normal(0, 1, 1000), "B": np.random.normal(0, 1, 1000)}
    )
    data2 = pd.DataFrame(
        {"A": np.random.normal(0, 1, 1000), "B": np.random.normal(2, 1, 1000)}
    )

    drift_detected, drift_report = check_data_drift(data1, data2)
    assert drift_detected
    assert "B" in drift_report
    assert "A" not in drift_report
