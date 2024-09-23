import json
from unittest.mock import patch

import numpy as np
import pandas as pd
from utils.ml_helpers import check_data_drift, generate_sample_data


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


@patch("utils.ml_helpers.load_data_from_s3")
def test_generate_sample_data(mock_load_data):
    mock_random_values = {
        "truck_capacities": np.random.uniform(1500, 3000, 3).tolist(),
        "truck_reliability": np.random.uniform(0.8, 1.2, 3).tolist(),
        "weekend_multipliers": np.random.uniform(1.1, 1.2, 2).tolist(),
        "monthly_variations": np.random.normal(0, 100, 12).tolist(),
        "christmas_multipliers": np.random.uniform(1.2, 1.4, 15).tolist(),
    }
    mock_load_data.return_value = json.dumps(mock_random_values)

    data = generate_sample_data(num_days=5, trucks=3)
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 15
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

    mock_load_data.assert_called_once()
