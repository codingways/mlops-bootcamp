# utils/ml_helpers.py
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from config import AWS_S3_BUCKET
from scipy.stats import ks_2samp
from utils.s3_helpers import load_data_from_s3, save_data_to_s3


def check_data_drift(new_data, reference_data, threshold=0.05):
    drift_detected = False
    drift_report = {}

    for column in new_data.columns:
        if new_data[column].dtype in ["int64", "float64"]:
            ks_statistic, p_value = ks_2samp(reference_data[column], new_data[column])
            if p_value < threshold:
                drift_detected = True
                drift_report[column] = {
                    "ks_statistic": ks_statistic,
                    "p_value": p_value,
                }

    return drift_detected, drift_report


def get_or_create_random_values(trucks=10):
    try:
        random_values = load_data_from_s3(AWS_S3_BUCKET, "random_values.json")
        random_values = json.loads(random_values)
    except (json.JSONDecodeError, Exception):
        random_values = {
            "truck_capacities": np.random.uniform(1500, 3000, trucks).tolist(),
            "truck_reliability": np.random.uniform(0.8, 1.2, trucks).tolist(),
            # Sábado y domingo
            "weekend_multipliers": np.random.uniform(1.1, 1.2, 2).tolist(),
            "monthly_variations": np.random.normal(0, 100, 12).tolist(),  # Una por mes
            # Para los últimos 15 días del año
            "christmas_multipliers": np.random.uniform(1.2, 1.4, 15).tolist(),
        }
        save_data_to_s3(json.dumps(random_values), AWS_S3_BUCKET, "random_values.json")

    return random_values


def generate_sample_data(num_days=1, trucks=10):
    data = []
    current_date = datetime.now() - timedelta(days=num_days - 1)
    random_values = get_or_create_random_values(trucks)

    # Use provided truck-specific characteristics
    truck_capacities = random_values["truck_capacities"]
    truck_reliability = random_values["truck_reliability"]
    weekend_multipliers = random_values["weekend_multipliers"]
    monthly_variations = random_values["monthly_variations"]
    christmas_multipliers = random_values["christmas_multipliers"]

    def generate_total_weight(truck_id, day_of_week, month, day_of_year):
        base_weight = truck_capacities[truck_id - 1] * truck_reliability[truck_id - 1]

        # Weekend effect
        if day_of_week in [5, 6]:
            base_weight *= weekend_multipliers[day_of_week - 5]

        # Seasonal effect (assuming Southern Hemisphere)
        season_effect = np.sin(2 * np.pi * day_of_year / 365) * 200 + 200
        base_weight += season_effect

        # Monthly variation
        base_weight += monthly_variations[month - 1]

        # Special events (e.g., holidays)
        if month == 12 and day_of_year >= 350:  # Christmas period
            base_weight *= christmas_multipliers[day_of_year - 350 - 1]

        # Long-term trend (slight increase over time)
        trend = day_of_year * 0.5
        base_weight += trend

        # Ensure weight is between 500 and 4000 kg
        return max(500, min(4000, round(base_weight, 2)))

    for day in range(num_days):
        for truck_id in range(1, trucks + 1):
            day_of_week = current_date.weekday()
            month = current_date.month
            day_of_year = current_date.timetuple().tm_yday

            total_weight = generate_total_weight(
                truck_id, day_of_week, month, day_of_year
            )

            data.append(
                {
                    "date": current_date.date(),
                    "truck_id": truck_id,
                    "day_of_week": day_of_week,
                    "month": month,
                    "day_of_year": day_of_year,
                    "total_weight": total_weight,
                }
            )

        current_date += timedelta(days=1)

    df = pd.DataFrame(data)
    return df
