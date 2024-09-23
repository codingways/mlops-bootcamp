import io
import logging
from datetime import timedelta

import pandas as pd
from airflow.operators.python import PythonOperator
from config import (
    AWS_S3_BUCKET,
    DATA_DATABASE_URL,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    MODEL_STAGE,
    default_args,
)
from sqlalchemy import MetaData, Table, create_engine
from utils.s3_helpers import load_data_from_s3

import mlflow
from airflow import DAG

print(f"DATA_DATABASE_URL: {DATA_DATABASE_URL}")
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def generate_future_predictions(**kwargs):
    try:
        # Load the current model
        current_model = mlflow.prophet.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

        # Load accumulated data from S3
        parquet_data = load_data_from_s3(AWS_S3_BUCKET, "accumulated_data.parquet")
        accumulated_data = pd.read_parquet(io.BytesIO(parquet_data))

        # Prepare data for Prophet
        accumulated_data["ds"] = pd.to_datetime(accumulated_data["date"])
        accumulated_data["y"] = accumulated_data["total_weight"]

        # Create future dataframe for prediction (next 30 days)
        last_date = accumulated_data["ds"].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)

        # Add truck_id to future_df (assuming we want predictions for all trucks)
        truck_ids = accumulated_data["truck_id"].unique()
        future_df = pd.DataFrame(
            [(date, truck_id) for date in future_dates for truck_id in truck_ids],
            columns=["ds", "truck_id"],
        )

        # Make predictions
        forecast = current_model.predict(future_df)

        # Add truck_id to the forecast
        forecast["truck_id"] = future_df["truck_id"].values

        # Select relevant columns and rename
        predictions = forecast[["ds", "truck_id", "yhat", "yhat_lower", "yhat_upper"]]
        predictions.columns = [
            "date",
            "truck_id",
            "predicted_weight",
            "lower_bound",
            "upper_bound",
        ]

        # Connect to the database and insert predictions
        with create_engine(DATA_DATABASE_URL).connect() as conn:
            metadata = MetaData()
            predictions_table = Table("predictions", metadata, autoload_with=conn)
            conn.execute(predictions_table.delete())
            predictions.to_sql("predictions", conn, if_exists="append", index=False)

        logger.info(
            "Generated and saved predictions for the next 30 days to the database"
        )
    except Exception as e:
        logger.error(f"Error in generating future predictions: {str(e)}")
        raise


# Define the DAG
dag_predictions = DAG(
    "predictor_dag",
    default_args=default_args,
    description="Generate daily predictions for the next 30 days",
    schedule_interval=timedelta(days=1),
    catchup=False,
)

# Define the task
generate_predictions_task = PythonOperator(
    task_id="generate_future_predictions",
    python_callable=generate_future_predictions,
    provide_context=True,
    dag=dag_predictions,
)

if __name__ == "__main__":
    dag_predictions.cli()
