import logging
from datetime import timedelta

import pandas as pd
from airflow.operators.python import PythonOperator
from config import (
    DATA_DATABASE_URL,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    MODEL_STAGE,
    default_args,
)
from sqlalchemy import create_engine

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


def postgres_upsert(table, conn, keys, data_iter):
    from sqlalchemy.dialects.postgresql import insert

    # Convert data_iter to a list of dictionaries
    data = [dict(zip(keys, row)) for row in data_iter]

    # Remove duplicates based on composite keys (date and truck_id)
    unique_data = list({(row["date"], row["truck_id"]): row for row in data}.values())

    insert_statement = insert(table.table).values(unique_data)
    upsert_statement = insert_statement.on_conflict_do_update(
        constraint="predictions_truck_id_date_key",  # Changed to handle composite key
        set_={c.key: c for c in insert_statement.excluded},
    )
    conn.execute(upsert_statement)


def generate_future_predictions(**kwargs):
    try:
        # Load the current model
        current_model = mlflow.prophet.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

        # Load accumulated data from PostgreSQL
        engine = create_engine(DATA_DATABASE_URL)
        with engine.connect() as conn:
            accumulated_data = pd.read_sql("SELECT * FROM raw_data", conn)

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
        with engine.connect() as conn:
            predictions.to_sql(
                "predictions",
                conn,
                if_exists="append",
                index=False,
                method=postgres_upsert,
            )

        logger.info(
            "Generated and saved predictions for the next 30 days to the database"
        )
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"MLflow error in generating future predictions: {str(e)}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error in generating future predictions: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generating future predictions: {str(e)}")
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
