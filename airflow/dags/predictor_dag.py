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
        # Load the current model from MLflow
        current_model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

        # Load accumulated data from PostgreSQL
        engine = create_engine(DATA_DATABASE_URL)
        with engine.connect() as conn:
            accumulated_data = pd.read_sql("SELECT * FROM raw_data", conn)

        # Create future dataframe for prediction (next 30 days)
        last_date = pd.to_datetime(accumulated_data["date"]).max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)

        # Add truck_id to future_df (assuming we want predictions for all trucks)
        truck_ids = accumulated_data["truck_id"].unique()
        future_df = pd.DataFrame(
            [(date, truck_id) for date in future_dates for truck_id in truck_ids],
            columns=["date", "truck_id"],
        )

        # Add additional features to future_df
        future_df["day_of_week"] = future_df["date"].dt.dayofweek
        future_df["month"] = future_df["date"].dt.month
        future_df["day_of_year"] = future_df["date"].dt.dayofyear

        # Make predictions
        X_future = future_df[["truck_id", "day_of_week", "month", "day_of_year"]]
        future_df["predicted_weight"] = current_model.predict(X_future)

        # Select relevant columns and rename
        predictions = future_df[["date", "truck_id", "predicted_weight"]]

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
