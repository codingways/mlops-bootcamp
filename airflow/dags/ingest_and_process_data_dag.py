import json
import logging
from datetime import timedelta

import pandas as pd
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from config import DATA_DATABASE_URL, MODEL_NAME, MODEL_STAGE, default_args
from sqlalchemy import create_engine
from utils.ml_helpers import check_data_drift, generate_sample_data

import mlflow
from airflow import DAG

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def postgres_upsert(table, conn, keys, data_iter):
    from sqlalchemy.dialects.postgresql import insert

    # Convert data_iter to a list of dictionaries
    data = [dict(zip(keys, row)) for row in data_iter]

    # Remove duplicates based on composite keys (date and truck_id)
    unique_data = list({(row["date"], row["truck_id"]): row for row in data}.values())

    insert_statement = insert(table.table).values(unique_data)
    upsert_statement = insert_statement.on_conflict_do_update(
        constraint="raw_data_truck_id_date_key",
        set_={c.key: c for c in insert_statement.excluded},
    )
    conn.execute(upsert_statement)


def ingest_and_process_data(**kwargs):
    try:
        engine = create_engine(DATA_DATABASE_URL)
        with engine.connect() as conn:
            db_data = pd.read_sql("SELECT * FROM raw_data", conn)

        db_data.drop(columns=["id"], inplace=True)

        if db_data.empty:
            logger.info("Initializing historical data")
            new_data = generate_sample_data(num_days=365)
        else:
            day_data = generate_sample_data()
            drift_detected, drift_report = check_data_drift(day_data, db_data)
            if drift_detected:
                logger.warning(
                    f"Data drift detected: {json.dumps(drift_report, indent=2)}"
                )
                # TODO: Implement logic to handle data drift
            new_data = (
                pd.concat([db_data, day_data], ignore_index=True)
                if not day_data.empty
                else db_data
            )

        # Save accumulated data
        with engine.connect() as conn:
            new_data.to_sql(
                "raw_data",
                conn,
                if_exists="append",
                index=False,
                method=postgres_upsert,
            )

        logger.info(
            f"Processed and accumulated {len(new_data)} new records. Total records: {len(new_data)}"
        )
    except Exception as e:
        logger.error(f"Error in data ingestion and processing: {str(e)}", exc_info=True)
        raise


def check_model_exists():
    try:
        mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        return True
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"Error checking model existence: {str(e)}")
        return False


def check_model_exists_and_branch(**kwargs):
    if not check_model_exists():
        return "trigger_model_training"
    return "skip_model_training"


dag_ingest_process = DAG(
    "ingest_and_process_data",
    default_args=default_args,
    description="Ingest and process daily data",
    schedule_interval=timedelta(days=1),
    catchup=False,
)

ingest_process_task = PythonOperator(
    task_id="ingest_and_process_data",
    python_callable=ingest_and_process_data,
    provide_context=True,
    dag=dag_ingest_process,
)

check_model_branch = BranchPythonOperator(
    task_id="check_model_exists",
    python_callable=check_model_exists_and_branch,
    dag=dag_ingest_process,
)

trigger_model_training = TriggerDagRunOperator(
    task_id="trigger_model_training",
    trigger_dag_id="model_retraining_deployment",
    dag=dag_ingest_process,
)

skip_model_training = EmptyOperator(
    task_id="skip_model_training", dag=dag_ingest_process
)

(
    ingest_process_task
    >> check_model_branch
    >> [trigger_model_training, skip_model_training]
)

if __name__ == "__main__":
    dag_ingest_process.cli()
