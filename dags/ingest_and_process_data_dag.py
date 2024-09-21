import io
import json
import logging
from datetime import timedelta

import mlflow
import pandas as pd
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from config import AWS_S3_BUCKET, MODEL_NAME, MODEL_STAGE, default_args
from utils.ml_helpers import check_data_drift, generate_sample_data
from utils.s3_helpers import load_data_from_s3, save_data_to_s3

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def ingest_and_process_data(**kwargs):
    try:
        # Generar datos para el día actual
        new_data = generate_sample_data()

        try:
            parquet_data = load_data_from_s3(AWS_S3_BUCKET, "accumulated_data.parquet")
            accumulated_data = pd.read_parquet(io.BytesIO(parquet_data))

            # Verificar data drift
            if not accumulated_data.empty:
                drift_detected, drift_report = check_data_drift(
                    new_data, accumulated_data
                )
                if drift_detected:
                    logger.warning(
                        f"Data drift detected: {json.dumps(drift_report, indent=2)}"
                    )
                    # Aquí podrías implementar lógica adicional, como enviar alertas

            # Agregar nuevos datos
            accumulated_data = pd.concat(
                [accumulated_data, new_data], ignore_index=True
            )
        except Exception as e:
            logger.warning(f"Error al cargar datos acumulados existentes: {str(e)}")
            logger.info("Iniciando con un DataFrame vacío para datos acumulados.")
            accumulated_data = generate_sample_data(num_days=365)

        # Guardar datos acumulados
        save_data_to_s3(
            accumulated_data.to_parquet(), AWS_S3_BUCKET, "accumulated_data.parquet"
        )

        logger.info(
            f"Processed and accumulated {len(new_data)} new records. Total records: {len(accumulated_data)}"
        )
    except Exception as e:
        logger.error(f"Error in data ingestion and processing: {str(e)}")
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
    schedule_interval=timedelta(minutes=1),
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
