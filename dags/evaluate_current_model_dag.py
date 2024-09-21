import io
import logging
import os
from datetime import timedelta

import mlflow
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from config import (
    AWS_S3_BUCKET,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_URI,
    MLFLOW_TRACKING_USERNAME,
    MODEL_NAME,
    MODEL_STAGE,
    default_args,
)
from sklearn.metrics import mean_squared_error, r2_score
from utils.s3_helpers import load_data_from_s3

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD


def evaluate_current_model(**kwargs):
    try:
        mlflow.set_experiment("Model Evaluation")
        current_model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

        # Load accumulated data
        parquet_data = load_data_from_s3(AWS_S3_BUCKET, "accumulated_data.parquet")
        accumulated_data = pd.read_parquet(io.BytesIO(parquet_data))

        # Get the last 10 samples
        recent_data = accumulated_data.iloc[-300:]

        X = recent_data[["truck_id", "day_of_week", "month"]]
        y = recent_data["total_weight"]

        predictions = current_model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)

        with mlflow.start_run():
            mlflow.log_metrics({"daily_mse": mse, "daily_r2": r2})

        logger.info(f"Daily evaluation (last 10 samples) - MSE: {mse}, R2: {r2}")
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise


dag_evaluate = DAG(
    "evaluate_current_model",
    default_args=default_args,
    description="Evaluate current model daily",
    schedule_interval=timedelta(minutes=1),
    catchup=False,
)

evaluate_task = PythonOperator(
    task_id="evaluate_current_model",
    python_callable=evaluate_current_model,
    provide_context=True,
    dag=dag_evaluate,
)

if __name__ == "__main__":
    dag_evaluate.cli()
