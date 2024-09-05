import io
import logging
import os

import mlflow
import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from config import (
    AWS_S3_BUCKET,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_URI,
    MLFLOW_TRACKING_USERNAME,
    MODEL_NAME,
    MODEL_STAGE,
    default_args,
)
from mlflow.exceptions import MlflowException
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from utils.s3_helpers import load_data_from_s3

np.float_ = np.float64
from prophet import Prophet  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD


def retrain_and_deploy(**kwargs):
    try:
        # Load accumulated data
        parquet_data = load_data_from_s3(AWS_S3_BUCKET, "accumulated_data.parquet")
        data = pd.read_parquet(io.BytesIO(parquet_data))

        # Prepare data for Prophet
        data["ds"] = pd.to_datetime(data["date"])  # Assuming 'date' column exists
        data["y"] = data["total_weight"]
        prophet_data = data[["ds", "y", "truck_id"]]

        # Split data
        train_data, test_data = train_test_split(
            prophet_data, test_size=0.2, random_state=42
        )

        experiment_name = "Model Retraining and Deployment"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Training
            model = Prophet()
            model.add_regressor("truck_id")
            model.fit(train_data)

            # Evaluate
            future = model.make_future_dataframe(periods=len(test_data))
            print(f"Length of future: {len(future)}")
            print(f"Length of test_data: {len(test_data)}")
            print(f"Length of test_data['truck_id']: {len(test_data['truck_id'])}")
            future["truck_id"] = test_data["truck_id"].values
            forecast = model.predict(future)
            y_pred = forecast.tail(len(test_data))["yhat"]
            y_true = test_data["y"]

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            mlflow.log_metrics({"mse": mse, "r2": r2, "rmse": rmse, "mae": mae})

            # Verificar si el modelo registrado existe, si no, crearlo
            client = mlflow.tracking.MlflowClient()
            try:
                client.get_registered_model(MODEL_NAME)
            except MlflowException:
                client.create_registered_model(MODEL_NAME)
                logger.info(f"Created new registered model: {MODEL_NAME}")

            # Inferir la firma del modelo
            # Registrar el modelo con firma y ejemplo de entrada
            mlflow.prophet.log_model(model, "model", registered_model_name=MODEL_NAME)

            # Crear nueva versión del modelo
            new_model_version = client.create_model_version(
                name=MODEL_NAME,
                source=f"runs:/{mlflow.active_run().info.run_id}/model",
                run_id=mlflow.active_run().info.run_id,
            )

            # Transición a la etapa de producción
            client.transition_model_version_stage(
                name=MODEL_NAME, version=new_model_version.version, stage=MODEL_STAGE
            )
            logger.info(
                f"New model deployed to production. Version: {new_model_version.version}"
            )

    except Exception as e:
        logger.error(f"Error in model retraining and deployment: {str(e)}")
        raise


dag_retrain = DAG(
    "model_retraining_deployment",
    default_args=default_args,
    description="Retrain and potentially deploy new model",
    schedule_interval=None,  # Triggered externally
    catchup=False,
)

retrain_task = PythonOperator(
    task_id="retrain_and_deploy",
    python_callable=retrain_and_deploy,
    provide_context=True,
    dag=dag_retrain,
)

if __name__ == "__main__":
    dag_retrain.cli()
