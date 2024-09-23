import logging

import numpy as np
import pandas as pd
from airflow.operators.python import PythonOperator
from config import (
    DATA_DATABASE_URL,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    MODEL_STAGE,
    default_args,
)
from mlflow.exceptions import MlflowException
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

import mlflow
from airflow import DAG

np.float_ = np.float64
from prophet import Prophet  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def retrain_and_deploy(**kwargs):
    try:
        engine = create_engine(DATA_DATABASE_URL)
        with engine.connect() as conn:
            db_data = pd.read_sql("SELECT * FROM raw_data", conn)

        # Prepare data for Prophet
        db_data["ds"] = pd.to_datetime(db_data["date"])
        db_data["y"] = db_data["total_weight"]
        prophet_data = db_data[["ds", "y", "truck_id"]]

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
            future = test_data[["ds"]].copy()  # Use dates from test_data
            future["truck_id"] = test_data["truck_id"].values
            forecast = model.predict(future)
            y_pred = forecast["yhat"]
            y_true = test_data["y"]

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            mlflow.log_metrics({"mse": mse, "r2": r2, "rmse": rmse, "mae": mae})

            # Check if the registered model exists, if not, create it
            client = mlflow.tracking.MlflowClient()
            try:
                client.get_registered_model(MODEL_NAME)
            except MlflowException:
                client.create_registered_model(MODEL_NAME)
                logger.info(f"Created new registered model: {MODEL_NAME}")

            # Infer model signature
            # Register the model with signature and input example
            mlflow.prophet.log_model(model, "model", registered_model_name=MODEL_NAME)

            # Create new model version
            new_model_version = client.create_model_version(
                name=MODEL_NAME,
                source=f"runs:/{mlflow.active_run().info.run_id}/model",
                run_id=mlflow.active_run().info.run_id,
            )

            # Transition to production stage
            client.transition_model_version_stage(
                name=MODEL_NAME, version=new_model_version.version, stage=MODEL_STAGE
            )
            logger.info(
                f"New model deployed to production. Version: {new_model_version.version}"
            )

    except MlflowException as e:
        logger.error(f"MlflowException in model retraining and deployment: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in model retraining and deployment: {str(e)}")
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
