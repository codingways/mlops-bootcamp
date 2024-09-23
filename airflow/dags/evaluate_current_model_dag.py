import logging
from datetime import timedelta

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
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine

import mlflow
from airflow import DAG

np.float_ = np.float64

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def evaluate_current_model(**kwargs):
    try:
        mlflow.set_experiment("Model Evaluation")
        current_model = mlflow.prophet.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

        engine = create_engine(DATA_DATABASE_URL)
        with engine.connect() as conn:
            query = "SELECT * FROM raw_data ORDER BY date DESC LIMIT 300"
            accumulated_data = pd.read_sql(query, conn)

        # Prepare data for Prophet
        recent_data = accumulated_data.iloc[
            ::-1
        ]  # Reverse the order to maintain chronological order
        recent_data["ds"] = pd.to_datetime(recent_data["date"])
        recent_data["y"] = recent_data["total_weight"]

        # Create future dataframe for prediction
        future = recent_data[["ds", "truck_id"]]

        # Make predictions
        forecast = current_model.predict(future)

        # Extract actual values and predictions
        y_true = recent_data["y"]
        y_pred = forecast["yhat"]

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # Log metrics
        logger.info(f"MSE: {mse}, RMSE: {rmse}, R2: {r2}")

        with mlflow.start_run():
            mlflow.log_metrics({"daily_mse": mse, "daily_rmse": rmse, "daily_r2": r2})

        logger.info(
            f"Daily evaluation (last 300 samples) - MSE: {mse}, RMSE: {rmse}, R2: {r2}"
        )
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"MLflow error in model evaluation: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"General error in model evaluation: {str(e)}")
        raise


dag_evaluate = DAG(
    "evaluate_current_model",
    default_args=default_args,
    description="Evaluate current model daily",
    schedule_interval=timedelta(days=1),
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
