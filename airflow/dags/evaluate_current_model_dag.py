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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

import mlflow
from airflow import DAG

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def evaluate_current_model(**kwargs):
    try:
        mlflow.set_experiment("Model Evaluation")
        current_model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

        engine = create_engine(DATA_DATABASE_URL)
        with engine.connect() as conn:
            db_data = pd.read_sql("SELECT * FROM raw_data", conn)

        X = db_data[["truck_id", "day_of_week", "month", "day_of_year"]]
        y = db_data["total_weight"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Make predictions
        y_pred = current_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        logger.info(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R2: {r2}")

        with mlflow.start_run():
            mlflow.log_metrics(
                {
                    "daily_mse": mse,
                    "daily_rmse": rmse,
                    "daily_r2": r2,
                    "daily_mae": mae,
                }
            )

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
