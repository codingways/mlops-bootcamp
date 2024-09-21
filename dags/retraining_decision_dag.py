import logging
import os
from datetime import timedelta

import mlflow
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from config import (
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_URI,
    MLFLOW_TRACKING_USERNAME,
    default_args,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD


def decide_retraining(**kwargs):
    try:
        client = mlflow.tracking.MlflowClient()
        experiment_name = "Model Evaluation"
        recent_runs = client.search_runs(
            experiment_ids=[
                mlflow.get_experiment_by_name(experiment_name).experiment_id
            ],
            filter_string="metrics.daily_r2 < 0.8",
            order_by=["attributes.start_time DESC"],
            max_results=7,
        )

        if (
            len(recent_runs) > 3
        ):  # Si el rendimiento fue malo más de 3 días en la última semana
            logger.info("Decision: Retraining needed")
            return "trigger_retraining"
        logger.info("Decision: Retraining not needed")
        return "skip_retraining"
    except Exception as e:
        logger.error(f"Error in retraining decision: {str(e)}")
        raise


dag_decide = DAG(
    "retraining_decision",
    default_args=default_args,
    description="Decide whether to retrain",
    schedule_interval=timedelta(minutes=7),
    catchup=False,
)

decision_task = BranchPythonOperator(
    task_id="decide_retraining",
    python_callable=decide_retraining,
    provide_context=True,
    dag=dag_decide,
)

trigger_retraining = TriggerDagRunOperator(
    task_id="trigger_retraining",
    trigger_dag_id="model_retraining_deployment",
    dag=dag_decide,
)

skip_retraining = EmptyOperator(task_id="skip_retraining", dag=dag_decide)

decision_task >> [trigger_retraining, skip_retraining]

if __name__ == "__main__":
    dag_decide.cli()
