# # config.py
from datetime import datetime, timedelta

from airflow.models import Variable

AWS_S3_BUCKET = Variable.get("AWS_S3_BUCKET")
AWS_ACCESS_KEY_ID = Variable.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = Variable.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = Variable.get("AWS_REGION")
MODEL_NAME = Variable.get("MODEL_NAME")
MODEL_STAGE = Variable.get("MODEL_STAGE")
MLFLOW_TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = Variable.get("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = Variable.get("MLFLOW_TRACKING_PASSWORD")

# Default DAG arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "email": ["your_email@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
