FROM ghcr.io/mlflow/mlflow:v2.16.2

RUN apt-get update && apt-get install -y curl
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt