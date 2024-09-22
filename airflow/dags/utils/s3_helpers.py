import boto3
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, MINIO_ENDPOINT

s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=boto3.session.Config(signature_version="s3v4"),
    region_name="us-east-1",
)


def load_data_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def save_data_to_s3(data, bucket, key):
    s3.put_object(Bucket=bucket, Key=key, Body=data)
