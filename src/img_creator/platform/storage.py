import re
from pathlib import Path
import os
import uuid


class Storage:
    """Private keys only. Browser never receives a bucket listing or storage credentials."""

    def __init__(self, settings):
        self.root = settings.storage_dir
        self.bucket = settings.s3_bucket
        self.client = None
        if self.bucket:
            import boto3

            self.client = boto3.client("s3", endpoint_url=settings.s3_endpoint)
        else:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate(key):
        if not re.fullmatch(r"[a-zA-Z0-9/_\-.]+", key) or ".." in key or key.startswith("/"):
            raise ValueError("Invalid storage key")

    def put(self, key, data, content_type="application/octet-stream"):
        self.validate(key)
        if self.client:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        else:
            path = self.root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
            try:
                temporary.write_bytes(data)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    def get(self, key):
        self.validate(key)
        if self.client:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return (self.root / key).read_bytes()

    def delete(self, key):
        self.validate(key)
        if self.client:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        else:
            Path(self.root / key).unlink(missing_ok=True)
