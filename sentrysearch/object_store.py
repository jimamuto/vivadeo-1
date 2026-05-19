"""S3-compatible object storage helpers."""

import mimetypes
from pathlib import Path

import boto3
from botocore.client import Config

from .config import Settings, get_settings


class ObjectStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.bucket = self.settings.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=self.settings.s3_endpoint_url,
            aws_access_key_id=self.settings.s3_access_key_id,
            aws_secret_access_key=self.settings.s3_secret_access_key,
            region_name=self.settings.s3_region,
            config=Config(signature_version="s3v4"),
        )
        self.public_client = boto3.client(
            "s3",
            endpoint_url=self.settings.s3_public_endpoint_url or self.settings.s3_endpoint_url,
            aws_access_key_id=self.settings.s3_access_key_id,
            aws_secret_access_key=self.settings.s3_secret_access_key,
            region_name=self.settings.s3_region,
            config=Config(signature_version="s3v4"),
        )

    def ensure_bucket(self) -> None:
        buckets = self.client.list_buckets().get("Buckets", [])
        if any(bucket["Name"] == self.bucket for bucket in buckets):
            return
        self.client.create_bucket(Bucket=self.bucket)

    def upload_file(
        self,
        path: str | Path,
        key: str,
        content_type: str | None = None,
    ) -> str:
        self.ensure_bucket()
        path = Path(path)
        guessed = mimetypes.guess_type(path.name)[0]
        extra_args = {"ContentType": content_type or guessed or "application/octet-stream"}
        self.client.upload_file(str(path), self.bucket, key, ExtraArgs=extra_args)
        return key

    def download_file(self, key: str, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(path))
        return str(path)

    def presigned_url(self, key: str, expires_in: int | None = None) -> str:
        return self.public_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in or self.settings.s3_presign_seconds,
        )


def video_object_key(video_id: str, filename: str) -> str:
    return f"videos/{video_id}/{Path(filename).name}"


def clip_object_key(clip_id: str) -> str:
    return f"clips/{clip_id}.mp4"
