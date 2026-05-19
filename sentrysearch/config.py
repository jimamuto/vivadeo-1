"""Runtime configuration for local and production deployments."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed application settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: str = Field("change-me", alias="SENTRYSEARCH_API_KEY")
    api_url: str | None = Field(None, alias="SENTRYSEARCH_API_URL")

    database_url: str = Field(
        "postgresql+psycopg://sentrysearch:sentrysearch@localhost:5432/sentrysearch",
        alias="DATABASE_URL",
    )
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")

    s3_endpoint_url: str = Field("http://localhost:9000", alias="S3_ENDPOINT_URL")
    s3_public_endpoint_url: str | None = Field(None, alias="S3_PUBLIC_ENDPOINT_URL")
    s3_bucket: str = Field("sentrysearch", alias="S3_BUCKET")
    s3_access_key_id: str = Field("minioadmin", alias="S3_ACCESS_KEY_ID")
    s3_secret_access_key: str = Field("minioadmin", alias="S3_SECRET_ACCESS_KEY")
    s3_region: str = Field("us-east-1", alias="S3_REGION")
    s3_presign_seconds: int = Field(3600, alias="S3_PRESIGN_SECONDS")

    modal_app: str = Field(
        "sentrysearch-qwen3-vl-embedding-2b",
        alias="SENTRYSEARCH_MODAL_APP",
    )
    modal_class: str = Field("QwenEmbedder", alias="SENTRYSEARCH_MODAL_CLASS")
    modal_timeout: int = Field(900, alias="SENTRYSEARCH_MODAL_TIMEOUT")

    chunk_duration: int = Field(30, alias="SENTRYSEARCH_CHUNK_DURATION")
    chunk_overlap: int = Field(5, alias="SENTRYSEARCH_CHUNK_OVERLAP")
    batch_size: int = Field(4, alias="SENTRYSEARCH_BATCH_SIZE")
    preprocess: bool = Field(True, alias="SENTRYSEARCH_PREPROCESS")
    target_resolution: int = Field(480, alias="SENTRYSEARCH_TARGET_RESOLUTION")
    target_fps: int = Field(5, alias="SENTRYSEARCH_TARGET_FPS")
    skip_still: bool = Field(False, alias="SENTRYSEARCH_SKIP_STILL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
