"""Tests for production configuration."""

from sentrysearch.config import Settings


def test_settings_reads_production_env(monkeypatch):
    monkeypatch.setenv("SENTRYSEARCH_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db:5432/app")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/1")
    monkeypatch.setenv("S3_BUCKET", "videos")

    settings = Settings()

    assert settings.api_key == "secret"
    assert settings.database_url.endswith("/app")
    assert settings.redis_url == "redis://redis:6379/1"
    assert settings.s3_bucket == "videos"

