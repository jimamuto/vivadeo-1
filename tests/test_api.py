"""Tests for the production FastAPI surface."""

from fastapi.testclient import TestClient

from sentrysearch.api import app


class _FakeConn:
    def execute(self, *args, **kwargs):
        return None


class _FakeBegin:
    def __enter__(self):
        return _FakeConn()

    def __exit__(self, *args):
        return False


class _FakeEngine:
    def begin(self):
        return _FakeBegin()


class _FakeObjectStore:
    def ensure_bucket(self):
        return None


def _disable_startup_io(monkeypatch):
    monkeypatch.setattr("sentrysearch.api.make_engine", lambda: _FakeEngine())
    monkeypatch.setattr("sentrysearch.api.ObjectStore", lambda: _FakeObjectStore())
    monkeypatch.setattr("sentrysearch.api.Base.metadata.create_all", lambda bind: None)


def test_healthz_without_api_key(monkeypatch):
    _disable_startup_io(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_stats_rejects_missing_api_key(monkeypatch):
    _disable_startup_io(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/v1/stats")

    assert response.status_code == 401
