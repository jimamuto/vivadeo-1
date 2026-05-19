"""Tests for CLI production API mode."""

from click.testing import CliRunner

from sentrysearch.cli import cli


def test_stats_uses_api_mode(monkeypatch):
    monkeypatch.setenv("SENTRYSEARCH_API_URL", "http://api.test")
    monkeypatch.setenv("SENTRYSEARCH_API_KEY", "secret")

    def fake_request(method, path, payload=None):
        assert method == "GET"
        assert path == "/v1/stats"
        return {"total_videos": 2, "total_chunks": 9}

    monkeypatch.setattr("sentrysearch.cli._api_request", fake_request)
    result = CliRunner().invoke(cli, ["stats"])

    assert result.exit_code == 0
    assert "Total videos:  2" in result.output
    assert "Total chunks:  9" in result.output


def test_index_uploads_file_in_api_mode(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setenv("SENTRYSEARCH_API_URL", "http://api.test")
    monkeypatch.setenv("SENTRYSEARCH_API_KEY", "secret")

    def fake_upload(path):
        assert path == str(video)
        return {"id": "job-1", "video_id": "video-1"}

    monkeypatch.setattr("sentrysearch.cli._api_upload", fake_upload)
    result = CliRunner().invoke(cli, ["index", str(video)])

    assert result.exit_code == 0
    assert "Uploaded and queued ingest job: job-1" in result.output


def test_index_directory_uses_mounted_path_in_api_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("SENTRYSEARCH_API_URL", "http://api.test")
    monkeypatch.setenv("SENTRYSEARCH_API_KEY", "secret")

    def fake_request(method, path, payload=None):
        assert method == "POST"
        assert path == "/v1/videos/local-path"
        assert payload == {"path": str(tmp_path)}
        return {"id": "job-2", "video_id": "video-2"}

    monkeypatch.setattr("sentrysearch.cli._api_request", fake_request)
    result = CliRunner().invoke(cli, ["index", str(tmp_path)])

    assert result.exit_code == 0
    assert "Queued mounted-path ingest job: job-2" in result.output
