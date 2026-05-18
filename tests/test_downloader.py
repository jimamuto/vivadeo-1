"""Tests for URL video downloads."""

from unittest.mock import MagicMock, patch

import pytest

from sentrysearch.downloader import VideoDownloadError, download_video_url


@patch("yt_dlp.YoutubeDL")
def test_download_video_url_returns_mp4(mock_ydl_cls, tmp_path):
    out = tmp_path / "Remote_job_video [abc123].mp4"
    out.write_bytes(b"video")

    ydl = MagicMock()
    ydl.extract_info.return_value = {"id": "abc123", "title": "Remote job video"}
    ydl.prepare_filename.return_value = str(out)
    mock_ydl_cls.return_value.__enter__.return_value = ydl

    result = download_video_url(
        "https://youtu.be/example",
        output_dir=tmp_path,
        max_height=360,
    )

    assert result == str(out)
    opts = mock_ydl_cls.call_args.args[0]
    assert "height<=360" in opts["format"]
    assert opts["merge_output_format"] == "mp4"
    assert opts["noplaylist"] is True


@patch("yt_dlp.YoutubeDL")
def test_download_video_url_raises_when_file_missing(mock_ydl_cls, tmp_path):
    ydl = MagicMock()
    ydl.extract_info.return_value = {"id": "abc123", "title": "Remote job video"}
    ydl.prepare_filename.return_value = str(tmp_path / "missing.webm")
    mock_ydl_cls.return_value.__enter__.return_value = ydl

    with pytest.raises(VideoDownloadError, match="no downloaded MP4"):
        download_video_url("https://youtu.be/example", output_dir=tmp_path)
