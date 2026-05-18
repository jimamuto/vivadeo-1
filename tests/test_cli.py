"""Tests for sentrysearch.cli."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from sentrysearch.cli import _fmt_time, cli


@pytest.fixture
def runner():
    return CliRunner()


class TestFmtTime:
    def test_zero(self):
        assert _fmt_time(0) == "00:00"

    def test_minutes(self):
        assert _fmt_time(125) == "02:05"


class TestCliGroup:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "video" in result.output.lower()


class TestStatsCommand:
    def test_stats_empty(self, runner):
        with patch("sentrysearch.store.VideoStore") as MockStore:
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 0,
                "unique_source_files": 0,
                "source_files": [],
            }
            MockStore.return_value = inst
            result = runner.invoke(cli, ["stats"])
            assert result.exit_code == 0
            assert "empty" in result.output.lower()

    def test_stats_with_data(self, runner):
        with patch("sentrysearch.store.VideoStore") as MockStore:
            inst = MagicMock()
            inst.get_stats.return_value = {
                "total_chunks": 10,
                "unique_source_files": 2,
                "source_files": ["/a/video1.mp4", "/b/video2.mp4"],
            }
            MockStore.return_value = inst
            result = runner.invoke(cli, ["stats"])
            assert result.exit_code == 0
            assert "10" in result.output
            assert "Qwen/Qwen3-VL-Embedding-2B" in result.output


class TestSearchCommand:
    def test_search_empty_index(self, runner):
        with patch("sentrysearch.store.VideoStore") as MockStore:
            inst = MagicMock()
            inst.get_stats.return_value = {"total_chunks": 0}
            MockStore.return_value = inst
            result = runner.invoke(cli, ["search", "red car"])
            assert result.exit_code == 0
            assert "No indexed footage" in result.output


class TestIndexCommand:
    def test_index_no_supported_videos(self, runner, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.VideoStore") as MockStore:
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, ["index", str(empty_dir)])
            assert result.exit_code == 0
            assert "No supported video files found" in result.output

    def test_index_scans_mov_files(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "iphone.MOV"
        source.write_bytes(b"fake")

        chunk_dir = tmp_path / "chunks"
        chunk_dir.mkdir()
        chunk_path = chunk_dir / "chunk_000.mp4"
        chunk_path.write_bytes(b"chunk")

        mock_store = MagicMock()
        mock_store.has_chunk.return_value = False
        mock_store.make_chunk_id.return_value = "abc123"
        mock_store.get_stats.return_value = {
            "total_chunks": 1,
            "unique_source_files": 1,
        }
        mock_embedder = MagicMock()
        mock_embedder.embed_video_chunks.return_value = [[0.1] * 768]

        with patch("sentrysearch.store.VideoStore", return_value=mock_store), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder), \
             patch("sentrysearch.chunker.chunk_video", return_value=[{
                 "chunk_path": str(chunk_path),
                 "source_file": str(source.resolve()),
                 "start_time": 0.0,
                 "end_time": 1.0,
             }]), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(cli, ["index", str(d), "--no-preprocess"])

        assert result.exit_code == 0
        mock_store.add_chunk.assert_called_once()
        mock_embedder.embed_video_chunks.assert_called_once_with(
            [str(chunk_path)],
            verbose=False,
        )

    def test_index_records_failed_chunk_to_dlq(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        source = d / "video.mp4"
        source.write_bytes(b"fake")

        chunk_dir = tmp_path / "chunks"
        chunk_dir.mkdir()
        chunk_path = chunk_dir / "chunk_000.mp4"
        chunk_path.write_bytes(b"chunk")

        mock_store = MagicMock()
        mock_store.has_chunk.return_value = False
        mock_store.make_chunk_id.return_value = "failing_id"
        mock_store.get_stats.return_value = {
            "total_chunks": 0,
            "unique_source_files": 0,
        }
        mock_embedder = MagicMock()
        mock_embedder.embed_video_chunks.side_effect = RuntimeError("CUDA out of memory")

        from sentrysearch.dlq import DeadLetterQueue

        dlq_instance = DeadLetterQueue(tmp_path / "dlq.json")

        with patch("sentrysearch.store.VideoStore", return_value=mock_store), \
             patch("sentrysearch.embedder.get_embedder", return_value=mock_embedder), \
             patch("sentrysearch.dlq.DeadLetterQueue", return_value=dlq_instance), \
             patch("sentrysearch.chunker.chunk_video", return_value=[{
                 "chunk_path": str(chunk_path),
                 "source_file": str(source.resolve()),
                 "start_time": 0.0,
                 "end_time": 30.0,
             }]), \
             patch("sentrysearch.chunker.is_still_frame_chunk", return_value=False):
            result = runner.invoke(cli, ["index", str(d), "--no-preprocess"])

        assert result.exit_code == 0
        mock_store.add_chunk.assert_not_called()
        assert dlq_instance.contains("failing_id")

    def test_index_overlap_equal_chunk_duration_errors(self, runner, tmp_path):
        d = tmp_path / "vids"
        d.mkdir()
        result = runner.invoke(cli, [
            "index", str(d), "--chunk-duration", "5", "--overlap", "5",
        ])
        assert result.exit_code != 0
        assert "overlap" in result.output


class TestDownloadUrlCommand:
    def test_download_url_prints_path(self, runner, tmp_path):
        out = tmp_path / "video.mp4"
        with patch("sentrysearch.downloader.download_video_url", return_value=str(out)) as mock_download:
            result = runner.invoke(cli, [
                "download-url",
                "https://youtu.be/example",
                "--output-dir",
                str(tmp_path),
                "--max-height",
                "360",
            ])

        assert result.exit_code == 0
        assert str(out) in result.output
        mock_download.assert_called_once_with(
            "https://youtu.be/example",
            output_dir=str(tmp_path),
            max_height=360,
            verbose=False,
        )

    def test_download_url_can_index_after_download(self, runner, tmp_path):
        out = tmp_path / "video.mp4"
        out.write_bytes(b"video")
        with patch("sentrysearch.downloader.download_video_url", return_value=str(out)), \
             patch("sentrysearch.embedder.get_embedder", return_value=MagicMock()), \
             patch("sentrysearch.store.VideoStore") as MockStore, \
             patch("sentrysearch.chunker.chunk_video", return_value=[]):
            MockStore.return_value = MagicMock()
            result = runner.invoke(cli, [
                "download-url",
                "https://youtu.be/example",
                "--output-dir",
                str(tmp_path),
                "--index",
            ])

        assert result.exit_code == 0
        assert "Downloaded:" in result.output
