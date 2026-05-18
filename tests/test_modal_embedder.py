"""Tests for the Modal remote embedder client."""

from unittest.mock import MagicMock, patch

import pytest

from sentrysearch.modal_embedder import ModalEmbedder, ModalEmbedderError


def _remote_with_methods(vector):
    remote = MagicMock()
    remote.embed_text.remote.return_value = vector
    remote.embed_video.remote.return_value = vector
    remote.embed_videos.remote.return_value = [vector]
    remote.embed_image.remote.return_value = vector
    return remote


@patch("sentrysearch.modal_embedder.modal.Cls.from_name")
def test_embed_query_calls_deployed_class(mock_from_name):
    remote = _remote_with_methods([0.1] * 768)
    cls = MagicMock(return_value=remote)
    mock_from_name.return_value = cls

    embedder = ModalEmbedder()
    result = embedder.embed_query("red car")

    assert result == [0.1] * 768
    mock_from_name.assert_called_once_with(
        "sentrysearch-qwen3-vl-embedding-2b",
        "QwenEmbedder",
    )
    remote.embed_text.remote.assert_called_once_with("red car")


@patch("sentrysearch.modal_embedder.modal.Cls.from_name")
def test_embed_video_sends_bytes(mock_from_name, tmp_path):
    remote = _remote_with_methods([0.2] * 768)
    mock_from_name.return_value = MagicMock(return_value=remote)

    video = tmp_path / "chunk.mp4"
    video.write_bytes(b"video-bytes")

    embedder = ModalEmbedder()
    result = embedder.embed_video_chunk(str(video))

    assert result == [0.2] * 768
    remote.embed_video.remote.assert_called_once_with(b"video-bytes", "chunk.mp4")


@patch("sentrysearch.modal_embedder.modal.Cls.from_name")
def test_embed_video_chunks_sends_batch(mock_from_name, tmp_path):
    remote = _remote_with_methods([0.2] * 768)
    remote.embed_videos.remote.return_value = [[0.2] * 768, [0.3] * 768]
    mock_from_name.return_value = MagicMock(return_value=remote)

    first = tmp_path / "chunk_000.mp4"
    second = tmp_path / "chunk_001.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    embedder = ModalEmbedder()
    result = embedder.embed_video_chunks([str(first), str(second)])

    assert result == [[0.2] * 768, [0.3] * 768]
    remote.embed_videos.remote.assert_called_once_with([
        (b"first", "chunk_000.mp4"),
        (b"second", "chunk_001.mp4"),
    ])


@patch("sentrysearch.modal_embedder.modal.Cls.from_name")
def test_bad_dimension_raises(mock_from_name):
    remote = _remote_with_methods([0.1] * 12)
    mock_from_name.return_value = MagicMock(return_value=remote)

    embedder = ModalEmbedder()
    with pytest.raises(ModalEmbedderError, match="Expected 768"):
        embedder.embed_query("red car")


@patch("sentrysearch.modal_embedder.modal.Cls.from_name")
def test_lookup_failure_has_deploy_message(mock_from_name):
    mock_from_name.side_effect = RuntimeError("not found")

    embedder = ModalEmbedder()
    with pytest.raises(ModalEmbedderError, match="modal deploy"):
        embedder.embed_query("red car")
