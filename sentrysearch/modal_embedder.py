"""Client for deployed Modal Qwen3-VL-Embedding functions."""

import time
from pathlib import Path

import modal

from .base_embedder import BaseEmbedder


APP_NAME = "sentrysearch-qwen3-vl-embedding-2b"
CLS_NAME = "QwenEmbedder"
DIMENSIONS = 768


class ModalEmbedderError(RuntimeError):
    """Raised when the Modal deployment is unavailable or returns bad data."""


class ModalEmbedder(BaseEmbedder):
    """Modal Python SDK client for deployed remote methods."""

    def __init__(
        self,
        app_name: str = APP_NAME,
        cls_name: str = CLS_NAME,
        timeout: int = 900,
    ):
        self._app_name = app_name
        self._cls_name = cls_name
        self._timeout = timeout
        self._remote = None

    def _get_remote(self):
        if self._remote is not None:
            return self._remote

        try:
            cls = modal.Cls.from_name(self._app_name, self._cls_name)
            self._remote = cls()
            return self._remote
        except Exception as exc:
            raise ModalEmbedderError(
                "Could not connect to the deployed Modal embedder.\n\n"
                "Deploy it first:\n"
                "  modal deploy sentrysearch/modal_app.py\n\n"
                "Make sure your Modal credentials are configured with "
                "`modal setup` or MODAL_TOKEN_ID/MODAL_TOKEN_SECRET."
            ) from exc

    @staticmethod
    def _read_file(path: str) -> tuple[bytes, str]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return file_path.read_bytes(), file_path.name

    @staticmethod
    def _validate(embedding: list[float]) -> list[float]:
        if len(embedding) != DIMENSIONS:
            raise ModalEmbedderError(
                f"Expected {DIMENSIONS} dimensions, got {len(embedding)}."
            )
        return embedding

    @classmethod
    def _validate_many(cls, embeddings: list[list[float]], expected: int) -> list[list[float]]:
        if len(embeddings) != expected:
            raise ModalEmbedderError(
                f"Expected {expected} embeddings, got {len(embeddings)}."
            )
        return [cls._validate(embedding) for embedding in embeddings]

    def _call(self, method_name: str, *args, verbose: bool = False) -> list[float]:
        remote = self._get_remote()
        method = getattr(remote, method_name)
        t0 = time.monotonic()
        try:
            embedding = method.remote(*args)
        except Exception as exc:
            raise ModalEmbedderError(
                f"Modal remote method {method_name} failed: {exc}"
            ) from exc

        if verbose:
            elapsed = time.monotonic() - t0
            print(
                f"  [verbose] modal remote {method_name}: "
                f"dims={len(embedding)}, time={elapsed:.2f}s"
            )
        return self._validate(embedding)

    def embed_video_chunk(self, chunk_path: str, verbose: bool = False) -> list[float]:
        data, filename = self._read_file(chunk_path)
        if verbose:
            print(f"    [verbose] sending {len(data) / 1024:.0f}KB video chunk to Modal")
        return self._call("embed_video", data, filename, verbose=verbose)

    def embed_video_chunks(
        self,
        chunk_paths: list[str],
        verbose: bool = False,
    ) -> list[list[float]]:
        items = [self._read_file(chunk_path) for chunk_path in chunk_paths]
        if verbose:
            total_kb = sum(len(data) for data, _ in items) / 1024
            print(
                f"    [verbose] sending batch of {len(items)} video chunks "
                f"({total_kb:.0f}KB) to Modal"
            )

        remote = self._get_remote()
        t0 = time.monotonic()
        try:
            embeddings = remote.embed_videos.remote(items)
        except Exception as exc:
            raise ModalEmbedderError(f"Modal remote method embed_videos failed: {exc}") from exc

        if verbose:
            elapsed = time.monotonic() - t0
            print(
                f"    [verbose] modal remote embed_videos: "
                f"count={len(embeddings)}, time={elapsed:.2f}s"
            )
        return self._validate_many(embeddings, len(items))

    def embed_query(self, query_text: str, verbose: bool = False) -> list[float]:
        return self._call("embed_text", query_text, verbose=verbose)

    def embed_image(self, image_path: str, verbose: bool = False) -> list[float]:
        data, filename = self._read_file(image_path)
        if verbose:
            print(f"  [verbose] sending {len(data) / 1024:.0f}KB image to Modal")
        return self._call("embed_image", data, filename, verbose=verbose)

    def dimensions(self) -> int:
        return DIMENSIONS
