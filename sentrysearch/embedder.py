"""Embedder factory for Modal-hosted Qwen3-VL-Embedding."""

from .base_embedder import BaseEmbedder

_current_embedder: BaseEmbedder | None = None


def get_embedder(backend: str = "modal", **kwargs) -> BaseEmbedder:
    """Factory to get or create the active embedder."""
    global _current_embedder
    if backend != "modal":
        raise ValueError(f"Unknown backend: {backend}")
    if _current_embedder is None:
        from .modal_embedder import ModalEmbedder

        _current_embedder = ModalEmbedder(
            app_name=kwargs.get("app_name", "sentrysearch-qwen3-vl-embedding-2b"),
            cls_name=kwargs.get("cls_name", "QwenEmbedder"),
            timeout=kwargs.get("timeout", 600),
        )
    return _current_embedder


def reset_embedder():
    """Reset the cached embedder."""
    global _current_embedder
    _current_embedder = None


def embed_video_chunk(chunk_path: str, verbose: bool = False) -> list[float]:
    return get_embedder().embed_video_chunk(chunk_path, verbose=verbose)


def embed_query(query_text: str, verbose: bool = False) -> list[float]:
    return get_embedder().embed_query(query_text, verbose=verbose)


def embed_image(image_path: str, verbose: bool = False) -> list[float]:
    return get_embedder().embed_image(image_path, verbose=verbose)
