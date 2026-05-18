"""Abstract base class for video/text embedding backends."""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_video_chunk(self, chunk_path: str, verbose: bool = False) -> list[float]:
        ...

    def embed_video_chunks(
        self,
        chunk_paths: list[str],
        verbose: bool = False,
    ) -> list[list[float]]:
        return [
            self.embed_video_chunk(chunk_path, verbose=verbose)
            for chunk_path in chunk_paths
        ]

    @abstractmethod
    def embed_query(self, query_text: str, verbose: bool = False) -> list[float]:
        ...

    @abstractmethod
    def embed_image(self, image_path: str, verbose: bool = False) -> list[float]:
        ...

    @abstractmethod
    def dimensions(self) -> int:
        ...
