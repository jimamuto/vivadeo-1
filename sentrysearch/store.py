"""ChromaDB vector store."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import chromadb


DEFAULT_DB_PATH = Path.home() / ".sentrysearch" / "db"
COLLECTION_NAME = "video_chunks_modal_qwen3_vl_2b"


def _make_chunk_id(source_file: str, start_time: float) -> str:
    """Deterministic chunk ID from source file + start time."""
    raw = f"{source_file}:{start_time}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class VideoStore:
    """Persistent vector store backed by local ChromaDB."""

    def __init__(self, db_path: str | Path | None = None):
        db_path = str(db_path or DEFAULT_DB_PATH)
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "embedding_backend": "modal",
                "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
            },
        )

    @property
    def collection(self) -> chromadb.Collection:
        return self._collection

    def add_chunk(
        self,
        chunk_id: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        """Store a single chunk embedding with metadata."""
        meta = {
            "source_file": metadata["source_file"],
            "start_time": float(metadata["start_time"]),
            "end_time": float(metadata["end_time"]),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        for key in metadata:
            if key not in meta and key != "embedding":
                meta[key] = metadata[key]

        self._collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            metadatas=[meta],
        )

    def add_chunks(self, chunks: list[dict]) -> None:
        """Batch-store chunks. Each dict must have 'embedding' and metadata keys."""
        now = datetime.now(timezone.utc).isoformat()
        ids = []
        embeddings = []
        metadatas = []
        for chunk in chunks:
            ids.append(_make_chunk_id(chunk["source_file"], chunk["start_time"]))
            embeddings.append(chunk["embedding"])
            metadatas.append({
                "source_file": chunk["source_file"],
                "start_time": float(chunk["start_time"]),
                "end_time": float(chunk["end_time"]),
                "indexed_at": now,
            })

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(self, query_embedding: list[float], n_results: int = 5) -> list[dict]:
        """Return top N results with distances and metadata."""
        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
        )

        hits = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            hits.append({
                "source_file": meta["source_file"],
                "start_time": meta["start_time"],
                "end_time": meta["end_time"],
                "score": 1.0 - distance,
                "distance": distance,
            })
        return hits

    def is_indexed(self, source_file: str) -> bool:
        results = self._collection.get(where={"source_file": source_file}, limit=1)
        return len(results["ids"]) > 0

    def has_chunk(self, chunk_id: str) -> bool:
        results = self._collection.get(ids=[chunk_id], limit=1)
        return len(results["ids"]) > 0

    def make_chunk_id(self, source_file: str, start_time: float) -> str:
        return _make_chunk_id(source_file, start_time)

    def remove_file(self, source_file: str) -> int:
        results = self._collection.get(where={"source_file": source_file})
        ids = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def get_stats(self) -> dict:
        total = self._collection.count()
        if total == 0:
            return {"total_chunks": 0, "unique_source_files": 0, "source_files": []}

        all_meta = self._collection.get(include=["metadatas"])
        source_files = sorted({m["source_file"] for m in all_meta["metadatas"]})
        return {
            "total_chunks": total,
            "unique_source_files": len(source_files),
            "source_files": source_files,
        }


SentryStore = VideoStore
