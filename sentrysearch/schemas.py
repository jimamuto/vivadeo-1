"""Pydantic API schemas."""

from pydantic import BaseModel, Field


class JobResponse(BaseModel):
    id: str
    kind: str
    status: str
    progress: float
    message: str | None = None
    error: str | None = None
    video_id: str | None = None
    clip_id: str | None = None


class VideoResponse(BaseModel):
    id: str
    source_type: str
    source_uri: str
    filename: str
    status: str
    duration: float | None = None
    object_key: str | None = None
    url: str | None = None


class UrlIngestRequest(BaseModel):
    url: str
    max_height: int = 480


class LocalPathIngestRequest(BaseModel):
    path: str


class SearchRequest(BaseModel):
    query: str
    results: int = Field(5, ge=1, le=100)
    threshold: float | None = None
    video_id: str | None = None


class SearchResult(BaseModel):
    chunk_id: str
    video_id: str
    filename: str
    source_uri: str
    start_time: float
    end_time: float
    similarity_score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class ClipRequest(BaseModel):
    video_id: str
    start_time: float
    end_time: float


class ClipResponse(BaseModel):
    id: str
    video_id: str
    status: str
    start_time: float
    end_time: float
    object_key: str | None = None
    url: str | None = None
    job_id: str | None = None

