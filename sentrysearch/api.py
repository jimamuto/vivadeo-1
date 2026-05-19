"""FastAPI production API."""

import os
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .config import Settings, get_settings
from .db import Base, Clip, Job, SessionLocal, Video, make_engine, new_id
from .embedder import get_embedder, reset_embedder
from .object_store import ObjectStore, video_object_key
from .production_store import PostgresVideoStore
from .schemas import (
    ClipRequest,
    ClipResponse,
    JobResponse,
    LocalPathIngestRequest,
    SearchRequest,
    SearchResponse,
    UrlIngestRequest,
    VideoResponse,
)
from .worker import ingest_local_path, ingest_uploaded_object, ingest_url, trim_clip_task


app = FastAPI(title="SentrySearch", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    engine = make_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
    ObjectStore().ensure_bucket()


def settings_dep() -> Settings:
    return get_settings()


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(settings_dep),
) -> None:
    if not settings.api_key or x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


def db_dep():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _job_response(job: Job) -> JobResponse:
    return JobResponse(
        id=job.id,
        kind=job.kind,
        status=job.status,
        progress=job.progress,
        message=job.message,
        error=job.error,
        video_id=job.video_id,
        clip_id=job.clip_id,
    )


def _video_response(video: Video, store: ObjectStore | None = None) -> VideoResponse:
    url = store.presigned_url(video.object_key) if store and video.object_key else None
    return VideoResponse(
        id=video.id,
        source_type=video.source_type,
        source_uri=video.source_uri,
        filename=video.filename,
        status=video.status,
        duration=video.duration,
        object_key=video.object_key,
        url=url,
    )


def _clip_response(clip: Clip, store: ObjectStore | None = None) -> ClipResponse:
    url = store.presigned_url(clip.object_key) if store and clip.object_key else None
    return ClipResponse(
        id=clip.id,
        video_id=clip.video_id,
        status=clip.status,
        start_time=clip.start_time,
        end_time=clip.end_time,
        object_key=clip.object_key,
        url=url,
        job_id=clip.job_id,
    )


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/v1/videos/upload", response_model=JobResponse, dependencies=[Depends(require_api_key)])
async def upload_video(file: UploadFile = File(...), session: Session = Depends(db_dep)):
    video_id = new_id()
    job_id = new_id()
    filename = Path(file.filename or f"{video_id}.mp4").name
    object_key = video_object_key(video_id, filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
        tmp_path = tmp.name
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)

    try:
        ObjectStore().upload_file(tmp_path, object_key, file.content_type)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    video = Video(
        id=video_id,
        source_type="upload",
        source_uri=filename,
        object_key=object_key,
        filename=filename,
        content_type=file.content_type,
        status="queued",
    )
    session.add(video)
    session.flush()
    job = Job(id=job_id, kind="ingest_uploaded_object", status="queued", video_id=video_id)
    session.add(job)
    session.commit()
    ingest_uploaded_object.delay(job_id, video_id)
    return _job_response(job)


@app.post("/v1/videos/url", response_model=JobResponse, dependencies=[Depends(require_api_key)])
def ingest_video_url(request: UrlIngestRequest, session: Session = Depends(db_dep)):
    video_id = new_id()
    job_id = new_id()
    video = Video(
        id=video_id,
        source_type="url",
        source_uri=request.url,
        filename=Path(request.url).name or "download.mp4",
        status="queued",
    )
    session.add(video)
    session.flush()
    job = Job(
        id=job_id,
        kind="ingest_url",
        status="queued",
        video_id=video_id,
        payload={"url": request.url, "max_height": request.max_height},
    )
    session.add(job)
    session.commit()
    ingest_url.delay(job_id, video_id, request.url, request.max_height)
    return _job_response(job)


@app.post("/v1/videos/local-path", response_model=JobResponse, dependencies=[Depends(require_api_key)])
def ingest_local_video(request: LocalPathIngestRequest, session: Session = Depends(db_dep)):
    path = Path(request.path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Local video path not found")
    video_id = new_id()
    job_id = new_id()
    video = Video(
        id=video_id,
        source_type="local_path",
        source_uri=str(path),
        filename=path.name,
        status="queued",
    )
    session.add(video)
    session.flush()
    job = Job(id=job_id, kind="ingest_local_path", status="queued", video_id=video_id)
    session.add(job)
    session.commit()
    ingest_local_path.delay(job_id, video_id, str(path))
    return _job_response(job)


@app.get("/v1/videos", response_model=list[VideoResponse], dependencies=[Depends(require_api_key)])
def list_videos(session: Session = Depends(db_dep)):
    store = ObjectStore()
    videos = session.scalars(select(Video).order_by(Video.created_at.desc())).all()
    return [_video_response(video, store) for video in videos]


@app.get("/v1/videos/{video_id}", response_model=VideoResponse, dependencies=[Depends(require_api_key)])
def get_video(video_id: str, session: Session = Depends(db_dep)):
    video = session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return _video_response(video, ObjectStore())


@app.get("/v1/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require_api_key)])
def get_job(job_id: str, session: Session = Depends(db_dep)):
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_response(job)


@app.post("/v1/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
def search(request: SearchRequest, session: Session = Depends(db_dep)):
    try:
        embedding = get_embedder().embed_query(request.query)
        results = PostgresVideoStore(session).search(
            embedding,
            n_results=request.results,
            video_id=request.video_id,
        )
    finally:
        reset_embedder()
    if request.threshold is not None:
        results = [r for r in results if r["similarity_score"] >= request.threshold]
    return SearchResponse(results=results)


@app.post("/v1/clips", response_model=ClipResponse, dependencies=[Depends(require_api_key)])
def create_clip(request: ClipRequest, session: Session = Depends(db_dep)):
    if request.end_time <= request.start_time:
        raise HTTPException(status_code=400, detail="end_time must be greater than start_time")
    video = session.get(Video, request.video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    clip = Clip(
        id=new_id(),
        video_id=request.video_id,
        start_time=request.start_time,
        end_time=request.end_time,
        status="queued",
    )
    job = Job(id=new_id(), kind="trim_clip", status="queued", video_id=request.video_id, clip_id=clip.id)
    clip.job_id = job.id
    session.add_all([clip, job])
    session.commit()
    trim_clip_task.delay(job.id, clip.id)
    return _clip_response(clip)


@app.get("/v1/clips/{clip_id}", response_model=ClipResponse, dependencies=[Depends(require_api_key)])
def get_clip(clip_id: str, session: Session = Depends(db_dep)):
    clip = session.get(Clip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    return _clip_response(clip, ObjectStore())


@app.get("/v1/stats", dependencies=[Depends(require_api_key)])
def stats(session: Session = Depends(db_dep)) -> dict:
    try:
        return PostgresVideoStore(session).stats()
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
