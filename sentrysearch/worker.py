"""Celery worker tasks for production ingestion and clip generation."""

import os
import shutil
import tempfile
from pathlib import Path

from celery import Celery
from sqlalchemy import select

from .chunker import chunk_video, is_still_frame_chunk, preprocess_chunk, _get_video_duration
from .config import get_settings
from .db import Clip, DeadLetterEntry, Job, Video, new_id, session_scope
from .downloader import download_video_url
from .embedder import get_embedder, reset_embedder
from .object_store import ObjectStore, clip_object_key, video_object_key
from .production_store import PostgresVideoStore
from .trimmer import trim_clip


settings = get_settings()
celery_app = Celery(
    "sentrysearch",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(task_track_started=True, worker_prefetch_multiplier=1)


def _update_job(job_id: str, **values) -> None:
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job:
            for key, value in values.items():
                setattr(job, key, value)


def _mark_video(video_id: str, **values) -> None:
    with session_scope() as session:
        video = session.get(Video, video_id)
        if video:
            for key, value in values.items():
                setattr(video, key, value)


def _record_dlq(video_id: str, chunk_id: str, source_uri: str, start: float, end: float, error: str) -> None:
    with session_scope() as session:
        session.add(
            DeadLetterEntry(
                video_id=video_id,
                chunk_id=chunk_id,
                source_uri=source_uri,
                start_time=start,
                end_time=end,
                error=error[:2000],
                attempts=1,
            )
        )


def _index_file(video_id: str, file_path: str, job_id: str) -> None:
    settings = get_settings()
    embedder = get_embedder(
        app_name=settings.modal_app,
        cls_name=settings.modal_class,
        timeout=settings.modal_timeout,
    )
    chunks = chunk_video(
        file_path,
        chunk_duration=settings.chunk_duration,
        overlap=settings.chunk_overlap,
    )
    files_to_cleanup: list[str] = []
    try:
        total = len(chunks) or 1
        batch: list[dict] = []
        stored_count = 0
        failed_count = 0

        def flush_batch() -> int:
            nonlocal stored_count
            if not batch:
                return 0
            embeddings = embedder.embed_video_chunks(
                [item["embed_path"] for item in batch],
                verbose=False,
            )
            with session_scope() as session:
                store = PostgresVideoStore(session)
                for item, embedding in zip(batch, embeddings):
                    store.add_chunk(
                        video_id=video_id,
                        start_time=item["start_time"],
                        end_time=item["end_time"],
                        embedding=embedding,
                        metadata={"source_file": file_path},
                    )
            stored = len(batch)
            stored_count += stored
            batch.clear()
            return stored

        processed = 0
        for chunk in chunks:
            chunk_path = chunk["chunk_path"]
            files_to_cleanup.append(chunk_path)
            processed += 1
            _update_job(
                job_id,
                status="running",
                progress=min(0.95, processed / total),
                message=f"Embedding chunk {processed}/{len(chunks)}",
            )

            if settings.skip_still and is_still_frame_chunk(chunk_path):
                continue

            embed_path = chunk_path
            if settings.preprocess:
                embed_path = preprocess_chunk(
                    chunk_path,
                    target_resolution=settings.target_resolution,
                    target_fps=settings.target_fps,
                )
                if embed_path != chunk_path:
                    files_to_cleanup.append(embed_path)

            batch.append(
                {
                    "chunk_id": f"{video_id}:{chunk['start_time']}",
                    "embed_path": embed_path,
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                }
            )
            if len(batch) >= settings.batch_size:
                try:
                    flush_batch()
                except Exception as exc:
                    failed_count += len(batch)
                    for item in batch:
                        _record_dlq(
                            video_id,
                            item["chunk_id"],
                            file_path,
                            item["start_time"],
                            item["end_time"],
                            repr(exc),
                        )
                    batch.clear()
        if batch:
            try:
                flush_batch()
            except Exception as exc:
                failed_count += len(batch)
                for item in batch:
                    _record_dlq(
                        video_id,
                        item["chunk_id"],
                        file_path,
                        item["start_time"],
                        item["end_time"],
                        repr(exc),
                    )
                batch.clear()
        if stored_count == 0 and failed_count > 0:
            raise RuntimeError(f"All {failed_count} chunk embedding attempt(s) failed.")
    finally:
        reset_embedder()
        for path in files_to_cleanup:
            try:
                os.unlink(path)
            except OSError:
                pass
        if chunks:
            shutil.rmtree(os.path.dirname(chunks[0]["chunk_path"]), ignore_errors=True)


@celery_app.task(name="sentrysearch.ingest_local_path")
def ingest_local_path(job_id: str, video_id: str, path: str) -> None:
    try:
        _update_job(job_id, status="running", progress=0.02, message="Uploading original")
        store = ObjectStore()
        with session_scope() as session:
            video = session.get(Video, video_id)
            if video is None:
                raise RuntimeError(f"Video not found: {video_id}")
            object_key = video_object_key(video_id, video.filename)
            store.upload_file(path, object_key, video.content_type)
            video.object_key = object_key
            video.duration = _get_video_duration(path)
            video.status = "indexing"

        _index_file(video_id, path, job_id)
        _mark_video(video_id, status="ready", error=None)
        _update_job(job_id, status="succeeded", progress=1.0, message="Indexed")
    except Exception as exc:
        _mark_video(video_id, status="failed", error=str(exc))
        _update_job(job_id, status="failed", error=str(exc), message="Failed")
        raise


@celery_app.task(name="sentrysearch.ingest_uploaded_object")
def ingest_uploaded_object(job_id: str, video_id: str) -> None:
    tmp_dir = tempfile.mkdtemp(prefix="sentrysearch_upload_")
    try:
        store = ObjectStore()
        with session_scope() as session:
            video = session.get(Video, video_id)
            if video is None or not video.object_key:
                raise RuntimeError(f"Video object not found: {video_id}")
            local_path = os.path.join(tmp_dir, video.filename)
            object_key = video.object_key
        _update_job(job_id, status="running", progress=0.05, message="Downloading original")
        store.download_file(object_key, local_path)
        _mark_video(video_id, status="indexing", duration=_get_video_duration(local_path))
        _index_file(video_id, local_path, job_id)
        _mark_video(video_id, status="ready", error=None)
        _update_job(job_id, status="succeeded", progress=1.0, message="Indexed")
    except Exception as exc:
        _mark_video(video_id, status="failed", error=str(exc))
        _update_job(job_id, status="failed", error=str(exc), message="Failed")
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@celery_app.task(name="sentrysearch.ingest_url")
def ingest_url(job_id: str, video_id: str, url: str, max_height: int = 480) -> None:
    tmp_dir = tempfile.mkdtemp(prefix="sentrysearch_url_")
    try:
        _update_job(job_id, status="running", progress=0.02, message="Downloading URL")
        path = download_video_url(url, output_dir=tmp_dir, max_height=max_height)
        filename = Path(path).name
        store = ObjectStore()
        object_key = video_object_key(video_id, filename)
        store.upload_file(path, object_key, "video/mp4")
        with session_scope() as session:
            video = session.get(Video, video_id)
            if video is None:
                raise RuntimeError(f"Video not found: {video_id}")
            video.filename = filename
            video.object_key = object_key
            video.duration = _get_video_duration(path)
            video.status = "indexing"
        _index_file(video_id, path, job_id)
        _mark_video(video_id, status="ready", error=None)
        _update_job(job_id, status="succeeded", progress=1.0, message="Indexed")
    except Exception as exc:
        _mark_video(video_id, status="failed", error=str(exc))
        _update_job(job_id, status="failed", error=str(exc), message="Failed")
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@celery_app.task(name="sentrysearch.trim_clip")
def trim_clip_task(job_id: str, clip_id: str) -> None:
    tmp_dir = tempfile.mkdtemp(prefix="sentrysearch_clip_")
    try:
        store = ObjectStore()
        with session_scope() as session:
            clip = session.get(Clip, clip_id)
            if clip is None:
                raise RuntimeError(f"Clip not found: {clip_id}")
            video = session.get(Video, clip.video_id)
            if video is None or not video.object_key:
                raise RuntimeError(f"Video object not found for clip: {clip_id}")
            local_video = os.path.join(tmp_dir, video.filename)
            local_clip = os.path.join(tmp_dir, f"{clip.id}.mp4")
            object_key = video.object_key
            start_time = clip.start_time
            end_time = clip.end_time

        _update_job(job_id, status="running", progress=0.2, message="Downloading source")
        store.download_file(object_key, local_video)
        trim_clip(local_video, start_time, end_time, local_clip)
        clip_key = clip_object_key(clip_id)
        store.upload_file(local_clip, clip_key, "video/mp4")
        with session_scope() as session:
            clip = session.get(Clip, clip_id)
            job = session.get(Job, job_id)
            if clip:
                clip.object_key = clip_key
                clip.status = "ready"
            if job:
                job.status = "succeeded"
                job.progress = 1.0
                job.message = "Clip ready"
    except Exception as exc:
        with session_scope() as session:
            clip = session.get(Clip, clip_id)
            job = session.get(Job, job_id)
            if clip:
                clip.status = "failed"
                clip.error = str(exc)
            if job:
                job.status = "failed"
                job.error = str(exc)
                job.message = "Failed"
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
