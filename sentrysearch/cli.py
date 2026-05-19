"""Click-based CLI entry point."""

import json
import mimetypes
import os
import platform
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request

import click


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _open_file(path: str) -> None:
    """Open a file with the system's default application."""
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", path])
        elif system == "Windows":
            os.startfile(path)
        else:
            subprocess.Popen(
                ["xdg-open", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass


def _api_configured() -> bool:
    return bool(os.environ.get("SENTRYSEARCH_API_URL"))


def _api_request(method: str, path: str, payload: dict | None = None) -> dict | list:
    """Call the production API when SENTRYSEARCH_API_URL is configured."""
    base_url = os.environ.get("SENTRYSEARCH_API_URL")
    api_key = os.environ.get("SENTRYSEARCH_API_KEY")
    if not base_url:
        raise click.ClickException("SENTRYSEARCH_API_URL is not configured.")
    if not api_key:
        raise click.ClickException("SENTRYSEARCH_API_KEY is required for API mode.")

    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()
        raise click.ClickException(f"API request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise click.ClickException(f"Could not reach API: {exc}") from exc
    return json.loads(body) if body else {}


def _api_upload(path: str) -> dict:
    base_url = os.environ.get("SENTRYSEARCH_API_URL")
    api_key = os.environ.get("SENTRYSEARCH_API_KEY")
    if not base_url:
        raise click.ClickException("SENTRYSEARCH_API_URL is not configured.")
    if not api_key:
        raise click.ClickException("SENTRYSEARCH_API_KEY is required for API mode.")

    filename = os.path.basename(path)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    boundary = "----sentrysearchupload"
    with open(path, "rb") as f:
        file_bytes = f.read()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            (
                'Content-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\n'
            ).encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            file_bytes,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", "/v1/videos/upload".lstrip("/"))
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "X-API-Key": api_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()
        raise click.ClickException(f"API upload failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise click.ClickException(f"Could not reach API: {exc}") from exc


def _embed_batch_once(
    embedder,
    batch: list[dict],
    dlq,
    *,
    verbose: bool = False,
) -> list[list[float]] | None:
    """Embed a batch once and route all batch failures to the DLQ."""
    try:
        return embedder.embed_video_chunks(
            [item["embed_path"] for item in batch],
            verbose=verbose,
        )
    except Exception as exc:
        for item in batch:
            chunk = item["chunk"]
            dlq.record(
                item["chunk_id"],
                source_file=chunk["source_file"],
                start_time=chunk["start_time"],
                end_time=chunk["end_time"],
                error=repr(exc),
                attempts=1,
            )
        click.secho(
            f"  Batch failed after 1 attempt, recorded {len(batch)} chunk(s) to DLQ: {exc}",
            fg="yellow",
            err=True,
        )
        return None


def _handle_error(e: Exception) -> None:
    """Print a user-friendly error and exit."""
    from .modal_embedder import ModalEmbedderError

    if isinstance(e, (ModalEmbedderError, PermissionError, FileNotFoundError)):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, RuntimeError) and "ffmpeg not found" in str(e).lower():
        click.secho(
            "Error: ffmpeg is not available.\n\n"
            "Install it with one of:\n"
            "  Ubuntu/Debian:  sudo apt install ffmpeg\n"
            "  macOS:          brew install ffmpeg\n"
            "  pip fallback:   uv add imageio-ffmpeg",
            fg="red",
            err=True,
        )
        raise SystemExit(1)
    raise e


@click.group()
def cli():
    """Search and research video footage using Modal-hosted embeddings."""


@cli.command("download-url")
@click.argument("url")
@click.option("-o", "--output-dir", default="~/sentrysearch_downloads", show_default=True,
              help="Directory to save the downloaded video.")
@click.option("--max-height", default=480, show_default=True,
              help="Maximum video height to download.")
@click.option("--index/--no-index", "index_after", default=False, show_default=True,
              help="Index the downloaded video after saving it.")
@click.option("--verbose", is_flag=True, help="Show yt-dlp output.")
def download_url(url, output_dir, max_height, index_after, verbose):
    """Download a lightweight MP4 from a video URL."""
    if _api_configured() and index_after:
        job = _api_request("POST", "/v1/videos/url", {"url": url, "max_height": max_height})
        click.echo(f"Queued URL ingest job: {job['id']} (video: {job.get('video_id')})")
        return

    from .downloader import VideoDownloadError, download_video_url

    try:
        path = download_video_url(
            url,
            output_dir=output_dir,
            max_height=max_height,
            verbose=verbose,
        )
        click.echo(f"Downloaded: {path}")
        if index_after:
            ctx = click.get_current_context()
            ctx.invoke(
                index,
                directory=path,
                chunk_duration=30,
                overlap=5,
                preprocess=True,
                target_resolution=480,
                target_fps=5,
                skip_still=True,
                retry_failed=False,
                batch_size=4,
                verbose=verbose,
            )
    except VideoDownloadError as e:
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--chunk-duration", default=30, show_default=True,
              help="Chunk duration in seconds.")
@click.option("--overlap", default=5, show_default=True,
              help="Overlap between chunks in seconds.")
@click.option("--preprocess/--no-preprocess", default=True, show_default=True,
              help="Downscale and reduce frame rate before embedding.")
@click.option("--target-resolution", default=480, show_default=True,
              help="Target video height in pixels for preprocessing.")
@click.option("--target-fps", default=5, show_default=True,
              help="Target frames per second for preprocessing.")
@click.option("--skip-still/--no-skip-still", default=True, show_default=True,
              help="Skip chunks with no meaningful visual change.")
@click.option("--retry-failed", is_flag=True,
              help="Retry chunks that previously failed and were routed to the DLQ.")
@click.option("--batch-size", default=4, show_default=True,
              help="Number of chunks per Modal embedding call.")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def index(directory, chunk_duration, overlap, preprocess, target_resolution,
          target_fps, skip_still, retry_failed, batch_size, verbose):
    """Index supported video files in DIRECTORY for searching."""
    if _api_configured():
        path = os.path.abspath(directory)
        if os.path.isfile(path):
            job = _api_upload(path)
            click.echo(f"Uploaded and queued ingest job: {job['id']} (video: {job.get('video_id')})")
        else:
            job = _api_request("POST", "/v1/videos/local-path", {"path": path})
            click.echo(f"Queued mounted-path ingest job: {job['id']} (video: {job.get('video_id')})")
        return

    from .chunker import (
        SUPPORTED_VIDEO_EXTENSIONS,
        _get_video_duration,
        chunk_video,
        expected_chunk_spans,
        is_still_frame_chunk,
        preprocess_chunk,
        scan_directory,
    )
    from .dlq import DeadLetterQueue
    from .embedder import get_embedder, reset_embedder
    from .store import VideoStore

    try:
        if overlap >= chunk_duration:
            raise click.BadParameter(
                f"overlap ({overlap}s) must be less than chunk_duration ({chunk_duration}s).",
                param_hint="'--overlap'",
            )
        if batch_size < 1:
            raise click.BadParameter(
                "batch_size must be at least 1.",
                param_hint="'--batch-size'",
            )

        embedder = get_embedder()
        videos = [os.path.abspath(directory)] if os.path.isfile(directory) else scan_directory(directory)

        if not videos:
            supported = ", ".join(SUPPORTED_VIDEO_EXTENSIONS)
            click.echo(f"No supported video files found ({supported}).")
            return

        store = VideoStore()
        dlq = DeadLetterQueue()
        total_files = len(videos)
        new_files = 0
        new_chunks = 0
        skipped_chunks = 0
        dlq_chunks = 0

        if verbose:
            click.echo(f"[verbose] DB path: {store._client._identifier}", err=True)
            click.echo(
                f"[verbose] embedder=modal/qwen3-vl-embedding-2b, "
                f"chunk_duration={chunk_duration}s, overlap={overlap}s, "
                f"batch_size={batch_size}",
                err=True,
            )

        for file_idx, video_path in enumerate(videos, 1):
            abs_path = os.path.abspath(video_path)
            basename = os.path.basename(video_path)

            try:
                duration = _get_video_duration(abs_path)
                expected_spans = expected_chunk_spans(
                    duration, chunk_duration=chunk_duration, overlap=overlap,
                )
                if expected_spans and all(
                    store.has_chunk(store.make_chunk_id(abs_path, s))
                    for s, _ in expected_spans
                ):
                    click.echo(
                        f"Skipping ({file_idx}/{total_files}): {basename} "
                        f"(already indexed)"
                    )
                    continue
            except Exception:
                pass

            chunks = chunk_video(abs_path, chunk_duration=chunk_duration, overlap=overlap)
            num_chunks = len(chunks)
            file_new_chunks = 0
            files_to_cleanup = []
            batch = []

            def flush_batch() -> int:
                if not batch:
                    return 0

                click.echo(f"Embedding batch of {len(batch)} chunk(s) on Modal")
                embeddings = _embed_batch_once(embedder, batch, dlq, verbose=verbose)
                if embeddings is None:
                    failed = len(batch)
                    batch.clear()
                    return -failed

                stored = 0
                for item, embedding in zip(batch, embeddings):
                    chunk = item["chunk"]
                    store.add_chunk(item["chunk_id"], embedding, {
                        "source_file": chunk["source_file"],
                        "start_time": chunk["start_time"],
                        "end_time": chunk["end_time"],
                    })
                    if retry_failed and dlq.contains(item["chunk_id"]):
                        dlq.remove(item["chunk_id"])
                    stored += 1
                batch.clear()
                return stored

            if verbose:
                click.echo(
                    f"  [verbose] {basename}: duration split into {num_chunks} chunks",
                    err=True,
                )

            for chunk_idx, chunk in enumerate(chunks, 1):
                chunk_id = store.make_chunk_id(abs_path, chunk["start_time"])

                if store.has_chunk(chunk_id):
                    files_to_cleanup.append(chunk["chunk_path"])
                    continue

                if dlq.contains(chunk_id):
                    if retry_failed:
                        dlq.remove(chunk_id)
                    else:
                        click.echo(
                            f"Skipping chunk {chunk_idx}/{num_chunks} (in DLQ; "
                            f"use --retry-failed to re-attempt)"
                        )
                        files_to_cleanup.append(chunk["chunk_path"])
                        continue

                if skip_still and is_still_frame_chunk(chunk["chunk_path"], verbose=verbose):
                    click.echo(f"Skipping chunk {chunk_idx}/{num_chunks} (still frame)")
                    skipped_chunks += 1
                    files_to_cleanup.append(chunk["chunk_path"])
                    continue

                click.echo(
                    f"Indexing file {file_idx}/{total_files}: {basename} "
                    f"[chunk {chunk_idx}/{num_chunks}]"
                )

                embed_path = chunk["chunk_path"]
                if preprocess:
                    original_size = os.path.getsize(embed_path)
                    embed_path = preprocess_chunk(
                        embed_path,
                        target_resolution=target_resolution,
                        target_fps=target_fps,
                    )
                    if verbose:
                        new_size = os.path.getsize(embed_path)
                        click.echo(
                            f"    [verbose] preprocess: {original_size / 1024:.0f}KB -> "
                            f"{new_size / 1024:.0f}KB",
                            err=True,
                        )
                    if embed_path != chunk["chunk_path"]:
                        files_to_cleanup.append(embed_path)

                batch.append({
                    "chunk_id": chunk_id,
                    "embed_path": embed_path,
                    "chunk": {
                        "source_file": abs_path,
                        "start_time": chunk["start_time"],
                        "end_time": chunk["end_time"],
                    },
                })
                files_to_cleanup.append(chunk["chunk_path"])

                if len(batch) >= batch_size:
                    flushed = flush_batch()
                    if flushed < 0:
                        dlq_chunks += -flushed
                    else:
                        file_new_chunks += flushed

            flushed = flush_batch()
            if flushed < 0:
                dlq_chunks += -flushed
            else:
                file_new_chunks += flushed

            for f in files_to_cleanup:
                try:
                    os.unlink(f)
                except OSError:
                    pass
            if chunks:
                shutil.rmtree(os.path.dirname(chunks[0]["chunk_path"]), ignore_errors=True)

            if file_new_chunks:
                new_files += 1
                new_chunks += file_new_chunks

        stats = store.get_stats()
        parts = []
        if skipped_chunks:
            parts.append(f"skipped {skipped_chunks} still")
        if dlq_chunks:
            parts.append(f"{dlq_chunks} failed -> DLQ")
        extra = f" ({', '.join(parts)})" if parts else ""
        click.echo(
            f"\nIndexed {new_chunks} new chunks from {new_files} files{extra}. "
            f"Total: {stats['total_chunks']} chunks from "
            f"{stats['unique_source_files']} files."
        )
    except Exception as e:
        _handle_error(e)
    finally:
        reset_embedder()


@cli.command()
@click.argument("query")
@click.option("-n", "--results", "n_results", default=5, show_default=True,
              help="Number of results to return.")
@click.option("-o", "--output-dir", default="~/sentrysearch_clips", show_default=True,
              help="Directory to save trimmed clips.")
@click.option("--trim/--no-trim", default=True, show_default=True,
              help="Auto-trim the top result.")
@click.option("--save-top", default=None, type=click.IntRange(min=1),
              help="Save the top N matching clips instead of just the #1 result.")
@click.option("--threshold", default=0.41, show_default=True, type=float,
              help="Minimum similarity score to consider a confident match.")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def search(query, n_results, output_dir, trim, save_top, threshold, verbose):
    """Search indexed footage with a natural language QUERY."""
    if _api_configured():
        response = _api_request(
            "POST",
            "/v1/search",
            {"query": query, "results": n_results, "threshold": threshold},
        )
        results = response.get("results", [])
        if not results:
            click.echo("No results found.")
            return
        for i, r in enumerate(results, 1):
            start_str = _fmt_time(r["start_time"])
            end_str = _fmt_time(r["end_time"])
            click.echo(
                f"  #{i} [{r['similarity_score']:.2f}] "
                f"{r['filename']} @ {start_str}-{end_str} "
                f"(video {r['video_id']})"
            )
        return

    from .embedder import get_embedder, reset_embedder
    from .search import search_footage
    from .store import VideoStore

    output_dir = os.path.expanduser(output_dir)

    try:
        store = VideoStore()
        if store.get_stats()["total_chunks"] == 0:
            click.echo("No indexed footage found. Run `sentrysearch index <directory>` first.")
            return

        get_embedder()
        if save_top is not None and save_top > n_results:
            n_results = save_top
        if verbose:
            click.echo("  [verbose] embedder=modal/qwen3-vl-embedding-2b", err=True)

        results = search_footage(query, store, n_results=n_results, verbose=verbose)
        _present_results(results, threshold, trim, save_top, output_dir, verbose)
    except Exception as e:
        _handle_error(e)
    finally:
        reset_embedder()


def _present_results(results, threshold, trim, save_top, output_dir, verbose):
    if not results:
        click.echo(
            "No results found.\n\n"
            "Suggestions:\n"
            "  - Try a broader or different query\n"
            "  - Re-index with smaller --chunk-duration for finer granularity\n"
            "  - Check `sentrysearch stats` to see what's indexed"
        )
        return

    best_score = results[0]["similarity_score"]
    low_confidence = best_score < threshold
    if low_confidence and not trim:
        click.secho(f"(low confidence: best score {best_score:.2f})", fg="yellow", err=True)

    for i, r in enumerate(results, 1):
        basename = os.path.basename(r["source_file"])
        start_str = _fmt_time(r["start_time"])
        end_str = _fmt_time(r["end_time"])
        score = r["similarity_score"]
        score_text = f"{score:.6f}" if verbose else f"{score:.2f}"
        click.echo(f"  #{i} [{score_text}] {basename} @ {start_str}-{end_str}")

    should_trim = trim or save_top is not None
    if should_trim:
        if low_confidence and not click.confirm(
            f"No confident match found (best score: {best_score:.2f}). "
            "Show results anyway?",
            default=False,
        ):
            return

        from .trimmer import trim_top_results

        count = save_top if save_top is not None else 1
        clip_paths = trim_top_results(results, output_dir, count=count)
        for clip_path in clip_paths:
            click.echo(f"\nSaved clip: {clip_path}")
        if clip_paths:
            _open_file(clip_paths[0])


@cli.command()
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.option("-n", "--results", "n_results", default=5, show_default=True,
              help="Number of results to return.")
@click.option("-o", "--output-dir", default="~/sentrysearch_clips", show_default=True,
              help="Directory to save trimmed clips.")
@click.option("--trim/--no-trim", default=True, show_default=True,
              help="Trim and save the top result as a clip.")
@click.option("--save-top", default=None, type=click.IntRange(min=1),
              help="Save the top N matches as separate clips.")
@click.option("--threshold", default=0.41, show_default=True, type=float,
              help="Minimum similarity score to consider a confident match.")
@click.option("--verbose", is_flag=True, help="Show debug info.")
def img(image, n_results, output_dir, trim, save_top, threshold, verbose):
    """Search indexed footage using an IMAGE as the query."""
    from .embedder import get_embedder, reset_embedder
    from .search import search_footage_by_image
    from .store import VideoStore

    output_dir = os.path.expanduser(output_dir)

    try:
        store = VideoStore()
        if store.get_stats()["total_chunks"] == 0:
            click.echo("No indexed footage found. Run `sentrysearch index <directory>` first.")
            return

        get_embedder()
        if save_top is not None and save_top > n_results:
            n_results = save_top
        results = search_footage_by_image(
            image, store, n_results=n_results, verbose=verbose,
        )
        _present_results(results, threshold, trim, save_top, output_dir, verbose)
    except Exception as e:
        _handle_error(e)
    finally:
        reset_embedder()


@cli.command()
def stats():
    """Print index statistics."""
    if _api_configured():
        s = _api_request("GET", "/v1/stats")
        click.echo(f"Total videos:  {s['total_videos']}")
        click.echo(f"Total chunks:  {s['total_chunks']}")
        return

    from .store import VideoStore

    store = VideoStore()
    s = store.get_stats()
    if s["total_chunks"] == 0:
        click.echo("Index is empty. Run `sentrysearch index <directory>` first.")
        return

    click.echo(f"Total chunks:  {s['total_chunks']}")
    click.echo(f"Source files:  {s['unique_source_files']}")
    click.echo("Embedder:      modal (Qwen/Qwen3-VL-Embedding-2B)")
    click.echo("\nIndexed files:")
    for f in s["source_files"]:
        exists = os.path.exists(f)
        label = "" if exists else "  [missing]"
        click.echo(f"  {f}{label}")


@cli.command()
@click.confirmation_option(prompt="This will delete all indexed data. Continue?")
def reset():
    """Delete all indexed data."""
    from .store import VideoStore

    store = VideoStore()
    s = store.get_stats()
    if s["total_chunks"] == 0:
        click.echo("Index is already empty.")
        return
    for f in s["source_files"]:
        store.remove_file(f)
    click.echo(f"Removed {s['total_chunks']} chunks from {s['unique_source_files']} files.")


@cli.command()
@click.argument("files", nargs=-1, required=True)
def remove(files):
    """Remove specific files from the index.

    Accepts full paths or substrings that match indexed file paths.
    """
    from .store import VideoStore

    store = VideoStore()
    s = store.get_stats()
    if s["total_chunks"] == 0:
        click.echo("Index is empty.")
        return

    removed_total = 0
    for needle in files:
        matches = [f for f in s["source_files"] if needle in f]
        if not matches:
            click.echo(f"No indexed file matched: {needle}")
            continue
        for match in matches:
            removed = store.remove_file(match)
            removed_total += removed
            click.echo(f"Removed {removed} chunks: {match}")

    click.echo(f"Removed {removed_total} chunks total.")


@cli.group()
def dlq():
    """Inspect or clear failed chunk records."""


@dlq.command("list")
def dlq_list():
    from .dlq import DeadLetterQueue

    queue = DeadLetterQueue()
    entries = queue.entries()
    if not entries:
        click.echo("DLQ is empty.")
        return
    for chunk_id, entry in entries.items():
        click.echo(
            f"{chunk_id} {entry['source_file']} "
            f"@ {_fmt_time(entry['start_time'])}-{_fmt_time(entry['end_time'])}: "
            f"{entry['error']}"
        )


@dlq.command("clear")
@click.confirmation_option(prompt="Clear all DLQ entries?")
def dlq_clear():
    from .dlq import DeadLetterQueue

    queue = DeadLetterQueue()
    count = len(queue.entries())
    queue.clear()
    click.echo(f"Cleared {count} DLQ entries.")


@cli.command("job")
@click.argument("job_id")
def job_status(job_id):
    """Print production API job status when server mode is configured."""
    job = _api_request("GET", f"/v1/jobs/{job_id}")
    click.echo(
        f"{job['id']} {job['kind']} {job['status']} "
        f"{job['progress']:.0%} {job.get('message') or ''}"
    )
    if job.get("error"):
        click.secho(job["error"], fg="red", err=True)


@cli.command("api-health")
def api_health():
    """Check the configured production API."""
    response = _api_request("GET", "/healthz")
    click.echo(response.get("status", "unknown"))
