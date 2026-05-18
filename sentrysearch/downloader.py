"""Download web videos for local indexing."""

from pathlib import Path


DEFAULT_DOWNLOAD_DIR = Path.home() / "sentrysearch_downloads"


class VideoDownloadError(RuntimeError):
    """Raised when a video URL cannot be downloaded."""


def download_video_url(
    url: str,
    output_dir: str | Path | None = None,
    *,
    max_height: int = 480,
    verbose: bool = False,
) -> str:
    """Download a lightweight MP4 for a URL supported by yt-dlp."""
    from .chunker import _get_ffmpeg_executable

    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise VideoDownloadError(
            "yt-dlp is not installed. Run `uv sync` to install project dependencies."
        ) from exc

    target_dir = Path(output_dir or DEFAULT_DOWNLOAD_DIR).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(target_dir / "%(title).180B [%(id)s].%(ext)s"),
        "noplaylist": True,
        "restrictfilenames": True,
        "quiet": not verbose,
        "no_warnings": not verbose,
        "ffmpeg_location": _get_ffmpeg_executable(),
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = ydl.prepare_filename(info)
    except DownloadError as exc:
        raise VideoDownloadError(str(exc)) from exc

    downloaded = Path(path)
    if downloaded.suffix.lower() != ".mp4":
        downloaded = downloaded.with_suffix(".mp4")
    if not downloaded.exists():
        matches = sorted(target_dir.glob(f"*[{info.get('id', '')}]*.mp4"))
        if matches:
            downloaded = matches[-1]
    if not downloaded.exists():
        raise VideoDownloadError("yt-dlp finished but no downloaded MP4 was found.")

    return str(downloaded)
