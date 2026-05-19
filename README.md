# SentrySearch

Semantic search over video footage using a Modal-hosted
`Qwen/Qwen3-VL-Embedding-2B` service. Index videos locally, store vectors in
local ChromaDB, then search by text or image and trim matching clips.

## Architecture

```text
local videos
  -> ffmpeg chunks/preprocessing
  -> Modal remote Qwen3-VL-Embedding-2B methods
  -> local ChromaDB at ~/.sentrysearch/db

text/image query
  -> same Modal remote methods
  -> ChromaDB nearest-neighbor search
  -> timestamped results / trimmed clips
```

The original Gemini, local GPU, and Tesla overlay paths have been removed. The
repo is now focused on one production path: Modal-hosted Qwen3-VL embeddings.

## Install

```bash
uv sync
```

## Deploy The Modal Embedder

Authenticate Modal first if you have not already:

```bash
modal setup
```

Deploy the Qwen3-VL-Embedding-2B remote class:

```bash
modal deploy sentrysearch/modal_app.py
```

That is the only deploy step. The local CLI calls the deployed class through
the Modal Python SDK with `modal.Cls.from_name(...)`, so no web endpoint URL or
extra environment variable is needed.

The Modal app uses an `L40S` GPU by default and caches model weights in a Modal
Volume named `qwen3-vl-embedding-2b-cache`. The first embedding call after a new
deploy may take longer while the model downloads into that Volume.

## Index Footage

```bash
sentrysearch index /path/to/video/footage
```

Options:

- `--chunk-duration 30`: seconds per chunk
- `--overlap 5`: overlap between chunks
- `--no-preprocess`: skip downscaling/frame-rate reduction
- `--target-resolution 480`: target height for preprocessing
- `--target-fps 5`: target frame rate for preprocessing
- `--no-skip-still`: embed still chunks too
- `--batch-size 4`: chunks per Modal embedding call
- `--retry-failed`: retry chunks recorded in the dead-letter queue

Supported video extensions: `.mp4`, `.mov`.

## Download A Video URL

Use `yt-dlp` to save a lightweight local MP4 from a supported video URL:

```bash
sentrysearch download-url "https://youtu.be/..." --max-height 480
```

Save and index in one step:

```bash
sentrysearch download-url "https://youtu.be/..." --index
```

Downloads are saved to `~/sentrysearch_downloads` by default. Only download
videos you have the right to use.

## Search

```bash
sentrysearch search "red truck running a stop sign"
```

Useful flags:

- `--results 10`: show more matches
- `--no-trim`: only show ranked results
- `--save-top 3`: trim the top three matches
- `--threshold 0.5`: adjust low-confidence prompt threshold
- `--output-dir ~/clips`: change clip output directory

## Search By Image

```bash
sentrysearch img /path/to/reference.jpg
```

The image is embedded into the same retrieval space as text queries and video
chunks.

## Manage The Index

```bash
sentrysearch stats
sentrysearch remove video-name-or-path-substring
sentrysearch reset
```

Embeddings are stored locally in:

```text
~/.sentrysearch/db
```

ChromaDB stores vectors and metadata only. Original videos are not copied; the
index points back to source paths for trimming.

## Failed Chunks

Chunks that fail repeatedly during indexing are recorded in:

```text
~/.sentrysearch/dlq.json
```

Inspect or clear them:

```bash
sentrysearch dlq list
sentrysearch dlq clear
```

## Development

```bash
uv run pytest
```

## Production Docker Stack

SentrySearch can also run as a single-node production stack with FastAPI,
Celery, Redis, Postgres/pgvector, and MinIO:

```bash
cp .env.example .env
# edit SENTRYSEARCH_API_KEY before exposing the service
docker compose up -d --build
```

The API is published on:

```text
http://localhost:8010
```

Core endpoints:

- `GET /healthz`
- `POST /v1/videos/upload`
- `POST /v1/videos/url`
- `POST /v1/search`
- `POST /v1/clips`
- `GET /v1/jobs/{job_id}`
- `GET /v1/stats`

Protected endpoints require:

```text
X-API-Key: <SENTRYSEARCH_API_KEY>
```

To point the CLI at the production API:

```bash
export SENTRYSEARCH_API_URL=http://localhost:8010
export SENTRYSEARCH_API_KEY=<your-api-key>
sentrysearch stats
sentrysearch index /path/to/video.mp4
sentrysearch search "red truck"
```

In API mode, indexing a single file uploads it to MinIO and queues an indexing
job. Directory indexing uses the container's mounted `./media:/media:ro` path,
so directory paths must be visible inside the API container.
