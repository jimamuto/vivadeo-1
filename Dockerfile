FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md uv.lock ./
COPY sentrysearch ./sentrysearch
COPY alembic.ini ./
COPY alembic ./alembic

RUN pip install --no-cache-dir uv \
    && uv pip install --system .

EXPOSE 8000

CMD ["uvicorn", "sentrysearch.api:app", "--host", "0.0.0.0", "--port", "8000"]
