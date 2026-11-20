# ── Stage 1: build the SvelteKit SPA ──────────────────────────────────────
# The built frontend is a git-ignored artifact, so a bare `git clone` has no
# static files. We compile it here at image-build time so the image is
# self-contained: rebuilding the image always ships the latest UI.
FROM node:24-slim AS frontend
WORKDIR /build/frontend

# Locked dependencies first so this layer is cached until the lockfile changes.
COPY frontend/package.json frontend/package-lock.json frontend/.npmrc ./
COPY frontend/patches/ ./patches/
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund

# Full frontend source, then produce the static SPA in /build/frontend/out.
COPY frontend/ ./
COPY paddleocr-service/README.md /build/paddleocr-service/README.md
COPY distribution/README.md /build/distribution/README.md
RUN npm run build

# ── Stage 2: Python API ────────────────────────────────────────────────────
FROM python:3.12-slim

# Install uv (grabbed from the official distroless image, no extra deps needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libmagic1 && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into a project-local venv, based only on the lockfile
# (no source yet, so this layer is cached until deps actually change)
RUN --mount=type=cache,target=/root/.cache/uv UV_LINK_MODE=copy uv sync --frozen --no-install-project

# Now copy the source (plus the alembic migration tooling the app relies on)
COPY ./src ./src
COPY ./paddleocr-service/ocr_contract ./paddleocr-service/ocr_contract

# The editable install reads the SPA from the source tree at runtime.
RUN --mount=type=cache,target=/root/.cache/uv UV_LINK_MODE=copy uv sync --frozen

COPY alembic.ini ./
COPY ./alembic ./alembic

# Make sure the venv's binaries are used
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

# Entrypoint migrates the schema, then execs the CMD below (uvicorn) as PID 1
# so `docker stop` reaches it for a graceful shutdown (satisfies BuildKit's
# JSONArgsRecommended check, which a shell-form `CMD … && …` cannot). Running
# the migration here keeps the image self-sufficient: `docker run` on a fresh
# DB works, and new migrations apply on upgrade. docker-compose's
# `depends_on: db: service_healthy` gate keeps this safe when orchestrated.
COPY --chmod=755 docker-entrypoint.sh /docker-entrypoint.sh

# Join the independent frontend and Python builds after both have finished.
COPY --from=frontend /build/frontend/out/. ./src/vision_bill/static/
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["uvicorn", "src.vision_bill.main:app", "--host", "0.0.0.0", "--port", "8080"]
