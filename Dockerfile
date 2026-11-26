# Collect only public guide inputs on an empty filesystem. Builders using
# Docker's vfs storage driver copy the base filesystem for every COPY layer.
FROM scratch AS guide-inputs
COPY README.md AGENTS.md HAENDLERVERZEICHNIS.md MULTI_IMAGE_RECEIPTS.md OPENAI_COMPATIBLE.md RELEASES.md UPGRADE_AND_VERIFY.md ./
COPY .github/workflows/distribution-inputs.yml .github/workflows/distribution-release.yml .github/workflows/
COPY distribution/README.md distribution/artifacts.schema.json distribution/
COPY frontend/README.md frontend/
COPY frontend/patches/README.md frontend/patches/
COPY frontend/scripts/operator_guides.py frontend/scripts/
COPY paddleocr-service/README.md paddleocr-service/profile.json paddleocr-service/release-bundle.schema.json paddleocr-service/
COPY paddleocr-service/first-run-fixtures/README.md paddleocr-service/first-run-fixtures/
COPY scripts/README.md scripts/distribution.mjs scripts/receipt_split_experiment.md scripts/receipt_split_results.md scripts/
COPY scripts/ocr_harness/README.md scripts/ocr_harness/BENCHMARK_RESULTS.md scripts/ocr_harness/
COPY tests/e2e/data/README.md tests/e2e/data/

FROM python:3.12-slim AS operator-guides
RUN --mount=type=bind,from=guide-inputs,target=/repo python /repo/frontend/scripts/operator_guides.py /guides.zip

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
COPY src/vision_bill/model/product_tags.json /build/src/vision_bill/model/product_tags.json
COPY paddleocr-service/README.md /build/paddleocr-service/README.md
COPY distribution/README.md /build/distribution/README.md
COPY --from=operator-guides /guides.zip ./static/vision-bill-operator-guides.zip
# The Python stage above already ran the prebuild step.
RUN npm --ignore-scripts run build

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
