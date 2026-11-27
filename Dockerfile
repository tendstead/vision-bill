# ── Stage 1: build the SvelteKit SPA ──────────────────────────────────────
# The built frontend is a git-ignored artifact, so a bare `git clone` has no
# static files. It is compiled here so the image is self-contained.
FROM node:24-slim AS web
WORKDIR /build/web

# Locked dependencies first so this layer is cached until the lockfile changes.
COPY web/package.json web/package-lock.json web/.npmrc ./
COPY web/patches/ ./patches/
RUN npm ci --no-audit --no-fund

COPY web/ ./
RUN npm run build

# ── Stage 2: Go server (static binary, SPA embedded) ──────────────────────
FROM golang:1.26 AS server
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd/ cmd/
COPY internal/ internal/
COPY --from=web /build/web/out/. cmd/server/static/
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /vision-bill ./cmd/server \
 && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /vbctl ./cmd/vbctl

# ── Stage 3: runtime ──────────────────────────────────────────────────────
# The only runtime dependency besides the binary is the image helper (vips)
# used for thumbnails and HEIC/HEIF decoding. The build fails when it is
# missing or lacks HEIF support, so every image supports HEIC/HEIF uploads.
FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates tzdata libvips-tools libheif1 \
 && rm -rf /var/lib/apt/lists/* \
 && vips --version \
 && vips -l | grep -q heifload \
 && vips -l | grep -q webpsave

WORKDIR /app
COPY --from=server /vision-bill /app/vision-bill
COPY --from=server /vbctl /app/vbctl

EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=4s --start-period=20s --retries=3 CMD ["/app/vision-bill", "-healthcheck"]
ENTRYPOINT ["/app/vision-bill"]
