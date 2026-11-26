#!/bin/sh
# Image entrypoint: migrate the schema, then start the server.
#
# We `exec` the final command so the server replaces this shell and becomes
# PID 1. That is what lets `docker stop` (SIGTERM) reach uvicorn directly for
# a graceful shutdown — the exact behavior BuildKit's JSONArgsRecommended
# check asks for, which a plain shell-form `CMD … && …` cannot provide.
#
# Running the migration here (rather than only in docker-compose) keeps the
# image self-sufficient: `docker run` on a fresh DB works, and new migrations
# apply on upgrade. docker-compose's `depends_on: db: service_healthy` gate
# keeps this safe when orchestrated.
set -e

# Prepared-install commands must work before any database or settings exist.
case "${1:-}" in
    init)
        shift
        exec python -m vision_bill.distribution.bootstrap "$@"
        ;;
    provision-model)
        echo "Bundled model provisioning was removed; configure an external Ollama or OpenAI endpoint and explicit model in runtime.env." >&2
        exit 64
        ;;
esac

if [ "${VISION_BILL_MANAGED_OCR:-}" = "true" ]; then
    python -m vision_bill.distribution.bootstrap --check-managed-ocr
fi

# Dependencies are installed during the image build. Startup must not resolve
# or rebuild packages before running the migrations from that locked environment.
alembic upgrade head

exec "$@"
