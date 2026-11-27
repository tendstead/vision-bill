#!/usr/bin/env bash
# Run the Go tests. Integration tests need PostgreSQL: use VB_TEST_DATABASE_URL
# when set, otherwise start a throwaway container for the duration of the run.
set -euo pipefail

if [[ -z "${VB_TEST_DATABASE_URL:-}" ]]; then
  name="vb-test-pg-$$"
  docker run -d --rm --name "$name" -e POSTGRES_PASSWORD=pw -e POSTGRES_USER=vb \
    -p 127.0.0.1::5432 postgres:18 >/dev/null
  trap 'docker stop "$name" >/dev/null 2>&1 || true' EXIT
  for _ in $(seq 1 60); do
    docker exec "$name" pg_isready -U vb -q 2>/dev/null && break
    sleep 1
  done
  # pg_isready can answer during initialisation; wait for a real login.
  for _ in $(seq 1 30); do
    docker exec "$name" psql -U vb -d postgres -qc 'select 1' >/dev/null 2>&1 && break
    sleep 1
  done
  port="$(docker port "$name" 5432/tcp | head -n1 | sed 's/.*://')"
  export VB_TEST_DATABASE_URL="postgres://vb:pw@127.0.0.1:${port}/postgres"
fi

go test -race -count=1 "$@" ./cmd/... ./internal/...
