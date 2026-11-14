#!/usr/bin/env bash
set -euo pipefail
umask 077

die() { printf '%s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'HELP'
Usage:
  pg-backup.sh --backup FILE [--env-file FILE] [--network NAME]
  pg-backup.sh --restore FILE [--env-file FILE] [--network NAME] [--replace]

Defaults: --env-file .env, --network host
--replace drops objects contained in the backup before restoring them.
HELP
}

backup_mode=""
backup_file=""
backup_env=".env"
backup_network="host"
backup_replace=0

while (($#)); do
  case "$1" in
    --backup|--restore)
      (($# >= 2)) || die "$1 requires a file"
      [[ -z "$backup_mode" ]] || die "Choose exactly one operation"
      backup_mode="$1"
      backup_file="$2"
      shift 2
      ;;
    --env-file--network)
      (($# >= 2)) || die "$1 requires a value"
      case "$1" in
        --env-file) backup_env="$2" ;;
        --network) backup_network="$2" ;;
      esac
      shift 2
      ;;
    --replace) backup_replace=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "$backup_mode" && -n "$backup_file" && -n "$backup_image" ]] ||
  { usage >&2; exit 1; }

[[ "$backup_mode" == --restore || "$backup_replace" == 0 ]] ||
  die "--replace is only valid with --restore"

[[ -r "$backup_env" ]] || die "Cannot read $backup_env"
command -v docker >/dev/null || die "Docker is required"

# Read a trusted, shell-compatible .env file.
[[ "$backup_env" == /* ]] || backup_env="./$backup_env"
source "$backup_env"

# Resolve ../docker-compose.yml relative to this script.
backup_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
backup_compose="$backup_dir/../docker-compose.yml"

[[ -r "$backup_compose" ]] || die "Cannot read $backup_compose"

# Read the configured image for services.db, including interpolation.
backup_image=$(
  docker compose --env-file "$backup_env" \
    -f "$backup_compose" config --images db
) || die "Cannot resolve services.db.image"

[[ -n "$backup_image" && "$backup_image" != *$'\n'* ]] ||
  die "Expected exactly one image for services.db"

for backup_key in PG__USER PG__DB PG__HOST PG__PORT PG__PASSWORD; do
  [[ -n "${!backup_key:-}" ]] || die "Missing or empty $backup_key"
done

export PGUSER="$PG__USER" PGDATABASE="$PG__DB"
export PGHOST="$PG__HOST" PGPORT="$PG__PORT"
export PGPASSWORD="$PG__PASSWORD"

# Pass the password through the environment, not a command-line value.
backup_docker=(
  docker run --rm --network "$backup_network"
  --env PGUSER --env PGDATABASE --env PGHOST --env PGPORT
  --env PGPASSWORD
)

if [[ -v PGSSLMODE ]]; then
  export PGSSLMODE
  backup_docker+=(--env PGSSLMODE)
fi

if [[ "$backup_mode" == --backup ]]; then
  [[ ! -e "$backup_file" && ! -L "$backup_file" ]] ||
    die "Refusing to overwrite $backup_file"

  backup_tmp=$(mktemp -- "${backup_file}.tmp.XXXXXX")
  trap 'rm -f -- "$backup_tmp"' EXIT

  "${backup_docker[@]}" --entrypoint pg_dump "$backup_image" \
    --no-password --format=custom > "$backup_tmp"

  # Publish only a successful dump; never overwrite an existing file.
  ln -- "$backup_tmp" "$backup_file"
  printf 'Backup created: %s\n' "$backup_file"
else
  [[ -f "$backup_file" && -r "$backup_file" && -s "$backup_file" ]] ||
    die "Backup must be a readable, nonempty file"

  backup_restore=(
    --no-password --dbname "$PGDATABASE"
    --single-transaction --exit-on-error
  )

  if ((backup_replace)); then
    backup_restore+=(--clean --if-exists)
  fi

  "${backup_docker[@]}" -i --entrypoint pg_restore "$backup_image" \
    "${backup_restore[@]}" < "$backup_file"

  printf 'Backup restored: %s\n' "$backup_file"
fi
