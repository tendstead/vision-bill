#!/usr/bin/env bash
set -euo pipefail
umask 077

fail() {
  printf '%s\n' "$*" >&2
  exit 1
}

usage() {
  printf '%s\n' 'Usage: ./start-linux.sh [--port PORT] [--expected-manifest-sha256 SHA256]' \
    '                        [--verify-only] [--show-initial-admin]'
}

port=8080
expected_manifest=
verify_only=false
show_admin=false
while (( $# )); do
  case "$1" in
    --port)
      [[ ${2:-} =~ ^[0-9]{1,5}$ ]] && (( 10#$2 >= 1 && 10#$2 <= 65535 )) ||
        fail 'Choose a port from 1 to 65535.'
      port=$((10#$2))
      shift
      ;;
    --expected-manifest-sha256)
      [[ ${2:-} =~ ^[a-fA-F0-9]{64}$ ]] || fail 'Supply a SHA-256 digest for SHA256SUMS.'
      expected_manifest=${2,,}
      shift
      ;;
    --verify-only) verify_only=true ;;
    --show-initial-admin) show_admin=true ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 1 ;;
  esac
  shift
done

runtime=runtime.env
identity=.vision-bill-install
staging=.vision-bill-start.tmp
[[ ! -L ${BASH_SOURCE[0]} ]] || fail 'Run the regular launcher file in the extracted bundle.'
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")"
bundle=$PWD

verify_bundle() {
  local line path parent digest actual
  local manifest_pattern='^([a-f0-9]{64})  ([A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*)$'
  local -A files=() directories=()
  [[ -f SHA256SUMS && ! -L SHA256SUMS && $(wc -c < SHA256SUMS) -le 1048576 ]] ||
    fail 'Expected a regular SHA256SUMS file no larger than 1 MiB.'
  if [[ -n $expected_manifest ]]; then
    actual=$(sha256sum SHA256SUMS)
    [[ ${actual%% *} == "$expected_manifest" ]] ||
      fail 'SHA256SUMS differs from the trusted manifest hash.'
  fi

  while IFS= read -r line || [[ -n $line ]]; do
    [[ $line =~ $manifest_pattern ]] || fail 'Malformed SHA256SUMS entry.'
    digest=${BASH_REMATCH[1]}
    path=${BASH_REMATCH[2]}
    case "$path" in
      compose.yaml|compose.wsl.yaml|release.json|runtime.env.example|check-compose.sh|start-linux.sh|start-windows.ps1|release/*) ;;
      *) fail "Unexpected path in SHA256SUMS: $path" ;;
    esac
    [[ ! ${files[$path]+present} ]] || fail "Repeated checksum path: $path"
    files[$path]=$digest
    parent=$path
    while [[ $parent == */* ]]; do
      parent=${parent%/*}
      directories[$parent]=1
    done
  done < SHA256SUMS
  for path in compose.yaml release.json runtime.env.example check-compose.sh start-linux.sh; do
    [[ ${files[$path]+present} ]] || fail "Missing bundle checksum: $path"
  done
  [[ ${directories[release]+present} ]] || fail 'Missing release assets in SHA256SUMS.'

  find . -mindepth 1 -print0 | while IFS= read -r -d '' path; do
    path=${path#./}
    [[ ! -L $path ]] || fail "Symbolic link in the bundle: $path"
    if [[ -d $path ]]; then
      [[ ${directories[$path]+present} ]] || fail "Unexpected directory in the bundle: $path"
    elif [[ -f $path ]]; then
      case "$path" in
        SHA256SUMS|"$runtime"|"$identity"|"$staging") ;;
        *) [[ ${files[$path]+present} ]] || fail "Unexpected file in the bundle: $path" ;;
      esac
    else
      fail "Expected a regular file or directory in the bundle: $path"
    fi
  done

  sha256sum --check --strict --status SHA256SUMS ||
    fail 'A bundle file is missing or has a checksum mismatch.'
  # These are fixed top-level fields of the generated release metadata.
  grep -Fxq '  "schema_version": 2,' release.json &&
    grep -Fxq '  "platform": "linux/amd64",' release.json &&
    grep -Eq '^  "variant": "app-ocr-(cuda|rocm)",$' release.json ||
    fail 'Extract a verified app-ocr-cuda or app-ocr-rocm Linux AMD64 release.'
}

assert_private_file() {
  local path=$1
  [[ -f $path && ! -L $path && $(stat -c '%a:%u:%h' -- "$path") == "600:$EUID:1" ]] ||
    fail "Keep $path as a regular file owned by your Linux account with mode 0600 and one link."
}

write_private_file() {
  local path=$1
  shift
  (set -o noclobber; printf '%s\n' "$@" > "$staging")
  mv -T -- "$staging" "$path"
}

read_answer() {
  printf '%s' "$1" >&2
  IFS= read -r "$2" || fail 'Complete setup in a terminal and run the launcher again.'
}

new_runtime() {
  local mode provider endpoint default_endpoint hint host model key
  local url_pattern='^https?://(\[[[:alnum:]:.%_-]+\]|[[:alnum:]._-]+)(:[0-9]+)?(/[^[:space:][:cntrl:]#]*)?$'
  local -a settings
  while true; do
    read_answer 'LLM connection type (ollama, llamacpp, openai) [ollama]: ' mode
    mode=${mode,,}
    case "${mode:-ollama}" in
      ollama) mode=ollama; provider=ollama; default_endpoint=http://host.docker.internal:11434; break ;;
      llamacpp) provider=openai; default_endpoint=http://host.docker.internal:8083/v1; break ;;
      openai) provider=openai; default_endpoint=; break ;;
    esac
  done
  hint=
  [[ -z $default_endpoint ]] || hint=" [$default_endpoint]"
  while true; do
    read_answer "LLM endpoint URL$hint: " endpoint
    endpoint=${endpoint:-$default_endpoint}
    if [[ $endpoint =~ $url_pattern ]]; then
      host=${BASH_REMATCH[1],,}
      case "$host" in
        localhost|localhost.*|127.*|0.0.0.0|'[::]'|'[::1]') ;;
        *) break ;;
      esac
    fi
    printf '%s\n' 'Enter a reachable HTTP or HTTPS URL without credentials. Use host.docker.internal for a service on this host.' >&2
  done
  while true; do
    read_answer 'Exact installed model ID: ' model
    [[ $model =~ [^[:space:]] && $model != *$'\r'* ]] && break
  done
  settings=("LLM__PROVIDER=$provider" "LLM__HOST=$endpoint" "LLM__MODEL_NAME=$model")
  if [[ $mode == llamacpp ]]; then
    settings+=(LLM__MODEL_LIFECYCLE=llamacpp)
  fi
  if [[ $provider == openai ]]; then
    printf '%s' 'API key (Enter for a credentialless endpoint): ' >&2
    IFS= read -rs key || fail 'Complete setup in a terminal and run the launcher again.'
    printf '\n' >&2
    [[ $key != *$'\r'* ]] || fail 'Enter the API key on one line.'
    [[ -z $key ]] || settings+=("LLM__API_KEY=$key")
  fi
  write_private_file "$runtime" "${settings[@]}"
}

select_docker() {
  local context endpoint engine
  command -v docker >/dev/null || fail 'Install Docker Engine and the Docker Compose plugin.'
  if [[ -z ${DOCKER_CONTEXT:-} && -n ${DOCKER_HOST:-} ]]; then
    endpoint=$DOCKER_HOST
  else
    context=$(docker context show) || fail 'Select a local Docker context.'
    endpoint=$(docker context inspect "$context" --format '{{.Endpoints.docker.Host}}') ||
      fail 'Select a local Docker context.'
  fi
  [[ $endpoint == unix:///* ]] || fail 'Use a local Docker Unix socket on this Linux host.'
  # Pin every command, including check-compose.sh, to the same selected socket.
  unset DOCKER_CONTEXT DOCKER_TLS DOCKER_TLS_VERIFY DOCKER_CERT_PATH
  export DOCKER_HOST="$endpoint"
  engine=$(docker info --format '{{.OSType}}|{{.Architecture}}') ||
    fail 'Start the local Docker engine and ensure your account can use its socket.'
  [[ $(uname -s) == Linux && $engine == 'linux|x86_64' ]] ||
    fail 'Use a local Linux x86_64 Docker engine for this bundle.'
}

load_compose_resources() {
  local resources role
  resources=$("${compose[@]}" config --volumes)
  while IFS= read -r role; do
    [[ $role =~ ^[a-z0-9][a-z0-9_-]*$ ]] || fail 'Invalid volume in the verified Compose file.'
    volumes[$role]=1
  done <<< "$resources"
  resources=$("${compose[@]}" config --services)
  while IFS= read -r role; do
    [[ $role =~ ^[a-z0-9][a-z0-9_-]*$ ]] || fail 'Invalid service in the verified Compose file.'
    services[$role]=1
  done <<< "$resources"
  [[ ${volumes[app-config]+present} && ${services[app]+present} ]] ||
    fail 'Expected app and app-config in the verified Compose file.'
}

assert_project_resources() {
  local rows name driver tagged_project role owner
  has_resources=false
  rows=$(docker volume ls --format \
    '{{.Name}}|{{.Driver}}|{{.Label "com.docker.compose.project"}}|{{.Label "com.docker.compose.volume"}}|{{.Label "io.vision-bill.installation"}}')
  while IFS='|' read -r name driver tagged_project role owner; do
    if [[ $tagged_project == "$project" || $name == "${project}_"* ]]; then
      [[ $driver == local && $tagged_project == "$project" && $owner == "$token" &&
         -n $role && ${volumes[$role]+present} && $name == "${project}_$role" ]] ||
        fail "Unexpected Docker volume ownership: $name"
      has_resources=true
    fi
  done <<< "$rows"
  rows=$(docker ps --all --format \
    '{{.Names}}|{{.Label "com.docker.compose.project"}}|{{.Label "com.docker.compose.service"}}|{{.Label "io.vision-bill.installation"}}')
  while IFS='|' read -r name tagged_project role owner; do
    if [[ $tagged_project == "$project" || $name == "${project}-"* || $name == "${project}_"* ]]; then
      [[ $tagged_project == "$project" && $owner == "$token" &&
         -n $role && ${services[$role]+present} && $name == "${project}-$role-1" ]] ||
        fail "Unexpected Docker container ownership: $name"
      has_resources=true
    fi
  done <<< "$rows"
  rows=$(docker network ls --format \
    '{{.Name}}|{{.Label "com.docker.compose.project"}}|{{.Label "com.docker.compose.network"}}|{{.Label "io.vision-bill.installation"}}')
  while IFS='|' read -r name tagged_project role owner; do
    if [[ $tagged_project == "$project" || $name == "${project}_default" ]]; then
      [[ $tagged_project == "$project" && $role == default && $owner == "$token" &&
         $name == "${project}_default" ]] || fail "Unexpected Docker network ownership: $name"
    fi
  done <<< "$rows"
}

assert_claim() {
  local record
  record=$(docker volume inspect --format \
    '{{.Name}}|{{.Driver}}|{{index .Labels "io.vision-bill.launcher-owner"}}' "$claim")
  [[ $record == "$claim|local|$token" ]] ||
    fail "Docker project $project is claimed by a different installation."
}

verify_bundle
if $verify_only; then
  printf '%s\n' 'Bundle files verified.'
  exit 0
fi
command -v flock >/dev/null || fail 'Install util-linux for the flock command.'
exec {bundle_lock}< .
flock -n "$bundle_lock" || fail 'Another Vision Bill setup is using this bundle directory.'
if [[ -e $staging || -L $staging ]]; then
  assert_private_file "$staging"
  rm -- "$staging"
fi
trap 'rm -f -- "$staging"' EXIT
select_docker

if [[ ! -e $runtime && ! -L $runtime ]]; then
  new_runtime
fi
assert_private_file "$runtime"
if [[ ! -e $identity && ! -L $identity ]]; then
  token=$(od -An -N24 -tx1 /dev/urandom | tr -d '[:space:]')
  [[ $token =~ ^[a-f0-9]{48}$ ]] || fail 'Could not generate an installation identity.'
  write_private_file "$identity" "$token"
fi
assert_private_file "$identity"
mapfile -t identities < "$identity"
[[ ${#identities[@]} -eq 1 && ${identities[0]} =~ ^[a-f0-9]{48}$ ]] ||
  fail 'The saved installation identity is invalid.'
token=${identities[0]}
project=vision-bill-${token:0:16}
claim=${project}-vision-bill-launcher-owner

unset COMPOSE_FILE COMPOSE_ENV_FILES COMPOSE_PROFILES COMPOSE_COMPATIBILITY COMPOSE_REMOVE_ORPHANS
export COMPOSE_DISABLE_ENV_FILE=1 COMPOSE_PROJECT_NAME="$project" VISION_BILL_INSTALLATION_ID="$token"
export APP_BIND_ADDRESS=127.0.0.1 APP_PORT="$port"
compose_file="$bundle/compose.yaml"
if grep -Fxq '  "variant": "app-ocr-rocm",' release.json && grep -qi microsoft /proc/sys/kernel/osrelease; then
  [[ -c /dev/dxg && -f /usr/lib/wsl/lib/libdxcore.so ]] || fail 'Install the AMD Windows WSL driver and enable GPU access in this WSL 2 distro.'
  [[ ${OCR_ROCM_DEVICE_ID:-0} =~ ^(0|[1-9][0-9]{0,3})$ ]] && (( ${OCR_ROCM_DEVICE_ID:-0} <= 1024 )) ||
    fail 'Select one AMD WSL GPU by numeric index in OCR_ROCM_DEVICE_ID.'
  compose_file="$bundle/compose.wsl.yaml"
fi
compose=(docker compose --file "$compose_file" --project-name "$project")
sh ./check-compose.sh || fail 'Update Docker Compose to 2.30 or later and check the bundle configuration.'
declare -A volumes=() services=()
load_compose_resources

# The daemon serializes creation by this exact name and preserves its first labels.
assert_project_resources
docker volume create --driver local --label "io.vision-bill.launcher-owner=$token" "$claim" >/dev/null
assert_claim
assert_project_resources
fresh=true
if $has_resources; then fresh=false; fi

"${compose[@]}" up --detach --no-build --wait --wait-timeout 300 ||
  fail 'Vision Bill could not start. Check the Compose output and run the starter again.'
assert_claim
assert_project_resources
if $fresh || $show_admin; then
  printf '%s\n' 'Initial administrator credentials for this installation'
  "${compose[@]}" exec -T app cat /app/config/initial-admin.yaml ||
    printf '%s\n' 'Use the administrator account for your existing installation.'
fi
printf 'Vision Bill is running at http://127.0.0.1:%s/\n' "$port"
printf '%s\n' 'Sign in as administrator and open Settings > Dedicated OCR to start setup.'
