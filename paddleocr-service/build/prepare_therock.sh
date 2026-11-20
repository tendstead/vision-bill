#!/usr/bin/env bash
set -euo pipefail

SOURCE="${1:?Pass the pinned TheRock checkout}"
INVENTORY="${2:?Pass the source inventory output path}"
JOBS="${3:?Pass the build job count}"
RECIPE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# TheRock's fetch only initializes selected top-level sources. Populate all
# recorded pins first so an update cannot reset an upstream-patched checkout.
git -C "$SOURCE" submodule update --init --recursive --filter=blob:none --checkout --jobs "$JOBS"
(
  cd "$SOURCE"
  python build_tools/fetch_sources.py --no-remote --apply-patches
)
python "$RECIPE/source_inventory.py" "$SOURCE" "$INVENTORY"
