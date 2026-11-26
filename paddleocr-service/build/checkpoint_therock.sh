#!/usr/bin/env bash
# Each successful invocation is an immutable Docker layer downstream of LLVM.
set -euo pipefail
stage="${1:-}"
case "$stage" in
  rccl) previous=llvm ;;
  miopen-configure) previous=rccl ;;
  miopen) previous=miopen-configure ;;
  *) echo "Expected rccl, miopen-configure or miopen" >&2; exit 1 ;;
esac
recipe="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_root="${SOURCE_ROOT:-/opt/native-src}"
build="$source_root/therock-build"
evidence="${EVIDENCE:-/opt/ocr/build-evidence}"
targets="$(python "$recipe/../ocr_service/targets.py" --targets "${AMDGPU_TARGETS:?Missing AMDGPU_TARGETS}" --output /dev/null)"
identity="${OCR_ROCM_TRANSPORT:-dual} $targets"
previous_marker="$build/.vision-bill-$previous"
if [[ ! -f "$previous_marker" || "$(<"$previous_marker")" != "$identity" ]]; then
  echo "Missing or mismatched $previous checkpoint for $identity" >&2
  exit 1
fi

patch="$recipe/miopen-sqlite-transient.patch"
if [[ "$stage" == miopen ]]; then
  digest="$(sha256sum "$patch")"
  identity+=" ${digest%% *}"
fi
marker="$build/.vision-bill-$stage"
if [[ -f "$marker" ]]; then
  if [[ "$(<"$marker")" != "$identity" ]]; then
    echo "Mismatched $stage checkpoint for $identity" >&2
    exit 1
  fi
  exit 0
fi

export VISION_BILL_AMDGPU_TARGETS="$targets"
jobs="$(python "$recipe/build_jobs.py" "${BUILD_JOBS:-auto}")"
export CMAKE_BUILD_PARALLEL_LEVEL="$jobs"
case "$stage" in
  rccl)
    TIMEFORMAT='ocr_rocm_phase phase=rccl_configure elapsed_s=%3R'
    time cmake --build "$build" --target rccl+configure
    python "$recipe/verify_rccl_targets.py" "$build" "$evidence/targets.json" \
      | tee "$evidence/therock-rccl-targets.txt"
    TIMEFORMAT='ocr_rocm_phase phase=rccl_compile elapsed_s=%3R'
    time cmake --build "$build" --target rccl
    ;;
  miopen-configure)
    TIMEFORMAT='ocr_rocm_phase phase=miopen_configure elapsed_s=%3R'
    time cmake --build "$build" --target MIOpen+configure
    ;;
  miopen)
    libraries="$source_root/TheRock/rocm-libraries"
    if ! git -C "$libraries" apply --index --reverse --check -- "$patch" 2>/dev/null; then
      git -C "$libraries" apply --index -- "$patch"
    fi
    git -C "$libraries" diff --binary HEAD -- projects/miopen/src/sqlite_db.cpp > "$evidence/miopen-sqlite-source.diff"
    sha256sum "$patch" > "$evidence/miopen-sqlite-patch.sha256"
    TIMEFORMAT='ocr_rocm_phase phase=miopen_compile elapsed_s=%3R'
    time cmake --build "$build" --target MIOpen
    ;;
esac
# Publish only after the complete target (and its validation) has succeeded.
printf '%s\n' "$identity" > "$marker.tmp"
mv "$marker.tmp" "$marker"
