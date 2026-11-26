#!/usr/bin/env bash
# Unexecuted native qualification candidate; this script never declares GPU support.
set -euo pipefail
SOURCE_ROOT="${SOURCE_ROOT:-/opt/native-src}"
SDK="${SDK:-/opt/rocm}"
RECIPE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_JOBS="$(python "$RECIPE/build_jobs.py" "${BUILD_JOBS:-auto}")"
echo "Native build parallelism: $BUILD_JOBS jobs"
EVIDENCE="${EVIDENCE:-/opt/ocr/build-evidence}"
THEROCK_SRC="$SOURCE_ROOT/TheRock"
THEROCK_BUILD="$SOURCE_ROOT/therock-build"
WIN_HEADERS="$SOURCE_ROOT/win32metadata"
transport="${OCR_ROCM_TRANSPORT:-dual}"
case "$transport" in
  kfd|wsl|dual) ;;
  *) echo "Invalid OCR_ROCM_TRANSPORT: $transport (expected kfd, wsl, or dual)" >&2; exit 1 ;;
esac
if [[ -z "${AMDGPU_TARGETS:-}" ]]; then
  echo "Use make ocr-build-rocm, set OCR_AMDGPU_TARGETS for Compose, or pass --build-arg AMDGPU_TARGETS to Docker." >&2
  exit 1
fi

select_targets() {
  TARGETS="$(python "$RECIPE/../ocr_service/targets.py" --targets "$AMDGPU_TARGETS" --output "$1")"
  export VISION_BILL_AMDGPU_TARGETS="$TARGETS"
}

require_checkpoint() {
  local marker="$THEROCK_BUILD/.vision-bill-$1"
  select_targets /dev/null
  if [[ ! -f "$marker" || "$(<"$marker")" != "$transport $TARGETS" ]]; then
    echo "Missing or mismatched TheRock $1 checkpoint for $transport $TARGETS" >&2
    exit 1
  fi
}

# Measure each tree in allocated bytes. Only prepare traverses the source checkout.
record_size() {
  local usage
  usage="$(du -s -B1 -- "$2")"
  printf 'ocr_rocm_checkpoint phase=%s path=%s allocated_bytes=%s\n' "$1" "$2" "${usage%%[[:space:]]*}" \
    | tee -a "$EVIDENCE/therock-checkpoints.txt"
}

prepare() {
  # Fresh sources/caches only: a previous native build is not a reproducible input.
  for path in "$THEROCK_SRC" "$THEROCK_BUILD" "$WIN_HEADERS"; do
    if [[ -e "$path" || -L "$path" ]]; then
      echo "Native build path already exists: $path" >&2
      exit 1
    fi
  done
  mkdir -p "$SOURCE_ROOT" "$EVIDENCE"
  printf '%s\n' "$transport" > "$EVIDENCE/rocm-transport.txt"
  select_targets "$EVIDENCE/targets.json"

  git init --quiet "$THEROCK_SRC"
  git -C "$THEROCK_SRC" remote add origin https://github.com/ROCm/TheRock.git
  local TIMEFORMAT='ocr_rocm_phase phase=therock_sources elapsed_s=%3R'
  time bash "$RECIPE/prepare_therock.sh" "$THEROCK_SRC" "$EVIDENCE/therock-sources.json" "$BUILD_JOBS" \
    b324869fb3116353ac7eec5e6926aa3d452ac987 \
    . "$RECIPE/therock-sqlite-headers.patch" . "$RECIPE/therock-hipblaslt-rdna2.patch" \
    rocm-libraries "$RECIPE/rocm-libraries-hipblaslt-rdna2.patch" \
    . "$RECIPE/therock-rccl-translate.patch"
  wsl_options=(-DTHEROCK_ENABLE_WSL=OFF -DTHEROCK_ENABLE_WSL_ROCDXG=OFF)
  if [[ "$transport" != kfd ]]; then
    # The WSL thunk needs Windows SDK headers at compile time and the AMD WKMI
    # archive referenced by the rocm-systems gitlink. Neither is a host driver.
    git init -q "$WIN_HEADERS"
    git -C "$WIN_HEADERS" remote add origin https://github.com/microsoft/win32metadata.git
    git -C "$WIN_HEADERS" sparse-checkout set generation/WinSDK/RecompiledIdlHeaders/shared
    git -C "$WIN_HEADERS" fetch --depth=1 --filter=blob:none origin a40ed3186943620f304649a077b78f8271034736
    git -C "$WIN_HEADERS" checkout --detach FETCH_HEAD
    git -C "$WIN_HEADERS" rev-parse HEAD > "$EVIDENCE/windows-sdk-headers.commit"
    WIN_SDK="$WIN_HEADERS/generation/WinSDK/RecompiledIdlHeaders/shared"
    test -s "$WIN_SDK/d3dkmthk.h"
    WKMI="$THEROCK_SRC/rocm-systems/shared/amdgpu-windows-interop/wkmi/lnx/lib/libwkmi.a"
    grep -Fxq -- '- md5: 53126f6263d5ca1381b84e8a637e6a1e' "$WKMI.dvc"
    curl --fail --location --retry 3 \
      https://therock-dvc.s3.us-east-2.amazonaws.com/rocm-systems/files/md5/53/126f6263d5ca1381b84e8a637e6a1e \
      --output "$WKMI"
    test "$(stat -c %s "$WKMI")" = 333088
    echo "53126f6263d5ca1381b84e8a637e6a1e  $WKMI" | md5sum -c -
    sha256sum "$WKMI" > "$EVIDENCE/wkmi.sha256"
    wsl_options=(-DTHEROCK_ENABLE_WSL=ON -DTHEROCK_ENABLE_WSL_ROCDXG=ON "-DTHEROCK_WSL_WIN_SDK=$WIN_SDK")
  fi
  (
    cd "$THEROCK_SRC"
    INSTALL_PREFIX=/usr/local bash dockerfiles/install_pinned_patchelf.sh
  )
  # LLVM links and Flang compiles have large working sets. Limit these pools
  # independently of overall parallelism while preserving single-job builds.
  flang_jobs=2
  if [[ "$BUILD_JOBS" == 1 ]]; then flang_jobs=1; fi
  cmake -S "$THEROCK_SRC" -B "$THEROCK_BUILD" -GNinja \
    -DCMAKE_BUILD_TYPE=Release -DTHEROCK_ENABLE_ALL=ON \
    -DTHEROCK_ENABLE_MIRAGE=OFF -DTHEROCK_ENABLE_HIPDNN_SAMPLES=OFF \
    -DLLVM_PARALLEL_LINK_JOBS=1 "-DFLANG_PARALLEL_COMPILE_JOBS=$flang_jobs" \
    -DTHEROCK_BACKGROUND_BUILD_JOBS=1 "${wsl_options[@]}" \
    -DTHEROCK_AMDGPU_FAMILIES= "-DTHEROCK_AMDGPU_TARGETS=$TARGETS" \
    -DTHEROCK_DIST_AMDGPU_FAMILIES= "-DTHEROCK_DIST_AMDGPU_TARGETS=$TARGETS" \
    -DTHEROCK_TEST_AMDGPU_FAMILIES= "-DTHEROCK_TEST_AMDGPU_TARGETS=$TARGETS" \
    -DTHEROCK_AMDGPU_DIST_BUNDLE_NAME=vision-bill-candidate-v1 -DBUILD_TESTING=OFF
  python "$RECIPE/verify_therock_targets.py" "$THEROCK_BUILD" "$EVIDENCE/targets.json" \
    | tee "$EVIDENCE/therock-toolchain-targets.txt"
  printf '%s\n' "$transport $TARGETS" > "$THEROCK_BUILD/.vision-bill-prepare"

  : > "$EVIDENCE/therock-checkpoints.txt"
  record_size prepare "$THEROCK_SRC"
  if [[ "$transport" != kfd ]]; then record_size prepare "$WIN_HEADERS"; fi
  record_size prepare "$THEROCK_BUILD"
}

build_llvm() {
  require_checkpoint prepare
  local TIMEFORMAT='ocr_rocm_phase phase=llvm_compile elapsed_s=%3R'
  time CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS" cmake --build "$THEROCK_BUILD" --target amd-llvm
  cp "$THEROCK_BUILD/.vision-bill-prepare" "$THEROCK_BUILD/.vision-bill-llvm"
  record_size llvm "$THEROCK_BUILD"
}

finish() {
  local TIMEFORMAT
  require_checkpoint llvm
  # The Dockerfile first copies the finish recipe after LLVM has been staged.
  source "$RECIPE/finish_therock.sh"
  record_size finish "$THEROCK_BUILD"
  record_size finish "$SDK"
}

phase="${1:-all}"
case "$phase" in
  all|prepare|llvm|finish) ;;
  *) echo "Usage: $0 [all|prepare|llvm|finish]" >&2; exit 1 ;;
esac
if [[ "$phase" == all || "$phase" == prepare ]]; then
  TIMEFORMAT='ocr_rocm_phase phase=therock_prepare elapsed_s=%3R'
  time (prepare)
fi
if [[ "$phase" == all || "$phase" == llvm ]]; then build_llvm; fi
if [[ "$phase" == all || "$phase" == finish ]]; then
  TIMEFORMAT='ocr_rocm_phase phase=therock_finish elapsed_s=%3R'
  time (finish)
fi
