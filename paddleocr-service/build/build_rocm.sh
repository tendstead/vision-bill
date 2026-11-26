#!/usr/bin/env bash
# Unexecuted native qualification candidate; this script never declares GPU support.
set -euo pipefail
SOURCE_ROOT="${SOURCE_ROOT:-/opt/native-src}"
SDK="${SDK:-/opt/rocm}"
RECIPE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_JOBS="$(python "$RECIPE/build_jobs.py" "${BUILD_JOBS:-auto}")"
echo "Native build parallelism: $BUILD_JOBS jobs"
EVIDENCE=/opt/ocr/build-evidence
THEROCK_SRC="$SOURCE_ROOT/TheRock"
THEROCK_BUILD="$SOURCE_ROOT/therock-build"
PADDLE_SRC="$SOURCE_ROOT/Paddle"
# Fresh sources/caches only: a previous native build is not a reproducible input.
paths=("$PADDLE_SRC")
if [[ ${1:-} != --paddle-only ]]; then paths+=("$THEROCK_SRC" "$THEROCK_BUILD"); fi
for path in "${paths[@]}"; do
  if [[ -e "$path" || -L "$path" ]]; then
    echo "Native build path already exists: $path" >&2
    exit 1
  fi
done
if [[ -z "${AMDGPU_TARGETS:-}" ]]; then
  echo "Use make ocr-build-rocm, set OCR_AMDGPU_TARGETS for Compose, or pass --build-arg AMDGPU_TARGETS to Docker." >&2
  exit 1
fi
if [[ ${1:-} != --paddle-only ]]; then bash "$RECIPE/build_therock.sh"; fi
mkdir -p "$SOURCE_ROOT" "$EVIDENCE" /opt/ocr/bin /opt/wheels
TARGETS="$(python "$RECIPE/../ocr_service/targets.py" --targets "${AMDGPU_TARGETS:-}" --output "$EVIDENCE/targets.json")"
export VISION_BILL_AMDGPU_TARGETS="$TARGETS"

TIMEFORMAT='ocr_rocm_phase phase=paddle_sources elapsed_s=%3R'
time (
  git clone --filter=blob:none --no-checkout https://github.com/PaddlePaddle/Paddle.git "$PADDLE_SRC"
  git -C "$PADDLE_SRC" checkout --detach 7688495538f4d6c1893f084dd238a402e8f68ab6
  git -C "$PADDLE_SRC" submodule update --init --recursive --filter=blob:none --jobs "$BUILD_JOBS"
  (
    cd "$PADDLE_SRC"
    git apply --check "$RECIPE/hip-rdna2-v1.patch"
    git apply "$RECIPE/hip-rdna2-v1.patch"
  )
  sha256sum "$RECIPE/hip-rdna2-v1.patch" > "$EVIDENCE/compatibility-patch.sha256"
  python "$RECIPE/source_inventory.py" "$PADDLE_SRC" "$EVIDENCE/paddle-sources.json"
)
export ROCM_PATH="$SDK" HIP_PATH="$SDK" HIP_CLANG_PATH="$SDK/lib/llvm/bin"
export PATH="$SDK/bin:$SDK/lib/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$SDK/lib:$SDK/lib/llvm/lib:$SDK/lib/rocm_sysdeps/lib"
export CMAKE_PREFIX_PATH="$SDK" CMAKE_MODULE_PATH="$SDK/lib/cmake/hip"
export PADDLE_VERSION=3.3.1 WITH_ROCM=ON WITH_GPU=OFF WITH_PYTHON=ON
export WITH_TESTING=OFF WITH_DISTRIBUTE=OFF WITH_RCCL=ON
export WITH_CINN=OFF WITH_TENSORRT=OFF WITH_FLASHATTN=OFF
export WITH_MKL=OFF WITH_ONEDNN=OFF MAX_JOBS="$BUILD_JOBS"
export CMAKE_EXPORT_COMPILE_COMMANDS=ON CMAKE_VERBOSE_MAKEFILE=ON VERBOSE=1
TIMEFORMAT='ocr_rocm_phase phase=paddle_compile elapsed_s=%3R'
time (
  cd "$PADDLE_SRC"
  python setup.py bdist_wheel 2>&1 | tee "$EVIDENCE/paddle-build.log"
)
cp "$PADDLE_SRC"/dist/*.whl /opt/wheels/
sha256sum /opt/wheels/*.whl > "$EVIDENCE/native-wheels.sha256"
cp "$PADDLE_SRC/build/CMakeCache.txt" "$EVIDENCE/paddle-CMakeCache.txt"
if [[ -f "$PADDLE_SRC/build/compile_commands.json" ]]; then
  cp "$PADDLE_SRC/build/compile_commands.json" "$EVIDENCE/paddle-compile-commands.json"
fi
python "$RECIPE/source_inventory.py" "$PADDLE_SRC" "$EVIDENCE/paddle-built-sources.json"
"$SDK/bin/hipcc" "$RECIPE/hip-device-info.cpp" -o /opt/ocr/bin/hip-device-info
# ELF/linker resolution and real code-object inventories are separate gates.
ldd /opt/ocr/bin/hip-device-info > "$EVIDENCE/hip-probe.ldd"
if grep -q 'not found' "$EVIDENCE/hip-probe.ldd"; then exit 1; fi
# Runtime wheel install and kernel inventory run in Docker after the locked closure.
