# Sourced by build_therock.sh for the SDK build after LLVM has been staged.
# Docker commits these stages separately. The standalone build follows the
# same target graph, and verifies existing target/transport and patch markers.
for stage in rccl miopen-configure miopen; do
  bash "$RECIPE/checkpoint_therock.sh" "$stage"
done

# TheRock launches separate cmake --build processes for its subprojects.
TIMEFORMAT='ocr_rocm_phase phase=sdk_compile elapsed_s=%3R'
time CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS" cmake --build "$THEROCK_BUILD" --target dist-rocm
cmake --install "$THEROCK_BUILD" --component rocm --prefix "$SDK"
if [[ "$transport" != kfd ]]; then
  sha256sum "$SDK/lib/librocdxg.so.1" > "$EVIDENCE/wsl-bridge.sha256"
fi
cp "$THEROCK_BUILD/CMakeCache.txt" "$EVIDENCE/therock-CMakeCache.txt"
# Legacy FindHIP + Linux linker helper from the SAME patched source pin.
HIP_COMMON="$THEROCK_SRC/rocm-systems/projects/hip"
if [[ ! -f "$SDK/lib/cmake/hip/FindHIP.cmake" ]]; then
  mkdir -p "$SDK/lib/cmake/hip"
  cp "$HIP_COMMON/cmake/FindHIP.cmake" "$SDK/lib/cmake/hip/"
  cp -R "$HIP_COMMON/cmake/FindHIP" "$SDK/lib/cmake/hip/"
fi
if [[ ! -x "$SDK/bin/hipcc_cmake_linker_helper" ]]; then
  install -m 755 "$HIP_COMMON/bin/hipcc_cmake_linker_helper" "$SDK/bin/"
fi
for helper in "$SDK/bin/hipcc" "$SDK/bin/hipconfig" "$SDK/bin/hipcc_cmake_linker_helper" \
  "$SDK/lib/cmake/hip/FindHIP.cmake" "$SDK/lib/cmake/hip/FindHIP/run_hipcc.cmake" \
  "$SDK/lib/cmake/hip/FindHIP/run_make2cmake.cmake"; do
  test -f "$helper"
  sha256sum "$helper" >> "$EVIDENCE/hip-helpers.sha256"
done
