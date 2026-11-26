"""Check the real pinned BLAS project declarations without building ROCm.

Set VISION_BILL_TEST_THEROCK to a local TheRock checkout containing the locked
commit. The fixture reads the committed sources and applies our patch in tmp.
"""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIN = "b324869fb3116353ac7eec5e6926aa3d452ac987"


@pytest.fixture(scope="module")
def blas_project(tmp_path_factory):
    upstream = os.getenv("VISION_BILL_TEST_THEROCK")
    if not upstream:
        pytest.skip("requires VISION_BILL_TEST_THEROCK pinned source checkout")
    project = tmp_path_factory.mktemp("therock-linalg")
    for name in (
        "cmake/therock_amdgpu_targets.cmake",
        "cmake/therock_subproject.cmake",
        "math-libs/BLAS/CMakeLists.txt",
    ):
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            subprocess.check_output(["git", "-C", upstream, "show", f"{PIN}:{name}"])
        )
    patch = ROOT / "paddleocr-service/build/therock-hipblaslt-rdna2.patch"
    subprocess.run(["git", "apply", patch], cwd=project, check=True)
    # Keep the actual architecture registry, filter, and complete BLAS declarations.
    # Capture their external-project boundary so this test needs no compiler or SDK.
    (project / "CMakeLists.txt").write_text("""
cmake_minimum_required(VERSION 3.25)
project(blas_plan NONE)
include(cmake/therock_amdgpu_targets.cmake)
include(cmake/therock_subproject.cmake)
function(therock_cmake_subproject_declare name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "BACKGROUND_BUILD"
    "COMPILER_TOOLCHAIN;EXTERNAL_SOURCE_DIR;BINARY_DIR"
    "DEFAULT_GPU_TARGETS;CMAKE_ARGS;CMAKE_INCLUDES;BUILD_DEPS;RUNTIME_DEPS")
  add_custom_target(${name})
  set_target_properties(${name} PROPERTIES THEROCK_AMDGPU_TARGETS "${THEROCK_AMDGPU_TARGETS}"
    THEROCK_DEFAULT_GPU_TARGETS "${arg_DEFAULT_GPU_TARGETS}")
  _therock_filter_project_gpu_targets(resolved ${name})
  file(WRITE "${CMAKE_BINARY_DIR}/${name}.targets" "${resolved}")
  file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/${name}.args"
    CONTENT "$<JOIN:${arg_CMAKE_ARGS},\n>")
endfunction()
foreach(fn IN ITEMS therock_cmake_subproject_dist_dir therock_cmake_subproject_glob_c_sources
    therock_cmake_subproject_provide_package therock_cmake_subproject_activate therock_provide_artifact)
  cmake_language(EVAL CODE "function(${fn})\nendfunction()")
endforeach()
set(THEROCK_ENABLE_SPARSE ON)
set(THEROCK_BUILD_TESTING OFF)
add_subdirectory(math-libs/BLAS)
""")
    return project


@pytest.mark.parametrize(
    ("selection", "blas_targets", "sparse_targets", "blas_host_only", "sparse_host_only"),
    [
        ("gfx1030", "gfx1030", "gfx1030", True, True),
        ("gfx1031", "gfx1031", "gfx1031", True, True),
        ("gfx1032", "gfx1032", "gfx1032", True, True),
        ("gfx1034", "gfx1034", "gfx1034", True, True),
        ("gfx1100", "gfx1100", "gfx1100", False, True),
        ("gfx1101", "gfx1101", "gfx1101", False, True),
        ("gfx1102", "gfx1102", "gfx1102", False, True),
        ("gfx1200", "gfx1200", "gfx1200", False, True),
        ("gfx1201", "gfx1201", "gfx1201", False, True),
        ("gfx1031;gfx1034", "gfx1031;gfx1034", "gfx1031;gfx1034", True, True),
        ("gfx1031;gfx1101", "gfx1101", "gfx1031;gfx1101", False, True),
        ("gfx1031;gfx942", "gfx942", "gfx942", False, False),
    ],
)
def test_blas_host_offload_and_tuned_libraries(
    blas_project,
    tmp_path,
    selection,
    blas_targets,
    sparse_targets,
    blas_host_only,
    sparse_host_only,
):
    subprocess.run(
        ["cmake", "-S", blas_project, "-B", tmp_path, f"-DTHEROCK_AMDGPU_TARGETS={selection}"],
        check=True,
    )
    assert (tmp_path / "hipBLASLt.targets").read_text() == blas_targets
    assert (tmp_path / "hipSPARSELt.targets").read_text() == sparse_targets
    for project, prefix, host_only in (
        ("hipBLASLt", "HIPBLASLT", blas_host_only),
        ("hipSPARSELt", "HIPSPARSELT", sparse_host_only),
    ):
        args = (tmp_path / f"{project}.args").read_text().splitlines()
        assert (f"-D{prefix}_ENABLE_DEVICE=OFF" in args) is host_only
        assert (f"-D{prefix}_ENABLE_CLIENT=OFF" in args) is host_only
        if host_only:
            assert "-DHIPBLASLT_BUNDLE_PYTHON_DEPS=OFF" in args
            assert "-DHIPBLASLT_ENABLE_THEROCK=ON" in args
            if project == "hipBLASLt":
                assert "-DHIPBLASLT_ENABLE_ROCROLLER=OFF" in args
    assert "-DBUILD_WITH_HIPBLASLT=ON" in (tmp_path / "rocBLAS.args").read_text().splitlines()
