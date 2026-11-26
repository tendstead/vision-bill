"""Exercise the native build guards on generated plans without an AMD toolchain."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[1] / "paddleocr-service/build"
CORE = ("comm-libs/rccl", "math-libs/BLAS/rocBLAS", "ml-libs/MIOpen")
GPU_KEYS = ("AMDGPU_TARGETS", "GPU_TARGETS", "CMAKE_HIP_ARCHITECTURES")
RADEON = (
    "gfx1030",
    "gfx1031",
    "gfx1032",
    "gfx1034",
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1200",
    "gfx1201",
)


def guard(script, root, selected):
    profile = root / "targets.json"
    profile.write_text(json.dumps({"schema": 1, "candidate_targets": selected}))
    return subprocess.run(
        [sys.executable, str(BUILD / script), str(root), str(profile)],
        capture_output=True,
        text=True,
        check=False,
    )


def toolchain(root, name, targets):
    path = root / name / "_toolchain.cmake"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f'set({k} {targets} CACHE STRING "From super-project" FORCE)\n' for k in GPU_KEYS)
    )
    return path


def superbuild(root, selected):
    targets = ";".join(selected)
    (root / "CMakeCache.txt").write_text(
        "".join(
            f"THEROCK_{scope}AMDGPU_TARGETS:STRING={targets}\n"
            f"THEROCK_{scope}AMDGPU_FAMILIES:STRING=\n"
            for scope in ("", "DIST_", "TEST_")
        )
    )
    for name in CORE:
        toolchain(root, name, targets)


def test_generated_math_project_subsets_and_host_only_toolchains(tmp_path):
    superbuild(tmp_path, RADEON)
    toolchain(tmp_path, "math-libs/BLAS/hipBLASLt", "gfx1100;gfx1200")
    toolchain(tmp_path, "math-libs/hipTensor", "")
    toolchain(tmp_path, "profiler/rocprofiler-compute", "gfx942")
    result = guard("verify_therock_targets.py", tmp_path, RADEON)
    assert result.returncode == 0, result.stderr
    assert "gfx1034" in result.stdout
    assert "hipBLASLt" in result.stdout


@pytest.mark.parametrize("project", [*CORE, "comm-libs/rocshmem", "math-libs/BLAS/hipBLASLt"])
def test_generated_project_cannot_broaden_selection(tmp_path, project):
    superbuild(tmp_path, ["gfx1030"])
    path = toolchain(tmp_path, project, "gfx1030")
    path.write_text(path.read_text().replace("set(GPU_TARGETS gfx1030 ", "set(GPU_TARGETS gfx942 "))
    result = guard("verify_therock_targets.py", tmp_path, ["gfx1030"])
    assert result.returncode != 0
    assert project in result.stderr and "gfx942" in result.stderr


@pytest.mark.parametrize(("selected", "passes"), [(["gfx1030"], False), (["gfx1250"], True)])
def test_rccl_translation_requires_selected_gfx1250_payload(tmp_path, selected, passes):
    superbuild(tmp_path, selected)
    # The pinned translation step disables generic GPU target propagation but
    # still unbundles and pretranslates gfx1250 from the RCCL artifact.
    path = tmp_path / "comm-libs/rccl-translate/_toolchain.cmake"
    path.parent.mkdir()
    path.write_text("set(CMAKE_EXPORT_COMPILE_COMMANDS ON)\n")
    result = guard("verify_therock_targets.py", tmp_path, selected)
    assert (result.returncode == 0) is passes
    if not passes:
        assert "rccl-translate" in result.stderr and "gfx1250" in result.stderr


def test_distinct_optional_target_lists_cannot_be_reported_as_one(tmp_path):
    selected = ["gfx1030", "gfx1100"]
    superbuild(tmp_path, selected)
    path = toolchain(tmp_path, "math-libs/BLAS/hipBLASLt", "gfx1100")
    path.write_text(
        path.read_text().replace("set(GPU_TARGETS gfx1100 ", "set(GPU_TARGETS gfx1030 ")
    )
    result = guard("verify_therock_targets.py", tmp_path, selected)
    assert result.returncode != 0 and "hipBLASLt" in result.stderr


def test_missing_test_selector_is_rejected_before_llvm(tmp_path):
    superbuild(tmp_path, ["gfx1030"])
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text(
        cache.read_text().replace(
            "TEST_AMDGPU_TARGETS:STRING=gfx1030", "TEST_AMDGPU_TARGETS:STRING="
        )
    )
    result = guard("verify_therock_targets.py", tmp_path, ["gfx1030"])
    assert result.returncode != 0
    assert "THEROCK_TEST_AMDGPU_TARGETS" in result.stderr


@pytest.fixture
def rccl_plan(tmp_path):
    cmake = shutil.which("cmake")
    if not cmake or not shutil.which("ninja"):
        pytest.skip("RCCL plan fixture requires CMake and Ninja")

    def configure(compile_targets, link_targets, extra=""):
        # These custom kernel commands never enter CMake's C/C++ compile_commands.json.
        # CMake generates their Ninja rules. Running any of them would fail the build.
        commands = [
            f"COMMAND /toolchain/rccl-device-compile --compile --arch={arch} -o kernel.o kernel.cpp"
            for arch in compile_targets
        ] + [
            f"COMMAND /toolchain/rccl-device-compile --link --arch={arch} @link_objects.rsp"
            for arch in link_targets
        ]
        (tmp_path / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.16)\nproject(rccl NONE)\n"
            "add_custom_target(rccl ALL\n"
            "COMMAND /toolchain/amdclang++ --offload-host-only -c host_gfx942.cpp\n"
            + "\n".join(commands)
            + "\n"
            + extra
            + "\nVERBATIM)\n"
        )
        subprocess.run(
            [
                cmake,
                "-S",
                str(tmp_path),
                "-B",
                str(tmp_path / "comm-libs/rccl/build"),
                "-G",
                "Ninja",
            ],
            check=True,
            capture_output=True,
        )
        return tmp_path

    return configure


@pytest.mark.parametrize("selected", [("gfx1030",), ("gfx1031", "gfx1201"), RADEON])
def test_generated_kernel_plan_reports_selected_targets(rccl_plan, selected):
    root = rccl_plan(
        selected, selected, f"COMMAND /toolchain/amdclang++ --offload-arch={selected[0]} -c gpu.hip"
    )
    result = guard("verify_rccl_targets.py", root, selected)
    assert result.returncode == 0, result.stderr
    assert ";".join(selected) in result.stdout and "gfx942" not in result.stdout


@pytest.mark.parametrize(
    ("compile_targets", "link_targets", "extra", "unexpected"),
    [
        (["gfx1030", "gfx942"], ["gfx1030"], "", "gfx942"),
        (["gfx1030"], ["gfx1030", "gfx1201"], "", "gfx1201"),
        (
            ["gfx1030"],
            ["gfx1030"],
            "COMMAND /toolchain/amdclang++ --offload-arch=gfx950 -c gpu.hip",
            "gfx950",
        ),
        (
            ["gfx1030"],
            ["gfx1030"],
            "COMMAND /toolchain/amdclang++ --offload-arch=native -c gpu.hip",
            "native",
        ),
    ],
)
def test_extra_targets_hidden_in_custom_ninja_rules(
    rccl_plan, compile_targets, link_targets, extra, unexpected
):
    root = rccl_plan(compile_targets, link_targets, extra)
    result = guard("verify_rccl_targets.py", root, ["gfx1030"])
    assert result.returncode != 0
    assert unexpected in result.stderr


def test_missing_kernel_compile_cannot_be_masked_by_host_or_link_rules(rccl_plan):
    root = rccl_plan(
        ["gfx1030"],
        ["gfx1030", "gfx1100"],
        "COMMAND /toolchain/amdclang++ --offload-arch=gfx1100 --offload-host-only -c host.cpp",
    )
    result = guard("verify_rccl_targets.py", root, ["gfx1030", "gfx1100"])
    assert result.returncode != 0
    assert "gfx1100" in result.stderr and "compile" in result.stderr


@pytest.mark.parametrize("script", ["verify_therock_targets.py", "verify_rccl_targets.py"])
def test_guard_rejects_invalid_profile_before_reading_plan(tmp_path, script):
    result = guard(script, tmp_path, ["native"])
    assert result.returncode != 0 and "Invalid candidate target profile" in result.stderr
