"""Exercise selected-device driver calls with CPU doubles."""

import struct
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("backend", ["cuda", "rocm"])
@pytest.mark.parametrize("failed", [False, True])
def test_preload_probe_reads_selected_device_and_restores_context(
    monkeypatch, capsysbinary, backend, failed
):
    from ocr_service import gpu_memory

    driver = Mock()
    initial_device = 1
    selected_device = 4
    cuda_context = 124
    current_device = initial_device
    current_context = 42

    def set_output(out, value):
        out._obj.value = value
        return 0

    def select_hip_device(index):
        nonlocal current_device
        current_device = index
        return 0

    def push_cuda_context(context):
        nonlocal current_context
        current_context = context.value
        return 0

    def pop_cuda_context(context):
        nonlocal current_context
        set_output(context, current_context)
        current_context = 42
        return 0

    def read(free, total):
        if backend == "cuda":
            assert current_context == cuda_context
        else:
            assert current_device == selected_device
        if failed:
            return 2
        set_output(free, 123)
        return set_output(total, 456)

    def cuda_device(out, ordinal):
        assert ordinal == selected_device
        return set_output(out, 9)

    def retain(context, native_device):
        assert native_device.value == 9
        return set_output(context, cuda_context)

    driver.cuInit.return_value = 0
    driver.cuDeviceGet.side_effect = cuda_device
    driver.cuDevicePrimaryCtxRetain.side_effect = retain
    driver.cuCtxPushCurrent_v2.side_effect = push_cuda_context
    driver.cuCtxPopCurrent_v2.side_effect = pop_cuda_context
    driver.cuDevicePrimaryCtxRelease_v2.return_value = 0
    driver.cuMemGetInfo_v2.side_effect = read
    driver.hipGetDevice.side_effect = lambda out: set_output(out, current_device)
    driver.hipSetDevice.side_effect = select_hip_device
    driver.hipMemGetInfo.side_effect = read

    def load_library(library):
        assert library == {"cuda": "libcuda.so.1", "rocm": "libamdhip64.so"}[backend]
        return driver

    monkeypatch.setattr(gpu_memory.ctypes, "CDLL", load_library)
    monkeypatch.setattr(gpu_memory, "_arm_parent_death", lambda parent: parent == 42)
    status = gpu_memory._run_probe([backend, str(selected_device), "42"])
    output = capsysbinary.readouterr().out
    if failed:
        assert status == 1
        assert output == b""
    else:
        assert status == 0
        assert struct.unpack("!QQ", output) == (123, 456)
    assert current_device == initial_device
    assert current_context == 42
    if backend == "cuda":
        assert driver.cuDevicePrimaryCtxRelease_v2.call_args.args[0].value == 9


@pytest.mark.parametrize("backend", ["cuda", "rocm"])
@pytest.mark.parametrize("outcome", ["ok", "timeout", "failed", "malformed", "invalid"])
def test_preload_query_never_touches_the_inference_process_driver(monkeypatch, backend, outcome):
    import os
    import subprocess
    import sys

    from ocr_service import gpu_memory

    def forbidden_driver(*_args, **_kwargs):
        pytest.fail("The inference process opened a CUDA or HIP driver")

    def run(command, **kwargs):
        assert command == [
            sys.executable, "-m", "ocr_service.gpu_memory", backend, "4", str(os.getpid())
        ]
        assert kwargs["timeout"] > 0
        assert kwargs["timeout"] <= 10
        assert kwargs["check"]
        assert kwargs["stderr"] == subprocess.DEVNULL
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if outcome == "failed":
            raise subprocess.CalledProcessError(1, command)
        output = {
            "ok": struct.pack("!QQ", 123, 456),
            "malformed": b"bad response",
            "invalid": struct.pack("!QQ", 456, 123),
        }[outcome]
        return subprocess.CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(gpu_memory.ctypes, "CDLL", forbidden_driver)
    monkeypatch.setattr(subprocess, "run", run)
    reading = gpu_memory.probe_gpu_memory(backend, 4)
    if outcome == "ok":
        assert reading.model_dump() == {
            "accelerator": backend, "gpu_index": 4, "free_bytes": 123, "total_bytes": 456
        }
    else:
        assert reading is None


def test_probe_child_refuses_an_unexpected_parent():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "ocr_service.gpu_memory", "cuda", "0", "0"],
        capture_output=True,
        check=False,
        timeout=5,
    )
    assert result.returncode != 0
    assert result.stdout == b""
