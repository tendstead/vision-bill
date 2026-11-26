"""Read local CUDA driver metadata without loading Paddle or creating a CUDA context."""

import ctypes
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from uuid import UUID


def docker_device_uuid(selector: str) -> str:
    """Docker numeric selectors use NVIDIA's device indexing, not CUDA ordinal order."""
    if selector.startswith("GPU-"):
        return selector.lower()
    smi = shutil.which("nvidia-smi")
    wsl_smi = Path("/usr/lib/wsl/lib/nvidia-smi")
    if not smi and wsl_smi.is_file():
        smi = str(wsl_smi)
    if smi:
        try:
            result = subprocess.run(
                [smi, "-i", selector, "--query-gpu=uuid", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            identity = result.stdout.strip().lower()
            if re.fullmatch(r"gpu-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", identity):
                return identity
        except (OSError, subprocess.SubprocessError):
            pass
    return ""


def driver_inventory() -> list[tuple[str, int, int, int]]:
    """Return UUID, compute major/minor and the driver API version for visible devices."""
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError:
        driver = ctypes.CDLL("/usr/lib/wsl/lib/libcuda.so.1")

    def call(name: str, *args: object) -> None:
        if getattr(driver, name)(*args):
            raise OSError(f"CUDA driver metadata unavailable: {name}")

    count, version = ctypes.c_int(), ctypes.c_int()
    call("cuInit", 0)
    call("cuDeviceGetCount", ctypes.byref(count))
    call("cuDriverGetVersion", ctypes.byref(version))
    devices = []
    for index in range(count.value):
        device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        uuid = ctypes.create_string_buffer(16)
        call("cuDeviceGet", ctypes.byref(device), index)
        call("cuDeviceGetUuid_v2", uuid, device)
        call("cuDeviceComputeCapability", ctypes.byref(major), ctypes.byref(minor), device)
        devices.append((f"gpu-{UUID(bytes=uuid.raw)}", major.value, minor.value, version.value))
    return devices


def preflight(selector: str, channel: str) -> str:
    expected = docker_device_uuid(selector)
    try:
        devices = driver_inventory()
    except (AttributeError, OSError):
        devices = []
    matches = [item for item in devices if expected and item[0].startswith(expected)]
    if not expected and len(devices) > 1 and "microsoft" in platform.release().lower():
        raise ValueError(
            "Cannot map the Docker index on multi-GPU WSL; set OCR_CUDA_DEVICE_ID to a GPU UUID"
        )
    if len(matches) > 1:
        raise ValueError("Ambiguous CUDA GPU UUID prefix; use the full GPU UUID")
    if (
        not expected
        and selector == "0"
        and len(devices) == 1
        and "CUDA_VISIBLE_DEVICES" not in os.environ
        and "NVIDIA_VISIBLE_DEVICES" not in os.environ
    ):
        matches = devices
    if len(matches) != 1:
        print(
            "CUDA host identity unavailable; verifying the selected device in Docker.", flush=True
        )
        return expected
    uuid, major, minor, version = matches[0]
    if (major, minor) < (7, 0):
        raise ValueError(f"Selected CUDA GPU is SM {major}.{minor}; OCR requires SM 7.0 or newer")
    if version < 12000:
        raise ValueError(f"Selected CUDA driver API {version} cannot run the {channel} wheel")
    print(
        f"CUDA preflight: {uuid}, SM {major}.{minor}, driver API {version}, {channel}.", flush=True
    )
    return uuid
