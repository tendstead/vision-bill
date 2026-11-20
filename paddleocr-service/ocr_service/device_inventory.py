"""Driver metadata only: no framework imports, contexts, models or compute probes."""

import ctypes
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Literal

from .profile import content_identity, file_digest
from .targets import validate_profile


class DeviceIdentityUnavailable(RuntimeError):
    reason_code = "native_device_identity_unavailable"


def _cuda_identity(index: int) -> dict[str, Any]:
    driver = ctypes.CDLL("libcuda.so.1")

    def call(name: str, types: list[Any], *args: Any) -> None:
        function = getattr(driver, name)
        function.argtypes = types
        function.restype = ctypes.c_int
        if function(*args) != 0:
            raise ValueError(f"CUDA driver metadata call failed: {name}")

    integer = ctypes.c_int
    pointer = ctypes.POINTER(integer)
    device, version, major, minor = (integer() for _ in range(4))
    call("cuInit", [ctypes.c_uint], 0)
    call("cuDeviceGet", [pointer, integer], ctypes.byref(device), index)
    call("cuDriverGetVersion", [pointer], ctypes.byref(version))
    name, bus, uuid = (
        ctypes.create_string_buffer(256),
        ctypes.create_string_buffer(32),
        ctypes.create_string_buffer(16),
    )
    call("cuDeviceGetName", [ctypes.c_void_p, integer, integer], name, len(name), device)
    call("cuDeviceGetPCIBusId", [ctypes.c_void_p, integer, integer], bus, len(bus), device)
    call("cuDeviceGetUuid_v2", [ctypes.c_void_p, integer], uuid, device)
    call(
        "cuDeviceComputeCapability",
        [pointer, pointer, integer],
        ctypes.byref(major),
        ctypes.byref(minor),
        device,
    )
    return {
        "accelerator": "cuda",
        "gpu_index": index,
        "name": name.value.decode(),
        "pci_bus_id": bus.value.decode(),
        "uuid": uuid.raw.hex(),
        "driver_version": version.value,
        "compute_capability": [major.value, minor.value],
    }


def inspect_device_identity(accelerator: Literal["cuda", "rocm"], gpu_index: int) -> dict[str, Any]:
    """Fail closed when actual driver/device metadata cannot be established."""
    try:
        if (
            accelerator not in {"cuda", "rocm"}
            or type(gpu_index) is not int
            or not 0 <= gpu_index <= 1024
        ):
            raise ValueError("Invalid native device selection")
        if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
            raise ValueError("Architecture spoofing is forbidden")
        if accelerator == "cuda":
            result = _cuda_identity(gpu_index)
        else:
            completed = subprocess.run(
                ["/opt/ocr/bin/hip-device-info", str(gpu_index)],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            if len(completed.stdout) > 8192:
                raise ValueError("Oversized HIP metadata")
            result = json.loads(completed.stdout)
        keys = {
            "accelerator",
            "gpu_index",
            "name",
            "uuid",
            "pci_bus_id",
            "driver_version",
            "compute_capability" if accelerator == "cuda" else "gfx_target",
        }
        if (
            not isinstance(result, dict)
            or set(result) != keys
            or result["accelerator"] != accelerator
            or type(result["gpu_index"]) is not int
            or result["gpu_index"] != gpu_index
        ):
            raise ValueError("Malformed device identity")
        if any(
            not isinstance(result[key], str)
            or not result[key].strip()
            or len(result[key]) > 256
            or any(ord(char) < 32 for char in result[key])
            for key in ("name", "uuid", "pci_bus_id")
        ):
            raise ValueError("Incomplete device identity")
        if type(result["driver_version"]) is not int or result["driver_version"] <= 0:
            raise ValueError("Missing driver version")
        if (
            not re.fullmatch(r"[0-9a-f]{32}", result["uuid"])
            or result["uuid"] == "0" * 32
            or not re.fullmatch(
                r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", result["pci_bus_id"]
            )
        ):
            raise ValueError("Malformed device UUID or PCI bus identity")
        if accelerator == "rocm":
            # The pinned ROCr UUID suffix is sixteen ASCII hex digits. When the
            # hardware ID is unavailable, HIP instead returns XX padded with zeros.
            hip_uuid = bytes.fromhex(result["uuid"])
            if not re.fullmatch(rb"[0-9a-fA-F]{16}", hip_uuid) or hip_uuid == b"0" * 16:
                raise ValueError("Unavailable or malformed HIP UUID")
            if not isinstance(result["gfx_target"], str) or not re.fullmatch(
                r"gfx[0-9a-f]+", result["gfx_target"]
            ):
                raise ValueError("Malformed HIP target")
        elif (
            not isinstance(result["compute_capability"], list)
            or len(result["compute_capability"]) != 2
            or any(type(value) is not int or value < 0 for value in result["compute_capability"])
        ):
            raise ValueError("Malformed CUDA capability")
        return result
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError) as exc:
        raise DeviceIdentityUnavailable(
            "Cannot verify native device metadata; explicit maintenance on a verifiable device is required"
        ) from exc


def execution_target(paddle: Any, device: dict[str, Any], accelerator: str) -> dict[str, Any]:
    """Bind the loader to the selected driver, build and installed HIP kernels without compute."""
    result = dict(device)
    result["paddle_build"] = str(paddle.version.full_version)
    if accelerator == "rocm":
        target = result["gfx_target"]
        kernels = json.loads(Path("/opt/ocr/build-evidence/kernels.json").read_text())
        profile = validate_profile(kernels.get("target_profile"))
        if target not in profile["candidate_targets"]:
            raise ValueError("Actual HIP target is outside the candidate target set")
        for library in ("paddle", "rocblas", "miopen"):
            if not kernels[library].get("files"):
                raise ValueError("Required kernel/library inventory is empty")
            if (
                kernels[library].get("offline_status") != "unverified"
                and target not in kernels[library]["targets"]
            ):
                raise ValueError("Required device kernels are missing")
            for filename, digest in kernels[library]["files"].items():
                if file_digest(Path(filename)) != digest:
                    raise ValueError("Required kernel inventory changed")
        result["gfx_target"] = target
        result["kernel_inventory_sha256"] = content_identity(kernels)
    return result
