import pytest

from scripts import ocr_cuda_probe as probe

FIRST = "gpu-11111111-1111-1111-1111-111111111111"
SECOND = "gpu-22222222-2222-2222-2222-222222222222"


def test_docker_index_uses_nvml_uuid_independent_of_cuda_order(monkeypatch):
    monkeypatch.setattr(probe, "docker_device_uuid", lambda selector: SECOND)
    monkeypatch.setattr(
        probe, "driver_inventory", lambda: [(FIRST, 6, 1, 12090), (SECOND, 7, 0, 12060)]
    )
    assert probe.preflight("0", "cu129") == SECOND


@pytest.mark.parametrize(("major", "minor", "driver"), [(6, 1, 12090), (8, 6, 11080)])
def test_selected_incompatible_device_fails_preflight(monkeypatch, major, minor, driver):
    monkeypatch.setattr(probe, "driver_inventory", lambda: [(SECOND, major, minor, driver)])
    with pytest.raises(ValueError, match="Selected CUDA"):
        probe.preflight(SECOND.upper(), "cu126")


def test_missing_host_library_defers_explicit_identity_to_container(monkeypatch):
    def unavailable():
        raise OSError("missing host driver library")

    monkeypatch.setattr(probe, "driver_inventory", unavailable)
    assert probe.preflight(SECOND.upper(), "cu129") == SECOND


def test_numeric_selection_never_guesses_cuda_ordinal_on_multi_gpu(monkeypatch):
    monkeypatch.setattr(probe.platform, "release", lambda: "Linux")
    monkeypatch.setattr(probe, "docker_device_uuid", lambda selector: "")
    monkeypatch.setattr(
        probe, "driver_inventory", lambda: [(FIRST, 6, 1, 11080), (SECOND, 12, 0, 12090)]
    )
    assert probe.preflight("1", "cu129") == ""


def test_unmapped_wsl_index_requires_explicit_uuid_on_multi_gpu(monkeypatch):
    monkeypatch.setattr(probe.platform, "release", lambda: "microsoft-standard-WSL2")
    monkeypatch.setattr(probe, "docker_device_uuid", lambda selector: "")
    monkeypatch.setattr(
        probe, "driver_inventory", lambda: [(FIRST, 8, 6, 12090), (SECOND, 12, 0, 12090)]
    )
    with pytest.raises(ValueError, match="OCR_CUDA_DEVICE_ID"):
        probe.preflight("1", "cu129")
