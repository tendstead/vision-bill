"""Candidate propagation only: fake libraries/devices do not qualify hardware."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_ocr_runtime_builds import SERVICE, ROOT, load_build_module


@pytest.mark.parametrize("transport", ["kfd", "wsl", "dual", None])
@pytest.mark.parametrize("wsl", [True, False])
def test_rocm_binding_checks_image_transport_and_preserves_portable_images(transport, wsl):
    from ocr_service.targets import verify_rocm_image_target

    inventory = {"target_profile": {"schema": 1, "candidate_targets": ["gfx1030"]}}
    if transport is not None:
        inventory["transport"] = transport
    device = {"gfx_target": "gfx1030"}
    if wsl:
        device["transport"] = "wsl-dxg"
    if transport in (None, "dual", "wsl" if wsl else "kfd"):
        verify_rocm_image_target(inventory, device)
    else:
        with pytest.raises(ValueError, match="transport"):
            verify_rocm_image_target(inventory, device)
    with pytest.raises(ValueError, match="target"):
        verify_rocm_image_target(inventory, dict(device, gfx_target="gfx1100"))


def test_candidate_parser_retains_defaults_and_rejects_injection():
    from ocr_service.targets import DEFAULT_TARGETS, parse_targets, validate_profile
    assert parse_targets('') == list(DEFAULT_TARGETS) == [
        'gfx1030', 'gfx1031', 'gfx1032', 'gfx1034', 'gfx1100', 'gfx1101', 'gfx1102',
        'gfx1200', 'gfx1201',
    ]
    assert parse_targets('gfx1100;gfx1030') == ['gfx1030', 'gfx1100']
    for invalid in ['gfx1100;', 'gfx1100;gfx1100', 'native', 'all', 'gfx1100:xnack+', 'gfx1100 -foo', 'gfx1100;set(X)', '../gfx1100']:
        with pytest.raises(ValueError):
            parse_targets(invalid)
    assert validate_profile({'schema': 1, 'candidate_targets': ['gfx1100']}) == {'schema': 1, 'candidate_targets': ['gfx1100']}
    for invalid in [{}, {'schema': 2, 'candidate_targets': ['gfx1100']}, {'schema': 1, 'candidate_targets': []}, {'schema': True, 'candidate_targets': ['gfx1100']}, {'schema': 1, 'candidate_targets': ['gfx1100'], 'extra': 1}]:
        with pytest.raises(ValueError):
            validate_profile(invalid)


def test_custom_compile_and_library_inventory_use_same_selection(tmp_path, monkeypatch):
    module = load_build_module('inspect_kernels')
    commands = tmp_path / 'compile.log'
    commands.write_text('hipcc -c x --offload-arch=gfx1100\n')
    module.validate_compile_targets(commands, {'gfx1100'})
    for flags in ['', '--offload-arch=gfx1030', '--offload-arch=gfx1100 --offload-arch=gfx1030']:
        commands.write_text('hipcc -c x ' + flags)
        with pytest.raises(ValueError):
            module.validate_compile_targets(commands, {'gfx1100'})
    lib = tmp_path / 'lib.so'
    lib.write_bytes(b'\x7fELFfake')
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='arch gfx1100\n'))
    assert module.inventory([lib], tmp_path / 'llvm-objdump', {'gfx1100'})['offline_status'] == 'verified'
    with pytest.raises(ValueError, match='missing'):
        module.inventory([lib], tmp_path / 'llvm-objdump', {'gfx1030'})
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=''))
    assert module.inventory([lib], tmp_path / 'llvm-objdump', {'gfx1100'})['offline_status'] == 'unverified'


def test_newer_observed_target_requires_selected_inventory_and_changes_seal(tmp_path, monkeypatch):
    from ocr_service import qualify
    from ocr_service.profile import file_digest
    lib = tmp_path / 'library'
    lib.write_bytes(b'synthetic-library')
    profile = {'schema': 1, 'candidate_targets': ['gfx1100']}
    kernels = {'target_profile': profile, **{name: {'files': {str(lib): file_digest(lib)}, 'targets': ['gfx1100'], 'offline_status': 'verified'} for name in ('paddle', 'rocblas', 'miopen')}}
    real_read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **k: json.dumps(kernels) if str(path) == '/opt/ocr/build-evidence/kernels.json' else real_read(path, *a, **k))
    monkeypatch.setattr(qualify, 'inspect_device_identity', lambda *a: {'gfx_target': 'gfx1100', 'name': 'candidate', 'gpu_index': 0})
    probes = []
    monkeypatch.setattr(qualify, 'compute_probe', lambda *a: probes.append('probe') or {'synthetic': True})
    paddle = SimpleNamespace(device=SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda i: 'candidate')), version=SimpleNamespace(full_version='pinned'))
    runtime = SimpleNamespace(accelerator='rocm', gpu_index=0)
    first = qualify.device_identity(paddle, runtime)
    assert first['gfx_target'] == 'gfx1100' and probes == ['probe']
    profile['candidate_targets'] = ['gfx1030', 'gfx1100']
    second = qualify.device_identity(paddle, runtime, compute=False)
    assert first['kernel_inventory_sha256'] != second['kernel_inventory_sha256']
    profile['candidate_targets'] = ['gfx1030']
    with pytest.raises(ValueError, match='candidate'):
        qualify.device_identity(paddle, runtime)
    kernels.pop('target_profile')
    with pytest.raises(ValueError, match='profile'):
        qualify.device_identity(paddle, runtime)


def test_executable_candidate_interface_reaches_both_compilers(tmp_path, monkeypatch):
    import os
    import subprocess
    import yaml
    overlay = yaml.safe_load((ROOT / 'docker-compose.rocm.yml').read_text())
    assert 'AMDGPU_TARGETS' in overlay['services']['paddleocr-service']['build']['args']
    docker = (SERVICE / 'Dockerfile.rocm').read_text()
    assert 'ARG AMDGPU_TARGETS' in docker and 'AMDGPU_TARGETS="$AMDGPU_TARGETS"' in docker
    assert '--target-profile /opt/ocr/build-evidence/targets.json' in docker
    script = (SERVICE / 'build/build_rocm.sh').read_text()
    sdk = (SERVICE / 'build/build_therock.sh').read_text()
    assert '"-DTHEROCK_AMDGPU_TARGETS=$TARGETS"' in sdk
    assert '"-DTHEROCK_DIST_AMDGPU_TARGETS=$TARGETS"' in sdk
    assert 'export VISION_BILL_AMDGPU_TARGETS="$TARGETS"' in script
    patch = (SERVICE / 'build/hip-rdna2-v1.patch').read_text()
    added = '\n'.join(line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++'))
    start = added.index('set(VISION_BILL_AMDGPU_TARGETS')
    end = added.index('endforeach()', start) + len('endforeach()')
    program = tmp_path / 'flags.cmake'
    program.write_text(added[start:end] + '\nfile(WRITE "' + str(tmp_path / 'flags') + '" "${HIP_HCC_FLAGS}\\n${HIP_CLANG_FLAGS}")\n')
    environment = {**os.environ, 'VISION_BILL_AMDGPU_TARGETS': 'gfx1100'}
    subprocess.run(['cmake', '-P', str(program)], env=environment, check=True, capture_output=True)
    assert (tmp_path / 'flags').read_text().splitlines() == ['--offload-arch=gfx1100', '--offload-arch=gfx1100']
    for bad in ['', 'gfx1100;--bad']:
        result = subprocess.run(['cmake', '-P', str(program)], env={**environment, 'VISION_BILL_AMDGPU_TARGETS': bad}, capture_output=True)
        assert result.returncode != 0
