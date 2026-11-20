"""Safe failure evidence crosses the isolated native stage without changing the gate."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("invalid", [False, True])
def test_static_collection_child_only_accepts_allowlisted_failure_record(tmp_path, monkeypatch, invalid):
    from ocr_service import qualify

    diagnostic = {
        "model_name": "PP-DocLayoutV3",
        "reason_code": "private tensor metadata" if invalid else "hook_metadata_invalid",
        "input_count": 14,
        "output_count": 10,
    }

    def wait(timeout):
        assert timeout == 600
        (tmp_path / "stage-output.json").write_text(json.dumps({
            "stage": "native_collection",
            "failure": "static_hook_collection_failed",
            "qualified": False,
            "static_hook_failure": diagnostic,
        }))
        return 43

    monkeypatch.setattr(
        qualify.subprocess, "Popen", lambda *a, **k: SimpleNamespace(wait=wait, args=["collect"])
    )
    if invalid:
        with pytest.raises(qualify.subprocess.CalledProcessError):
            qualify.run_stage("collect", {}, tmp_path)
    else:
        with pytest.raises(ValueError, match="static_hook_collection_failed") as failure:
            qualify.run_stage("collect", {}, tmp_path)
        assert failure.value.diagnostic.model_dump() == diagnostic


def test_full_qualification_carries_static_hook_failure_without_publishing(tmp_path, monkeypatch):
    from ocr_service import qualify
    from ocr_service.profile import MODEL_NAMES, RuntimeSettings

    runtime = RuntimeSettings(
        accelerator="cuda", gpu_index=0, model_dirs={name: tmp_path / name for name in MODEL_NAMES},
        cache_dir=tmp_path, qualification_manifest=tmp_path / "manifest.json", expected_packages={},
        expected_paddle_commit="1234567", image_identity="sha256:" + "a" * 64,
    )
    row = qualify.QualificationCase(kind="upright", path=Path("image"), sha256="a" * 64)
    diagnostic = {
        "model_name": "UVDoc",
        "reason_code": "hook_input_missing",
        "input_count": 0,
        "output_count": 3,
    }

    def fail(stage, payload, directory):
        if stage == "inspect":
            return {}
        raise qualify.StaticHookCollectionFailed(qualify.StaticHookFailure(**diagnostic))

    monkeypatch.setattr(qualify, "load_cases", lambda _path: [row])
    monkeypatch.setattr(qualify, "seal_dependencies", lambda **kwargs: ({}, {}))
    monkeypatch.setattr(qualify, "verify_dependencies", lambda *args, **kwargs: None)
    monkeypatch.setattr(qualify, "tree_identity", lambda _path: "model-hash")
    monkeypatch.setattr(qualify, "run_stage", fail)
    output = tmp_path / "qualification.json"
    report = qualify.qualify_runtime(runtime, tmp_path / "cases.toml", output)
    assert not report.qualified
    assert report.failure == "static_hook_collection_failed"
    assert json.loads(output.read_bytes())["static_hook_failure"] == diagnostic
    assert not output.with_suffix(".runtime.json").exists()
    assert not runtime.qualification_manifest.exists()


def test_native_collection_emits_the_failing_models_safe_diagnostic(tmp_path, monkeypatch):
    import sys
    from contextlib import nullcontext

    from ocr_service import qualify
    from ocr_service.profile import MODEL_NAMES, RuntimeSettings

    runtime = RuntimeSettings(
        accelerator="cuda", gpu_index=0, model_dirs={name: tmp_path / name for name in MODEL_NAMES},
        cache_dir=tmp_path, qualification_manifest=tmp_path / "manifest.json", expected_packages={},
        expected_paddle_commit="1234567", image_identity="sha256:" + "a" * 64,
    )

    class Predictor:
        def register_input_hook(self, callback):
            self.input_hook = callback

        def register_output_hook(self, callback):
            self.output_hook = callback

    models = {
        name: SimpleNamespace(runner=SimpleNamespace(
            predictor=Predictor(), _model_name=name, model_dir=runtime.model_dirs[name],
            model_file_prefix="graph",
            _config={
                "run_mode": "paddle_fp32", "device_type": "gpu", "device_id": 0,
                "enable_cinn": False,
            },
        ))
        for name in MODEL_NAMES[:3]
    }

    def predict(*_args, **_kwargs):
        for name, model in models.items():
            place = SimpleNamespace(
                is_cpu_place=lambda name=name: name == "UVDoc",
                is_gpu_place=lambda name=name: name != "UVDoc",
                gpu_device_id=lambda: 0,
            )
            tensor = SimpleNamespace(dtype="paddle.float32", place=place)
            model.runner.predictor.input_hook("private operation", "private parameter", tensor)
            model.runner.predictor.output_hook("private operation", "private result", tensor)
        yield {}

    native = SimpleNamespace(
        doc_preprocessor_pipeline=SimpleNamespace(
            doc_ori_classify_model=models[MODEL_NAMES[1]],
            doc_unwarping_model=models[MODEL_NAMES[2]],
        ),
        layout_det_model=models[MODEL_NAMES[0]],
        vl_rec_model=SimpleNamespace(infer=object()),
        predict=predict,
    )
    row = qualify.QualificationCase(kind="upright", path=Path("image"), sha256="a" * 64)
    payload = {
        "runtime": runtime.model_dump(mode="json"),
        "dependencies": {},
        "inventory": {
            name: {"graph_file": "graph.json", "parameters": [{"name": "private parameter"}]}
            for name in models
        },
        "cases": [row.model_dump(mode="json")],
    }
    incoming = tmp_path / "incoming.json"
    output = tmp_path / "output.json"
    incoming.write_text(json.dumps(payload))
    monkeypatch.setattr(qualify, "verify_dependencies", lambda *args, **kwargs: None)
    monkeypatch.setattr(qualify, "construct_native", lambda *a, **k: (native, None, SimpleNamespace))
    monkeypatch.setattr(qualify, "device_identity", lambda *args: {})
    monkeypatch.setattr(qualify, "audit_dynamic_model", lambda *args: None)
    normalized = SimpleNamespace(path=row.path, transforms=[], findings=[])
    monkeypatch.setattr(qualify, "normalize_source", lambda *_args: nullcontext(normalized))
    monkeypatch.setattr(
        qualify.PipelineAdapter, "convert_result", lambda *a, **k: SimpleNamespace(blocks=[])
    )
    monkeypatch.setattr(
        sys, "argv",
        ["qualify", "--stage", "collect", "--stage-input", str(incoming), "--output", str(output)],
    )

    with pytest.raises(SystemExit) as stopped:
        qualify.main()
    assert stopped.value.code == 43
    text = output.read_text()
    assert "private" not in text
    assert json.loads(text)["static_hook_failure"] == {
        "model_name": "UVDoc",
        "reason_code": "hook_cpu_float",
        "input_count": 1,
        "output_count": 1,
    }


@pytest.mark.parametrize("kind", ["fifo", "symlink"])
def test_static_hook_failure_reader_rejects_nonregular_paths_without_blocking(tmp_path, kind):
    import os
    import subprocess
    import sys

    report = tmp_path / "failure.json"
    if kind == "fifo":
        os.mkfifo(report)
    else:
        target = tmp_path / "real.json"
        target.write_text(json.dumps({
            "qualified": False,
            "stage": "native_collection",
            "failure": "static_hook_collection_failed",
            "static_hook_failure": {
                "model_name": "UVDoc",
                "reason_code": "hook_cpu_float",
                "input_count": 1,
                "output_count": 1,
            },
        }))
        report.symlink_to(target)

    subprocess.run(
        [
            sys.executable, "-c",
            (
                "from pathlib import Path\n"
                "import sys\n"
                "from ocr_service.qualify import read_static_hook_failure\n"
                "assert read_static_hook_failure(Path(sys.argv[1])) is None"
            ),
            str(report),
        ],
        cwd=Path(__file__).resolve().parents[1] / "paddleocr-service",
        check=True,
        timeout=3,
    )
