import hashlib
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image


def tensor(dtype="float32", gpu=0):
    return SimpleNamespace(
        dtype="paddle." + dtype,
        place=SimpleNamespace(
            is_gpu_place=lambda: gpu is not None,
            gpu_device_id=lambda: gpu,
            is_cpu_place=lambda: gpu is None,
        ),
    )


def test_fp32_override_precedes_single_loader_even_on_bf16_device():
    from ocr_service.fp32 import fp32_predictor_class

    calls = []

    class Upstream:
        def __init__(self, **kwargs):
            self.dtype = "bfloat16"
            self.infer, self.processor = self._build(**kwargs)

        def _build(self, **kwargs):
            calls.append(self.dtype)
            return SimpleNamespace(parameters=lambda: [tensor()]), object()

    predictor = fp32_predictor_class(Upstream, 0)()
    assert calls == ["float32"]
    assert predictor.dtype == "float32"


@pytest.mark.parametrize(
    "dtype,gpu", [("bfloat16", 0), ("float16", 0), ("float32", None), ("float32", 1)]
)
def test_dynamic_audit_rejects_wrong_precision_or_place(dtype, gpu):
    from ocr_service.fp32 import fp32_predictor_class

    class Upstream:
        def _build(self, **kwargs):
            return SimpleNamespace(parameters=lambda: [tensor(dtype, gpu)]), object()

    with pytest.raises(ValueError):
        fp32_predictor_class(Upstream, 0)()._build()


def test_dynamic_audit_rejects_empty_parameters():
    from ocr_service.fp32 import audit_dynamic_model

    with pytest.raises(ValueError, match="floating parameters"):
        audit_dynamic_model(SimpleNamespace(parameters=list), 0)


def test_local_pipeline_only_substitutes_vl_and_normalizes_engine(tmp_path):
    from ocr_service.fp32 import native_pipeline_class

    seen = []

    class Upstream:
        def create_model(self, config, **kwargs):
            return ("delegated", config, kwargs)

    def predictor(**kwargs):
        seen.append(kwargs)
        return "native"

    def normalize(engine, config, **kwargs):
        assert engine == "paddle_dynamic"
        assert kwargs == {"model_name": "PaddleOCR-VL-1.6-0.9B", "device": "gpu:2"}
        return {**config, "normalized": True}

    pipeline = native_pipeline_class(Upstream, predictor, normalize)()
    pipeline.device = "gpu:2"
    config = {
        "module_name": "vl_recognition",
        "model_name": "PaddleOCR-VL-1.6-0.9B",
        "model_dir": str(tmp_path),
        "batch_size": 1,
        "engine_config": {"device_type": "gpu", "device_id": 2},
    }
    assert pipeline.create_model(config) == "native"
    assert seen == [
        {
            "model_name": "PaddleOCR-VL-1.6-0.9B",
            "model_dir": str(tmp_path),
            "batch_size": 1,
            "engine_config": {"device_type": "gpu", "device_id": 2, "normalized": True},
        }
    ]
    assert (
        pipeline.create_model({"module_name": "layout_detection"}, threshold=0.3)[0] == "delegated"
    )


def test_accelerator_build_checks_actual_backend():
    from ocr_service.profile import check_accelerator

    cuda = SimpleNamespace(
        is_compiled_with_rocm=lambda: False,
        is_compiled_with_cuda=lambda: True,
        device=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1)),
    )
    check_accelerator(cuda, "cuda", 0)
    with pytest.raises(ValueError):
        check_accelerator(cuda, "rocm", 0)
    with pytest.raises(ValueError):
        check_accelerator(cuda, "cuda", 1)


def artifact(tmp_path):
    from ocr_service.precision import StaticArtifact

    graph = tmp_path / "inference.json"
    params = tmp_path / "inference.pdiparams"
    graph.write_bytes(b"graph")
    params.write_bytes(b"weights")
    return StaticArtifact(
        model_name="UVDoc",
        graph_file=graph.name,
        graph_sha256=hashlib.sha256(b"graph").hexdigest(),
        params_file=params.name,
        params_sha256=hashlib.sha256(b"weights").hexdigest(),
        inspector_identity="paddle-3.3.1-source-sha",
        inspector_version="1",
        parameters=[{"name": "w", "shape": [3, 3], "dtype": "float32"}],
        parameter_bindings={"w": "compiled_w"},
        coverage=[
            {
                "hook": "input",
                "op": "conv2d",
                "name": "compiled_w",
                "dtype": "float32",
                "place": "gpu:0",
            },
            {
                "hook": "output",
                "op": "conv2d",
                "name": "result",
                "dtype": "float32",
                "place": "gpu:0",
            },
        ],
    )


def static_session(tmp_path, config=None):
    from ocr_service.precision import begin_static_audit

    class Runner:
        def __init__(self):
            self._model_name = "UVDoc"
            self.model_dir = tmp_path
            self.model_file_prefix = "inference"
            self._config = config or {
                "run_mode": "paddle_fp32",
                "device_type": "gpu",
                "device_id": 0,
                "enable_cinn": False,
            }
            self.predictor = SimpleNamespace(
                register_input_hook=lambda f: setattr(self, "input", f),
                register_output_hook=lambda f: setattr(self, "output", f),
            )

    runner = Runner()
    session = begin_static_audit(
        SimpleNamespace(runner=runner),
        expected_model_name="UVDoc",
        gpu_index=0,
        sealed_artifact=artifact(tmp_path),
        runner_type=Runner,
        inspector_identity="paddle-3.3.1-source-sha",
    )
    return session, runner


def test_static_audit_rejects_empty_hooks_and_missing_weight(tmp_path):
    session, runner = static_session(tmp_path)
    session.start()
    with pytest.raises(ValueError, match="coverage"):
        session.finish()
    session.start()
    runner.output("conv2d", "result", tensor())
    with pytest.raises(ValueError, match="coverage"):
        session.finish()


def test_static_audit_requires_all_declared_weights_and_outputs(tmp_path):
    session, runner = static_session(tmp_path)
    session.start()
    runner.input("conv2d", "compiled_w", tensor())
    runner.output("conv2d", "result", tensor())
    evidence = session.finish()
    assert evidence["model_name"] == "UVDoc"
    assert evidence["input_count"] == evidence["output_count"] == 1
    assert evidence["floating_parameters"] == 1


@pytest.mark.parametrize(
    "dtype,gpu", [("bfloat16", 0), ("float16", 0), ("float32", None), ("float32", 1)]
)
def test_static_audit_rejects_hidden_wrong_precision_even_with_fp32_output(tmp_path, dtype, gpu):
    session, runner = static_session(tmp_path)
    session.start()
    runner.input("conv2d", "compiled_w", tensor(dtype, gpu))
    runner.output("conv2d", "result", tensor())
    with pytest.raises(ValueError):
        session.finish()


def test_static_audit_rejects_resolved_cpu_fallback(tmp_path):
    with pytest.raises(ValueError, match="resolved"):
        static_session(
            tmp_path,
            {"run_mode": "paddle_fp32", "device_type": "cpu", "device_id": 0, "enable_cinn": False},
        )


def test_static_manifest_is_bound_to_artifact_bytes(tmp_path):
    from ocr_service.precision import validate_artifact

    manifest = artifact(tmp_path)
    (tmp_path / "inference.pdiparams").write_bytes(b"drift")
    with pytest.raises(ValueError, match="hash"):
        validate_artifact(manifest, tmp_path, "paddle-3.3.1-source-sha")


def test_integer_cpu_shape_tensors_are_allowed(tmp_path):
    session, runner = static_session(tmp_path)
    session.start()
    runner.input("shape", "shape", tensor("int64", None))
    runner.input("conv2d", "compiled_w", tensor())
    runner.output("conv2d", "result", tensor())
    assert session.finish()["input_count"] == 2


def test_recognized_blocks_use_working_image_geometry_only():
    from ocr_service.pipeline import recognized_blocks

    blocks = recognized_blocks(
        {
            "parsing_res_list": [
                {"block_content": "TOTAL 1.25", "block_label": "text", "block_bbox": [0, 0, 3, 8]},
                SimpleNamespace(content="<table>1.25</table>", label="table", bbox=[0, 0, 3, 8]),
                {"block_content": " ", "block_bbox": [0, 0, 3, 8]},
            ],
            "layout_det_res": {"input_img": object()},
        },
        3,
        8,
    )
    assert [b.label for b in blocks] == ["text", "table"]
    assert blocks[0].bbox == [[0, 0], [3, 0], [3, 8], [0, 8]]
    with pytest.raises(ValueError, match="bounds"):
        recognized_blocks(
            {"parsing_res_list": [{"block_content": "bad", "block_bbox": [0, 0, 9, 8]}]}, 3, 8
        )
    with pytest.raises(ValueError, match="recognized text"):
        recognized_blocks({"layout_det_res": {"boxes": [{"text": "invented"}]}}, 3, 8)


def test_adapter_uses_exact_predict_limits_and_real_output(tmp_path):
    from ocr_service.pipeline import PipelineAdapter
    from ocr_service.profile import load_profile

    profile = load_profile(Path(__file__).parents[1] / "paddleocr-service/profile.json")
    source = tmp_path / "source.png"
    Image.new("RGB", (9, 5), "white").save(source)
    calls = []

    class Native:
        def predict(self, image, **kwargs):
            assert Image.open(image).mode == "RGB"
            calls.append(kwargs)
            yield {
                "doc_preprocessor_res": {
                    "output_img": np.zeros((8, 3, 3), dtype=np.uint8),
                    "angle": 90,
                },
                "parsing_res_list": [
                    {"block_label": "formula", "block_content": "1+1", "block_bbox": [0, 0, 3, 8]}
                ],
            }

    adapter = PipelineAdapter(Native(), profile, "profile", "runtime", [])
    with pytest.raises(ValueError, match="qualified"):
        adapter.predict(
            source,
            request_id=uuid4(),
            photo_index=0,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )
    # An empty audit inventory must never qualify production; focused output tests use the
    # conversion helper, which is independently exercised by the adapter after auditing.
    with pytest.raises(ValueError, match="three static"):
        adapter.qualify(source)
    response = adapter.convert_result(
        next(Native().predict(str(source))),
        request_id=uuid4(),
        photo_index=0,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        transforms=[],
        findings=[],
    )
    assert response.working_image.width == 3
    assert response.working_image.height == 8
    assert response.blocks[0].label == "formula"
    assert response.working_image.inverse_mapping is None
    assert adapter.predict_kwargs == {
        "use_doc_orientation_classify": True,
        "use_doc_unwarping": True,
        "use_queues": False,
        "min_pixels": 112896,
        "max_pixels": 2000000,
        "max_new_tokens": 4096,
    }


def test_pinned_source_contract_uses_real_paddleocr_vl_module():
    from ocr_service.pipeline import PINNED_SOURCES

    assert "pipelines/paddleocr_vl/pipeline.py" in PINNED_SOURCES
    assert "pipelines/ocr_vl/pipeline.py" not in PINNED_SOURCES


def test_factory_rejects_preimported_paddlex_without_modifying_environment(monkeypatch):
    import os
    import sys

    from ocr_service.pipeline import build_pipeline

    monkeypatch.setitem(sys.modules, "paddlex", SimpleNamespace())
    monkeypatch.setenv("PADDLE_PDX_DISABLE_DEVICE_FALLBACK", "0")
    with pytest.raises(ValueError, match="fresh worker"):
        build_pipeline(None, None)
    assert os.environ["PADDLE_PDX_DISABLE_DEVICE_FALLBACK"] == "0"


def test_source_verifier_rejects_drift(tmp_path):
    from ocr_service.pipeline import PINNED_SOURCES, verify_sources

    for relative in PINNED_SOURCES:
        path = tmp_path / "inference" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unsupported")
    with pytest.raises(ValueError, match="source identity"):
        verify_sources(tmp_path)


@pytest.fixture
def native_runtime(tmp_path, monkeypatch):
    import json
    from types import ModuleType

    import ocr_service.pipeline as factory
    from ocr_service import device_inventory
    from ocr_service.profile import MODEL_NAMES, RuntimeSettings

    source = tmp_path / "smoke.png"
    Image.new("RGB", (3, 8), "white").save(source)
    manifests = {}
    model_dirs = {}
    for name in MODEL_NAMES:
        directory = tmp_path / name
        directory.mkdir()
        model_dirs[name] = directory
        record = artifact(directory).model_dump()
        record["model_name"] = name
        record["inspector_identity"] = "paddle:commit-331"
        manifests[name] = record
    dependency_versions = {
        "paddleocr": "3.7.0",
        "paddlex": "3.7.2",
        "paddlepaddle-gpu": "3.3.1",
        "pillow": "12.3.0",
        "pillow-heif": "1.7.0",
        "numpy": "2.2.6",
        "pydantic": "2.13.4",
        "packaging": "26.2",
    }
    cuda_metadata = {
        "accelerator": "cuda",
        "gpu_index": 0,
        "name": "fake-native-GPU",
        "uuid": "000102030405060708090a0b0c0d0e0f",
        "pci_bus_id": "0000:01:00.0",
        "driver_version": 12000,
        "compute_capability": [8, 6],
    }
    monkeypatch.setattr(
        device_inventory, "_cuda_identity", lambda index: {**cuda_metadata, "gpu_index": index}
    )
    manifest_path = tmp_path / "qualification.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dependencies": {
                    name: {"version": version, "content_sha256": "a" * 64}
                    for name, version in dependency_versions.items()
                },
                "qualified_target": {**cuda_metadata, "paddle_build": "3.3.1"},
                "image_identity": "image-sha256",
                "paddle_identity": "paddle:commit-331",
                "smoke_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "model_content": {
                    name: factory.tree_identity(directory) for name, directory in model_dirs.items()
                },
                "static_artifacts": {name: manifests[name] for name in MODEL_NAMES[:3]},
            }
        )
    )
    runtime = RuntimeSettings(
        accelerator="cuda",
        gpu_index=0,
        model_dirs=model_dirs,
        cache_dir=tmp_path / "cache",
        qualification_manifest=manifest_path,
        expected_packages=dependency_versions,
        expected_paddle_commit="commit-331",
        image_identity="image-sha256",
    )
    versions = runtime.expected_packages
    monkeypatch.setattr(
        factory.importlib.metadata,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": "paddlepaddle-gpu"})],
    )
    monkeypatch.setattr(factory.importlib.metadata, "version", versions.__getitem__)
    # Sources verified independently against official hard pins above. This fake
    # framework test intentionally isolates construction and execution behavior.
    monkeypatch.setattr(factory, "verify_sources", lambda root: None)
    modules = {}

    def module(name, **values):
        item = ModuleType(name)
        item.__dict__.update(values)
        modules[name] = item
        return item

    calls = []

    class Runner:
        def __init__(self, config):
            self._model_name = config["model_name"]
            self.model_dir = Path(config["model_dir"])
            self.model_file_prefix = "inference"
            self._config = config["engine_config"]
            self.predictor = SimpleNamespace(
                register_input_hook=lambda f: setattr(self, "input", f),
                register_output_hook=lambda f: setattr(self, "output", f),
            )

    class Predictor:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.dtype = "bfloat16"
            self.infer, self.processor = self._build(**kwargs)

        def _build(self, **kwargs):
            calls.append({"loader_dtype": self.dtype})
            return SimpleNamespace(parameters=lambda: [tensor()]), object()

    class Base:
        def create_model(self, config, **kwargs):
            return SimpleNamespace(runner=Runner(config))

    class Pipeline(Base):
        def __init__(self, config, **kwargs):
            self.device = kwargs["device"]
            self.layout_det_model = self.create_model(config["SubModules"]["LayoutDetection"])
            self.vl_rec_model = self.create_model(config["SubModules"]["VLRecognition"])
            prep = config["SubPipelines"]["DocPreprocessor"]["SubModules"]
            self.doc_preprocessor_pipeline = SimpleNamespace(
                doc_ori_classify_model=self.create_model(prep["DocOrientationClassify"]),
                doc_unwarping_model=self.create_model(prep["DocUnwarping"]),
            )

        def predict(self, image, **kwargs):
            calls.append(kwargs)
            for model in [
                self.layout_det_model,
                self.doc_preprocessor_pipeline.doc_ori_classify_model,
                self.doc_preprocessor_pipeline.doc_unwarping_model,
            ]:
                if hasattr(model.runner, "input"):
                    model.runner.input("conv2d", "compiled_w", tensor())
                    model.runner.output("conv2d", "result", tensor())
            yield {
                "doc_preprocessor_res": {
                    "output_img": np.zeros((8, 3, 3), dtype=np.uint8),
                    "angle": 0,
                },
                "parsing_res_list": [{"block_content": "SHOP", "block_bbox": [0, 0, 3, 8]}],
            }

    cuda = SimpleNamespace(
        device_count=lambda: 1, get_device_name=lambda index: cuda_metadata["name"]
    )
    module(
        "paddle",
        is_compiled_with_rocm=lambda: False,
        is_compiled_with_cuda=lambda: True,
        device=SimpleNamespace(cuda=cuda),
        version=SimpleNamespace(commit="commit-331", full_version="3.3.1"),
    )
    module("paddlex", __file__=str(tmp_path / "paddlex/__init__.py"))
    module(
        "paddlex.inference.models", normalize_engine_config=lambda engine, config, **kwargs: config
    )
    module("paddlex.inference.models.doc_vlm.predictor", DocVLMLocalPredictor=Predictor)
    module("paddlex.inference.models.runners.paddle_static", PaddleStaticRunner=Runner)
    module("paddlex.inference.pipelines.base", BasePipeline=Base)
    module("paddlex.inference.pipelines.paddleocr_vl.pipeline", _PaddleOCRVLPipeline=Pipeline)
    # Inject fake frameworks only when the factory reaches its lazy imports.
    original_import = factory.importlib.import_module

    def importing(name, *args, **kwargs):
        if name in modules:
            return modules[name]
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(factory.importlib, "import_module", importing)
    from ocr_service import profile as profile_module

    monkeypatch.setattr(profile_module.importlib.metadata, "requires", lambda name: [])
    monkeypatch.setattr(profile_module, "dependency_identity", lambda name: "a" * 64)
    return runtime, source, cuda_metadata, calls


@pytest.mark.parametrize("target_change", ["name", "architecture", "override", "missing_seal"])
def test_factory_bootstraps_then_requires_real_forward_for_all_models(native_runtime, monkeypatch, target_change):
    import json

    import ocr_service.pipeline as factory
    from ocr_service.profile import MODEL_NAMES, load_profile

    runtime, source, cuda_metadata, calls = native_runtime
    manifest_path = runtime.qualification_manifest
    adapter = factory.build_pipeline(
        load_profile(Path(__file__).parents[1] / "paddleocr-service/profile.json"), runtime
    )
    assert adapter.qualified is False
    assert [call for call in calls if "loader_dtype" in call] == [{"loader_dtype": "float32"}]
    evidence = adapter.qualify(source)
    assert {item["model_name"] for item in evidence} == set(MODEL_NAMES[:3])
    response = adapter.predict(
        source,
        request_id=uuid4(),
        photo_index=2,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    assert response.text == "SHOP"
    assert response.photo_index == 2
    assert calls[-1] == adapter.predict_kwargs
    assert response.profile_id != response.runtime_fingerprint
    # A later unexecuted static model invalidates qualification again.
    adapter.pipeline.doc_preprocessor_pipeline.doc_unwarping_model.runner.input = lambda *args: None
    with pytest.raises(ValueError, match="coverage"):
        adapter.predict(
            source,
            request_id=uuid4(),
            photo_index=2,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )
    assert adapter.qualified is False
    if target_change == "name":
        cuda_metadata["name"] = "another-device"
    elif target_change == "architecture":
        cuda_metadata["compute_capability"] = [9, 0]
    elif target_change == "override":
        monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    else:
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("qualified_target")
        manifest_path.write_text(json.dumps(manifest))
    expected_error = "spoofing" if target_change == "override" else "qualified target"
    previous_calls = calls.copy()
    with pytest.raises(ValueError, match=expected_error):
        factory.build_pipeline(
            load_profile(Path(__file__).parents[1] / "paddleocr-service/profile.json"), runtime
        )
    assert calls == previous_calls


@pytest.mark.parametrize("prepared", [False, True], ids=["qualified-offline", "prepared"])
def test_serving_load_leaves_first_forward_to_real_receipt(native_runtime, monkeypatch, prepared):
    import json

    from ocr_service import pipeline
    from ocr_service.profile import PreparedRuntimeSettings, load_profile, serving_identity

    runtime, receipt, device, calls = native_runtime
    profile = load_profile(Path(__file__).parents[1] / "paddleocr-service/profile.json")
    if prepared:
        runtime = PreparedRuntimeSettings(
            **runtime.model_dump(exclude={"qualification_manifest"}),
            schema_version=2,
            model_content=json.loads(runtime.qualification_manifest.read_bytes())["model_content"],
            device_identity=device.copy(),
        )
    monkeypatch.setattr(
        pipeline,
        "begin_static_audit",
        lambda *a, **kw: pytest.fail("serving attached a diagnostic hook"),
    )
    adapter = pipeline.build_serving_pipeline(profile, runtime)
    assert not any("min_pixels" in call for call in calls)
    if prepared:
        identity = serving_identity(profile, runtime, runtime.model_content, device)
        assert (adapter.profile_id, adapter.runtime_fingerprint) == (
            identity["profile_id"],
            identity["runtime_fingerprint"],
        )
    page = adapter.predict(
        receipt,
        request_id=uuid4(),
        photo_index=0,
        source_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
    )
    assert page.text == "SHOP"
    assert "precision_audits" not in page.telemetry
    assert [call for call in calls if "max_pixels" in call] == [
        {
            "use_doc_orientation_classify": True,
            "use_doc_unwarping": True,
            "use_queues": False,
            "min_pixels": 112896,
            "max_pixels": 2000000,
            "max_new_tokens": 4096,
        }
    ]


@pytest.mark.parametrize("drift", ["device", "model", "symlink"])
def test_prepared_serving_refuses_identity_drift_before_loading(native_runtime, drift):
    import json

    from ocr_service.pipeline import build_serving_pipeline
    from ocr_service.profile import PreparedRuntimeSettings, RuntimeIdentityMismatch, load_profile

    old, _, device, calls = native_runtime
    runtime = PreparedRuntimeSettings(
        **old.model_dump(exclude={"qualification_manifest"}),
        schema_version=2,
        model_content=json.loads(old.qualification_manifest.read_bytes())["model_content"],
        device_identity=device.copy(),
    )
    if drift == "model":
        (runtime.model_dirs["UVDoc"] / "extra").write_bytes(b"changed")
    elif drift == "symlink":
        directory = runtime.model_dirs["UVDoc"]
        moved = directory.rename(directory.with_name("unowned"))
        directory.symlink_to(moved, target_is_directory=True)
    else:
        device["uuid"] = "11" * 16
    profile = load_profile(Path(__file__).parents[1] / "paddleocr-service/profile.json")
    with pytest.raises(RuntimeIdentityMismatch):
        build_serving_pipeline(profile, runtime)
    assert calls == []


def test_static_runner_actual_model_name_must_match(tmp_path):
    from ocr_service.precision import begin_static_audit

    class Runner:
        def __init__(self):
            self._model_name = "another-model"
            self.predictor = SimpleNamespace(
                register_input_hook=lambda f: None, register_output_hook=lambda f: None
            )
            self.model_dir = tmp_path
            self.model_file_prefix = "inference"
            self._config = {
                "run_mode": "paddle_fp32",
                "device_type": "gpu",
                "device_id": 0,
                "enable_cinn": False,
            }

    with pytest.raises(ValueError, match="identity"):
        begin_static_audit(
            SimpleNamespace(runner=Runner()),
            expected_model_name="UVDoc",
            gpu_index=0,
            sealed_artifact=artifact(tmp_path),
            runner_type=Runner,
            inspector_identity="paddle-3.3.1-source-sha",
        )


def test_importing_sidecar_does_not_import_native_frameworks():
    import os
    import subprocess
    import sys

    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(__file__).parents[1] / "src"),
            str(Path(__file__).parents[1] / "paddleocr-service"),
        ]
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import ocr_service.pipeline; "
                'assert not any(name in sys.modules for name in ("paddle", "paddlex", "paddleocr", "numpy"))'
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_effective_predict_arguments_cannot_be_mutated_outside_profile():
    from ocr_service.pipeline import PipelineAdapter
    from ocr_service.profile import load_profile

    profile = load_profile(Path(__file__).parents[1] / "paddleocr-service/profile.json")
    adapter = PipelineAdapter(None, profile, "profile", "runtime", [])
    arguments = adapter.predict_kwargs
    arguments["max_pixels"] = 1
    arguments["temperature"] = 0.2
    assert adapter.predict_kwargs["max_pixels"] == 2000000
    assert "temperature" not in adapter.predict_kwargs


def test_declared_coverage_cannot_authorize_another_gpu(tmp_path):
    from ocr_service.precision import HookRecord, StaticAuditSession

    manifest = artifact(tmp_path)
    manifest.coverage.append(
        HookRecord(hook="output", op="conv2d", name="other_gpu", dtype="float32", place="gpu:1")
    )
    session = StaticAuditSession(manifest, gpu_index=0)
    session.start()
    session.observe_input("conv2d", "compiled_w", tensor())
    session.observe_output("conv2d", "result", tensor())
    session.observe_output("conv2d", "other_gpu", tensor(gpu=1))
    with pytest.raises(ValueError, match="device"):
        session.finish()


@pytest.mark.parametrize("place_name", ["gpu_pinned", "custom", "xpu"])
def test_declared_cpu_coverage_rejects_non_cpu_non_gpu_place(tmp_path, place_name):
    from ocr_service.precision import HookRecord, StaticAuditSession

    manifest = artifact(tmp_path)
    manifest.coverage.append(
        HookRecord(hook="output", op="transfer", name="cpu_fetch", dtype="float32", place="cpu")
    )
    session = StaticAuditSession(manifest, gpu_index=0)
    session.start()
    session.observe_input("conv2d", "compiled_w", tensor())
    session.observe_output("conv2d", "result", tensor())
    unknown_place = SimpleNamespace(
        is_gpu_place=lambda: False, is_cpu_place=lambda: False, name=place_name
    )
    session.observe_output(
        "transfer", "cpu_fetch", SimpleNamespace(dtype="paddle.float32", place=unknown_place)
    )
    with pytest.raises(ValueError, match="device"):
        session.finish()


def test_genuine_cpu_float_requires_exact_approved_transfer_record(tmp_path):
    from ocr_service.precision import HookRecord, StaticAuditSession

    manifest = artifact(tmp_path)
    manifest.coverage.append(
        HookRecord(hook="output", op="transfer", name="cpu_fetch", dtype="float32", place="cpu")
    )
    session = StaticAuditSession(manifest, gpu_index=0)
    session.start()
    session.observe_input("conv2d", "compiled_w", tensor())
    session.observe_output("conv2d", "result", tensor())
    session.observe_output("transfer", "cpu_fetch", tensor(gpu=None))
    assert session.finish()["output_count"] == 2


@pytest.mark.parametrize("drift", ["omission", "version", "content"])
def test_factory_rejects_dependency_identity_drift_before_heavy_imports(
    tmp_path, monkeypatch, drift
):
    import json

    import ocr_service.pipeline as factory
    from ocr_service.profile import MODEL_NAMES, RuntimeSettings, load_profile

    expected = {
        "paddleocr": "3.7.0",
        "paddlex": "3.7.2",
        "paddlepaddle-gpu": "3.3.1",
        "pillow": "12.3.0",
        "pillow-heif": "1.7.0",
        "numpy": "2.2.6",
        "pydantic": "2.13.4",
        "packaging": "26.2",
    }
    manifest = tmp_path / "qualification.json"
    manifest.write_text(
        json.dumps(
            {
                "dependencies": {
                    name: {"version": version, "content_sha256": "a" * 64}
                    for name, version in expected.items()
                }
            }
        )
    )
    actual_versions = dict(expected)
    actual_contents = {name: "a" * 64 for name in expected}
    if drift == "omission":
        expected.pop("numpy")
    elif drift == "version":
        actual_versions["pillow"] = "99.0"
    else:
        actual_contents["pillow"] = "b" * 64
    runtime = RuntimeSettings(
        accelerator="cuda",
        gpu_index=0,
        model_dirs={name: tmp_path / name for name in MODEL_NAMES},
        cache_dir=tmp_path / "cache",
        qualification_manifest=manifest,
        expected_packages=expected,
        expected_paddle_commit="commit-331",
        image_identity="image-identity",
    )
    from ocr_service import profile as profile_module

    monkeypatch.setattr(
        factory.importlib.metadata,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": "paddlepaddle-gpu"})],
    )
    monkeypatch.setattr(factory.importlib.metadata, "version", actual_versions.__getitem__)
    monkeypatch.setattr(factory.importlib.metadata, "requires", lambda name: [])
    monkeypatch.setattr(profile_module, "dependency_identity", actual_contents.__getitem__)

    def forbidden_import(name):
        pytest.fail("Required identity omission reached a heavy import: " + name)

    monkeypatch.setattr(factory.importlib, "import_module", forbidden_import)
    with pytest.raises(ValueError, match="required|identity"):
        factory.build_pipeline(
            load_profile(Path(__file__).parents[1] / "paddleocr-service/profile.json"), runtime
        )
