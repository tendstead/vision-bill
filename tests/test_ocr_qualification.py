"""CPU-only contract tests: fake probes never establish native qualification."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_ocr_pipeline import artifact, tensor


def qualification():
    assert importlib.util.find_spec("ocr_service.qualify"), "qualification tooling missing"
    return importlib.import_module("ocr_service.qualify")


def bootstrap():
    assert importlib.util.find_spec("ocr_service.bootstrap"), "bootstrap tooling missing"
    return importlib.import_module("ocr_service.bootstrap")


def test_collector_requires_each_forward_complete_not_corpus_union():
    module = bootstrap()
    collector = module.HookCollector(0, max_records=4)
    collector.start()
    collector.observe_input("matmul", "weight", tensor())
    collector.observe_output("matmul", "result", tensor())
    first = collector.finish()
    collector.start()
    collector.observe_input("matmul", "other_weight", tensor())
    collector.observe_output("matmul", "result", tensor())
    second = collector.finish()
    with pytest.raises(ValueError, match="coverage differs"):
        module.common_coverage([first, second])
    assert module.common_coverage([first, first]) == first


def test_fresh_static_runner_replays_pir_temporaries_without_changing_weight_names(tmp_path):
    from ocr_service.precision import StaticAuditSession

    source = artifact(tmp_path)
    collector = bootstrap().HookCollector(0, parameter_names={"w"})
    collector.start()
    collector.observe_input("pd_op.conv2d:9", "w", tensor())
    collector.observe_output("pd_op.conv2d:9", "0x7f642a1301732000000000000000_inner_var_16", tensor())
    coverage = collector.finish()
    assert bootstrap().parameter_bindings(source.parameters, coverage, 0) == {"w": "w"}
    session = StaticAuditSession(
        source.model_copy(update={"coverage": coverage, "parameter_bindings": {"w": "w"}}), 0
    )
    session.start()
    session.observe_input("pd_op.conv2d:9", "w", tensor())
    session.observe_output("pd_op.conv2d:9", "0x7fb6416801732000000000000001_inner_var_16", tensor())
    assert session.finish()["floating_parameters"] == 1

    session.start()
    session.observe_input("pd_op.conv2d:9", "w", tensor())
    session.observe_output("pd_op.conv2d:9", "0x7fb6416801732000000000000001_inner_var_17", tensor())
    with pytest.raises(ValueError, match="coverage"):
        session.finish()


def test_collector_rejects_ambiguous_pir_scope_instead_of_collapsing_variables():
    collector = bootstrap().HookCollector(0)
    collector.start()
    collector.observe_input("pd_op.add:9", "0x7f642a1301732000000000000000_inner_var_16", tensor())
    collector.observe_output("pd_op.add:9", "0x7fb6416801732000000000000001_inner_var_16", tensor())
    with pytest.raises(ValueError, match="coverage"):
        collector.finish()


def test_same_static_predictor_rejects_anonymous_scope_changes_between_forwards(tmp_path):
    from ocr_service.precision import StaticAuditSession

    source = artifact(tmp_path)
    collector = bootstrap().HookCollector(0)
    initial = "0x7f642a1301732000000000000000_inner_var_16"
    changed = "0x7fb6416801732000000000000001_inner_var_16"
    collector.start()
    collector.observe_input("pd_op.conv2d:9", "w", tensor())
    collector.observe_output("pd_op.conv2d:9", initial, tensor())
    session = StaticAuditSession(
        source.model_copy(update={"coverage": collector.finish(), "parameter_bindings": {"w": "w"}}),
        0,
    )
    session.start()
    session.observe_input("pd_op.conv2d:9", "w", tensor())
    session.observe_output("pd_op.conv2d:9", initial, tensor())
    session.finish()
    for audit in (collector, session):
        audit.start()
        audit.observe_input("pd_op.conv2d:9", "w", tensor())
        audit.observe_output("pd_op.conv2d:9", changed, tensor())
        with pytest.raises(ValueError, match="coverage"):
            audit.finish()


def test_exported_weight_cannot_use_a_reserved_pir_temporary_identity():
    with pytest.raises(ValueError, match="reserved PIR identity prefix"):
        bootstrap().HookCollector(0, parameter_names={"pir-temporary:16"})


def test_hook_collection_preserves_a_named_tensor_that_only_resembles_a_pir_temporary():
    collector = bootstrap().HookCollector(0)
    collector.start()
    collector.observe_input("pd_op.relu:5", "0xdeadbeef_inner_var_1", tensor())
    collector.observe_output("pd_op.relu:5", "out", tensor())
    assert [record.name for record in collector.finish()] == ["0xdeadbeef_inner_var_1", "out"]


@pytest.mark.parametrize(
    "dtype,gpu", [("float16", 0), ("bfloat16", 0), ("float32", 1), ("float32", None)]
)
def test_unsealed_collector_rejects_precision_device_and_unproven_cpu_float(dtype, gpu):
    collector = bootstrap().HookCollector(0)
    collector.start()
    collector.observe_input("matmul", "weight", tensor(dtype, gpu))
    collector.observe_output("matmul", "result", tensor())
    with pytest.raises(ValueError):
        collector.finish()


def test_collector_reports_first_safe_reason_and_native_callback_counts():
    module = bootstrap()
    collector = module.HookCollector(0)
    collector.start()
    collector.observe_input("memcpy_h2d", "private-tensor-name", tensor("float32", None))
    collector.observe_input("conv2d", "another-private-name", tensor("float16"))
    collector.observe_output("conv2d", "out", tensor())
    with pytest.raises(module.HookCollectionError) as captured:
        collector.finish()
    failure = captured.value
    assert (failure.reason_code, failure.input_count, failure.output_count) == ("hook_cpu_float", 2, 1)
    assert "private" not in str(failure)


def test_collector_empty_and_unbounded_stream_fail():
    collector = bootstrap().HookCollector(0, max_records=2)
    collector.start()
    with pytest.raises(ValueError):
        collector.finish()
    collector.start()
    for index in range(10):
        collector.observe_input("op", str(index), tensor())
    assert len(collector.observed) <= 2
    with pytest.raises(ValueError):
        collector.finish()


def test_parameter_binding_uses_exact_names_not_shapes_or_list_order():
    from ocr_service.precision import ParameterRecord

    module = bootstrap()
    parameters = [ParameterRecord(name="weight", shape=[4], dtype="float32")]
    collector = module.HookCollector(0)
    collector.start()
    collector.observe_input("op", "renamed", tensor())
    collector.observe_output("op", "out", tensor())
    with pytest.raises(ValueError, match="parameter binding"):
        module.parameter_bindings(parameters, collector.finish(), 0)
    collector.start()
    collector.observe_input("op", "weight", tensor())
    collector.observe_output("op", "out", tensor())
    assert module.parameter_bindings(parameters, collector.finish(), 0) == {"weight": "weight"}


def test_dependency_sealer_traverses_extras_and_hashes_real_identity(monkeypatch):
    module = bootstrap()
    roots = {
        "paddleocr",
        "paddlex",
        "paddlepaddle-gpu",
        "pillow",
        "pillow-heif",
        "numpy",
        "pydantic",
        "packaging",
    }
    requirements = {
        "paddlex": ['extra-child[fast]>=1; extra == "ocr"'],
        "extra-child": ['leaf==2; extra == "fast"', 'ignored; python_version < "2"'],
    }
    monkeypatch.setattr(
        module.metadata,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": "paddlepaddle-gpu"})],
    )
    monkeypatch.setattr(module.metadata, "requires", lambda name: requirements.get(name, []))
    monkeypatch.setattr(module.metadata, "version", lambda name: "2" if name == "leaf" else "1")
    seen = []
    monkeypatch.setattr(module, "dependency_identity", lambda name: seen.append(name) or "a" * 64)
    expected, sealed = module.seal_dependencies("paddlepaddle-gpu")
    assert set(expected) == roots | {"extra-child", "leaf"}
    assert set(seen) == set(expected) == set(sealed)
    assert sealed["leaf"] == {"version": "2", "content_sha256": "a" * 64}


def test_case_manifest_requires_all_five_real_hash_bound_inputs(tmp_path):
    module = qualification()
    manifest = tmp_path / "fixtures.toml"
    manifest.write_text('[[cases]]\nkind="upright"\npath="a.png"\nsha256="' + "a" * 64 + '"\n')
    with pytest.raises(ValueError):
        module.load_cases(manifest)


@pytest.fixture
def dcu_metadata(tmp_path, monkeypatch):
    site_packages = tmp_path / "site-packages"
    monkeypatch.syspath_prepend(str(site_packages))

    def install(name, version, requires=()):
        info = site_packages / f"{name.lower().replace('-', '_')}-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
            + "".join(f"Requires-Dist: {entry}\n" for entry in requires)
        )
        (info / "RECORD").write_text(f"{info.name}/METADATA,,\n")

    install("paddlepaddle-dcu", "3.3.1", ["numpy>=2"])
    install("paddlex", "3.7.2", ['layout-extra>=1; extra == "ocr"'])
    for name, version in {
        "paddleocr": "3.7.0",
        "layout-extra": "1.0",
        "pillow": "12.3.0",
        "pillow-heif": "1.7.0",
        "numpy": "2.2.6",
        "pydantic": "2.13.4",
        "packaging": "26.2",
    }.items():
        install(name, version)
    return install


def test_rocm_binding_seals_exact_installed_metadata_closure(dcu_metadata):
    from ocr_service.bootstrap import bind_approved_dependencies, seal_dependencies
    from ocr_service.profile import verify_dependencies

    approved = {"paddlepaddle-dcu": "3.3.1", "paddleocr": "3.7.0", "paddlex": "3.7.2"}
    versions = bind_approved_dependencies(approved, accelerator="rocm")
    assert versions == {
        **approved,
        "layout-extra": "1.0",
        "pillow": "12.3.0",
        "pillow-heif": "1.7.0",
        "numpy": "2.2.6",
        "pydantic": "2.13.4",
        "packaging": "26.2",
    }
    assert seal_dependencies("paddlepaddle-dcu")[0] == versions
    _, sealed = seal_dependencies("paddlepaddle-dcu", accelerator="rocm")
    assert verify_dependencies(versions, sealed, accelerator="rocm")[0] == versions
    with pytest.raises(ValueError, match="accelerator"):
        verify_dependencies(versions, sealed, accelerator="cuda")
    with pytest.raises(ValueError, match="approved numpy"):
        bind_approved_dependencies({**approved, "numpy": "2.2.5"})


@pytest.mark.parametrize("extra_native", ["paddlepaddle", "paddlepaddle-gpu", "PaddlePaddle_DCU"])
def test_undeclared_native_install_rejected_during_sealing_and_verification(
    dcu_metadata, extra_native
):
    from ocr_service.bootstrap import seal_dependencies
    from ocr_service.profile import verify_dependencies

    versions, sealed = seal_dependencies("paddlepaddle-dcu")
    dcu_metadata(extra_native, "3.3.1+extra")
    with pytest.raises(ValueError, match="one.*Paddle"):
        seal_dependencies("paddlepaddle-dcu")
    with pytest.raises(ValueError, match="one.*Paddle"):
        verify_dependencies(versions, sealed)


def test_rocm_qualification_passes_dcu_metadata_to_inspection(tmp_path, monkeypatch, dcu_metadata):
    module = qualification()
    from ocr_service.profile import MODEL_NAMES, RuntimeSettings

    model_dirs = {name: tmp_path / name for name in MODEL_NAMES}
    for directory in model_dirs.values():
        directory.mkdir()
        (directory / "weights").write_text("synthetic model data")
    runtime = RuntimeSettings(
        accelerator="rocm",
        gpu_index=0,
        model_dirs=model_dirs,
        cache_dir=tmp_path,
        qualification_manifest=tmp_path / "manifest.json",
        expected_packages={},
        expected_paddle_commit="1234567",
        image_identity="sha256:" + "a" * 64,
    )
    cases = [
        module.QualificationCase(kind=kind, path=tmp_path / kind, sha256=str(i) * 64)
        for i, kind in enumerate(sorted(module.KINDS))
    ]
    monkeypatch.setattr(module, "load_cases", lambda path: cases)
    handoff = []

    def stop_before_hardware(stage, payload, directory):
        handoff.append((stage, payload))
        raise RuntimeError("Native inspection requires actual hardware")

    monkeypatch.setattr(module, "run_stage", stop_before_hardware)
    output = tmp_path / "qualification.json"
    report = module.qualify_runtime(runtime, tmp_path / "cases.toml", output)
    assert not report.qualified
    assert not output.with_suffix(".runtime.json").exists()
    assert report.stage == "artifact_inspection"
    [(stage, payload)] = handoff
    assert stage == "inspect"
    assert payload["runtime"]["expected_packages"]["paddlepaddle-dcu"] == "3.3.1"
    assert payload["dependencies"]["paddlepaddle-dcu"]["version"] == "3.3.1"


def test_failure_report_does_not_claim_native_success_or_leak_errors(tmp_path, monkeypatch):
    module = qualification()
    from ocr_service.profile import RuntimeSettings

    runtime = RuntimeSettings(
        accelerator="rocm",
        gpu_index=0,
        model_dirs={},
        cache_dir=tmp_path,
        qualification_manifest=tmp_path / "manifest.json",
        expected_packages={},
        expected_paddle_commit="1234567",
        image_identity="candidate",
    )
    monkeypatch.setattr(
        module,
        "load_cases",
        lambda path: (_ for _ in ()).throw(RuntimeError("private receipt text")),
    )
    output = tmp_path / "report.json"
    report = module.qualify_runtime(runtime, tmp_path / "cases.toml", output)
    assert not report.qualified
    assert report.qualified_targets == []
    assert "private receipt text" not in output.read_text()
    assert json.loads(output.read_text())["failure"] == "RuntimeError"


def test_build_pipeline_has_separate_unsealed_native_construction():
    from ocr_service import pipeline

    assert hasattr(pipeline, "construct_native"), "shared construction seam missing"


def test_hip_device_inventory_missing_kernel_fails(tmp_path, monkeypatch):
    module = qualification()
    from ocr_service.profile import RuntimeSettings

    runtime = RuntimeSettings(
        accelerator="rocm",
        gpu_index=0,
        model_dirs={},
        cache_dir=tmp_path,
        qualification_manifest=tmp_path / "manifest.json",
        expected_packages={},
        expected_paddle_commit="1234567",
        image_identity="candidate",
    )
    paddle = SimpleNamespace(
        device=SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda index: "Radeon")),
        version=SimpleNamespace(full_version="3.3.1"),
    )
    monkeypatch.setattr(
        module, "inspect_device_identity", lambda *a: {"gfx_target": "gfx1030", "name": "Radeon", "gpu_index": 0}
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self: json.dumps({"target_profile": {"schema": 1, "candidate_targets": ["gfx1030"]}, "paddle": {"targets": [], "files": {"library": "hash"}}}),
    )
    with pytest.raises(ValueError, match="kernels are missing"):
        module.device_identity(paddle, runtime)


def test_wrong_native_wheel_fails_accelerator_before_inference():
    from ocr_service.profile import check_accelerator

    paddle = SimpleNamespace(
        is_compiled_with_rocm=lambda: False, is_compiled_with_cuda=lambda: True
    )
    with pytest.raises(ValueError, match="native Paddle build"):
        check_accelerator(paddle, "rocm", 0)


def test_inspector_reads_actual_dense_tensor_metadata(tmp_path):
    module = bootstrap()
    from contextlib import nullcontext

    (tmp_path / "inference.json").write_bytes(b"graph")
    (tmp_path / "inference.pdiparams").write_bytes(b"params")
    variable = SimpleNamespace(name="weight", shape=[4], dtype="paddle.float32", persistable=True)
    dense = SimpleNamespace(
        shape=lambda: [4],
        _dtype=lambda: 7,
        _place=lambda: SimpleNamespace(is_cpu_place=lambda: True),
    )
    static = SimpleNamespace(
        Executor=lambda place: object(),
        load_inference_model=lambda *a: ("program", [], []),
        pir_io=SimpleNamespace(get_pir_parameters=lambda p: ([variable], [])),
        global_scope=lambda: SimpleNamespace(
            find_var=lambda n: SimpleNamespace(get_tensor=lambda: dense)
        ),
    )
    paddle = SimpleNamespace(
        pir_utils=SimpleNamespace(IrGuard=nullcontext),
        enable_static=lambda: None,
        CPUPlace=lambda: object(),
        static=static,
        base=SimpleNamespace(
            core=SimpleNamespace(VarDesc=SimpleNamespace(VarType=SimpleNamespace(FP32=7)))
        ),
    )
    result = module.inspect_artifact(tmp_path, "model", paddle, "source")
    assert result["parameters"] == [{"name": "weight", "shape": [4], "dtype": "float32"}]
    assert "coverage" not in result and "parameter_bindings" not in result
    dense._dtype = lambda: 8
    with pytest.raises(ValueError, match="not FP32"):
        module.inspect_artifact(tmp_path, "model", paddle, "source")


def test_qualification_rejects_success_shaped_but_incomplete_fresh_replay(tmp_path, monkeypatch):
    module = qualification()
    assert hasattr(module, "validate_replay"), "replay evidence validator missing"
    with pytest.raises(ValueError):
        module.validate_replay({"passed": True, "static_audits": [{"model_name": "one"}]}, {})


def test_native_compute_probe_requires_device_fp32_and_checked_math():
    module = qualification()
    assert hasattr(module, "compute_probe"), "bounded native compute probe missing"
    from contextlib import nullcontext

    class Value:
        dtype = "paddle.float32"
        place = tensor().place

        def __init__(self, number):
            self.number = number

        def numpy(self):
            import numpy as np

            return np.array([self.number])

    paddle = SimpleNamespace(
        set_device=lambda device: None,
        no_grad=nullcontext,
        ones=lambda *a, **k: Value(1),
        matmul=lambda a, b: Value(4),
        nn=SimpleNamespace(functional=SimpleNamespace(conv2d=lambda a, b: Value(9))),
    )
    assert module.compute_probe(paddle, 0)["passed"] is True
    paddle.matmul = lambda a, b: Value(3)
    with pytest.raises(ValueError, match="result"):
        module.compute_probe(paddle, 0)


def test_unverified_kernel_inventory_cannot_be_empty(tmp_path, monkeypatch):
    module = qualification()
    from ocr_service.profile import RuntimeSettings

    runtime = RuntimeSettings(
        accelerator="rocm",
        gpu_index=0,
        model_dirs={},
        cache_dir=tmp_path,
        qualification_manifest=tmp_path / "manifest.json",
        expected_packages={},
        expected_paddle_commit="1234567",
        image_identity="candidate",
    )
    paddle = SimpleNamespace(
        device=SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda index: "Radeon")),
        version=SimpleNamespace(full_version="3.3.1"),
    )
    monkeypatch.setattr(
        module, "inspect_device_identity", lambda *a: {"gfx_target": "gfx1030", "name": "Radeon", "gpu_index": 0}
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self: json.dumps(
            {"target_profile": {"schema": 1, "candidate_targets": ["gfx1030"]}, **{
                name: {"targets": [], "offline_status": "unverified", "files": {}}
                for name in ("paddle", "rocblas", "miopen")
            }}
        ),
    )
    monkeypatch.setattr(module, "compute_probe", lambda *args: {"passed": True})
    with pytest.raises(ValueError, match="empty"):
        module.device_identity(paddle, runtime)


@pytest.mark.parametrize(
    "native_label,recognized_formula",
    [
        ("formula", True),
        ("display_formula", True),
        ("inline_formula", True),
        ("formula_number", False),
    ],
)
def test_fresh_replay_checks_native_formula_and_reports_actual_timings(
    tmp_path, monkeypatch, native_label, recognized_formula
):
    module = qualification()
    from ocr_service.profile import RuntimeSettings

    runtime = RuntimeSettings(
        accelerator="cuda",
        gpu_index=0,
        model_dirs={},
        cache_dir=tmp_path,
        qualification_manifest=tmp_path / "manifest.json",
        expected_packages={},
        expected_paddle_commit="1234567",
        image_identity="candidate",
    )
    adapter = SimpleNamespace(
        qualify=lambda image: None,
        predict=lambda *a, **k: SimpleNamespace(blocks=[SimpleNamespace(label=native_label)]),
        evidence=[],
        text_dtype="float32",
        profile_id="profile",
        runtime_fingerprint="runtime",
        pipeline=SimpleNamespace(vl_rec_model=SimpleNamespace(infer=object())),
    )
    monkeypatch.setattr(module, "build_pipeline", lambda *a: adapter)
    monkeypatch.setattr(module, "audit_dynamic_model", lambda *a: 1)
    payload = {
        "runtime": runtime.model_dump(mode="json"),
        "smoke": str(tmp_path / "smoke.png"),
        "case": {
            "path": str(tmp_path / "case.png"),
            "kind": "formula",
            "sha256": "a" * 64,
            "required_labels": ["formula"],
        },
    }
    if not recognized_formula:
        with pytest.raises(ValueError, match="Required recognized structure absent"):
            module.replay(payload)
        return
    result = module.replay(payload)
    assert "cold_smoke_seconds" in result and "first_case_seconds" in result
    assert "cold_seconds" not in result


@pytest.fixture
def successful_qualification(tmp_path, monkeypatch):
    module = qualification()
    from ocr_service.profile import MODEL_NAMES, RuntimeSettings

    runtime = RuntimeSettings(
        accelerator="cuda",
        gpu_index=0,
        model_dirs={name: tmp_path / name for name in MODEL_NAMES},
        cache_dir=tmp_path,
        qualification_manifest=tmp_path / "manifest.json",
        expected_packages={},
        expected_paddle_commit="1234567",
        image_identity="sha256:" + "a" * 64,
    )
    cases = [
        module.QualificationCase(kind=kind, path=tmp_path / (kind + ".png"), sha256=str(i) * 64)
        for i, kind in enumerate(sorted(module.KINDS))
    ]
    monkeypatch.setattr(module, "load_cases", lambda path: cases)
    monkeypatch.setattr(
        module,
        "seal_dependencies",
        lambda **kwargs: ({"native": "version"}, {"native": "identity"}),
    )
    monkeypatch.setattr(module, "verify_dependencies", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "tree_identity", lambda path: "actual-model-hash")
    stages = []

    def stage(name, payload, directory):
        stages.append(name)
        if name == "inspect":
            assert not runtime.qualification_manifest.exists()
            return {"parameters": ["actual-inspection"]}
        if name == "collect":
            assert not runtime.qualification_manifest.exists()
            assert len(payload["inventory"]) == 2
            return {
                "static_artifacts": {"collected": "coverage"},
                "device": {"name": "actual-device", "component_precision": {key: "float32" for key in ("text", "mlp_AR", "visual", "static")}},
            }
        assert runtime.qualification_manifest.exists()
        return {
            "passed": True,
            "kind": payload["case"]["kind"],
            "profile_id": "profile",
            "runtime_fingerprint": "runtime",
            "startup_seconds": 1,
            "cold_smoke_seconds": 1,
            "first_case_seconds": 1,
            "warm_case_seconds": 1,
            "dynamic_components": {key: {"parameters": 1, "dtype": "float32"} for key in ("text", "mlp_AR", "visual")},
            "static_audits": [
                {"model_name": name, "input_count": 1, "output_count": 1, "floating_parameters": 1}
                for name in MODEL_NAMES[:-1]
            ],
        }

    monkeypatch.setattr(module, "run_stage", stage)
    return module, runtime, stages


def test_coordinator_inspects_then_collects_then_seals_then_fresh_replays(
    tmp_path, successful_qualification
):
    module, runtime, stages = successful_qualification
    report = module.qualify_runtime(
        runtime, tmp_path / "cases.toml", tmp_path / "qualification.json"
    )
    assert report.qualified and len(report.cases) == 10
    assert json.loads(runtime.qualification_manifest.read_text())["qualified_target"]["name"] == "actual-device"
    assert stages == ["inspect"] * 2 + ["collect"] + ["replay"] * 10
    assert (tmp_path / "qualification.runtime.json").exists()


def test_qualifier_rejects_architecture_override_before_native_probe(tmp_path, monkeypatch):
    module = qualification()
    from ocr_service.profile import RuntimeSettings

    runtime = RuntimeSettings(
        accelerator="rocm",
        gpu_index=0,
        model_dirs={},
        cache_dir=tmp_path,
        qualification_manifest=tmp_path / "manifest.json",
        expected_packages={},
        expected_paddle_commit="1234567",
        image_identity="candidate",
    )
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    with pytest.raises(ValueError, match="spoof"):
        module.device_identity(None, runtime)


def test_stage_timeout_reaps_child_even_if_it_exits_during_kill(tmp_path, monkeypatch):
    module = qualification()
    calls = []

    class Process:
        pid = 123456
        args = ("fake-native-stage",)

        def wait(self, timeout):
            calls.append(timeout)
            if len(calls) == 1:
                raise module.subprocess.TimeoutExpired(self.args, timeout)
            return -9

    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(module.os, "killpg", lambda *a: (_ for _ in ()).throw(ProcessLookupError()))
    with pytest.raises(TimeoutError):
        module.run_stage("collect", {}, tmp_path)
    assert calls == [600, 30]


@pytest.mark.parametrize("failed_artifact", ["runtime", "report"])
def test_publication_failure_resets_success_and_removes_partial_runtime(
    tmp_path, monkeypatch, successful_qualification, failed_artifact
):
    module, runtime, _ = successful_qualification
    output = tmp_path / "qualification.json"
    serving = output.with_suffix(".runtime.json")
    serving.write_text("stale serving settings")
    output.write_text('{"qualified": true}')
    original = module.write_json

    def fail_write(path, value):
        if path == (serving if failed_artifact == "runtime" else output):
            # A write may fail after replacement, not only before creating its file.
            original(path, value)
            raise OSError("injected publication failure")
        original(path, value)

    monkeypatch.setattr(module, "write_json", fail_write)
    report = module.qualify_runtime(runtime, tmp_path / "cases.toml", output)
    assert not report.qualified and report.qualified_targets == []
    assert report.failure == "OSError" and report.stage == "publish_" + failed_artifact
    assert not serving.exists()
    if output.exists():
        assert json.loads(output.read_text())["qualified"] is False


def test_cli_exits_nonzero_when_serving_runtime_publication_fails(
    tmp_path, monkeypatch, successful_qualification
):
    module, runtime, _ = successful_qualification
    candidate = tmp_path / "candidate.runtime.json"
    candidate.write_text(runtime.model_dump_json())
    output = tmp_path / "qualification.json"
    original = module.write_json

    def fail_write(path, value):
        if path == output.with_suffix(".runtime.json"):
            raise OSError("injected publication failure")
        original(path, value)

    monkeypatch.setattr(module, "write_json", fail_write)
    monkeypatch.setattr(module, "qualification_is_supervised", lambda descriptor: descriptor == 3)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "qualify",
            "--supervisor-fd",
            "3",
            "--accelerator",
            "cuda",
            "--runtime",
            str(candidate),
            "--cases",
            str(tmp_path / "cases.toml"),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 1
    assert json.loads(output.read_text())["qualified"] is False
    assert not output.with_suffix(".runtime.json").exists()


@pytest.mark.parametrize("status", [42, -9])
def test_qualification_only_recognizes_reserved_gpu_allocation_exit(tmp_path, monkeypatch, status):
    import subprocess

    module = qualification()
    process = SimpleNamespace(args=["stage"], wait=lambda timeout: status)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: process)
    if status == 42:
        with pytest.raises(RuntimeError, match="ocr_gpu_out_of_memory"):
            module.run_stage("collect", {}, tmp_path)
    else:
        with pytest.raises(subprocess.CalledProcessError):
            module.run_stage("collect", {}, tmp_path)


def test_collection_stage_converts_explicit_gpu_allocation_failure_to_exit(tmp_path, monkeypatch):
    import sys

    module = qualification()
    stage_input = tmp_path / "input.json"
    stage_input.write_text("{}")
    output = tmp_path / "result.json"

    def collect(_payload):
        raise MemoryError("Out of memory error on GPU 2. private native text")

    monkeypatch.setattr(module, "collect", collect)
    monkeypatch.setattr(
        sys, "argv",
        ["qualify", "--stage", "collect", "--stage-input", str(stage_input), "--output", str(output)],
    )
    with pytest.raises(SystemExit) as raised:
        module.main()
    assert raised.value.code == 42
    assert not output.exists()


def test_full_qualification_preserves_gpu_allocation_exit(tmp_path, monkeypatch):
    import sys

    module = qualification()
    runtime = module.RuntimeSettings(
        accelerator="rocm",
        gpu_index=2,
        model_dirs={},
        cache_dir=tmp_path,
        qualification_manifest=tmp_path / "manifest.json",
        expected_packages={},
        expected_paddle_commit="1234567",
        image_identity="candidate",
    )
    settings = tmp_path / "runtime.json"
    settings.write_text(runtime.model_dump_json())
    monkeypatch.setattr(module, "qualification_is_supervised", lambda _fd: True)
    report = module.QualificationReport(
        accelerator="rocm", image_identity="candidate", failure="ocr_gpu_out_of_memory"
    )
    monkeypatch.setattr(module, "qualify_runtime", lambda *_args: report)
    monkeypatch.setattr(
        sys, "argv",
        [
            "qualify", "--accelerator", "rocm", "--runtime", str(settings),
            "--cases", str(tmp_path / "cases"), "--output", str(tmp_path / "report"),
        ],
    )
    with pytest.raises(SystemExit) as raised:
        module.main()
    assert raised.value.code == 42


@pytest.mark.parametrize(
    ("stage", "source", "retryable"),
    [
        ("collect", "process_wait", True),
        ("inspect", "process_wait", False),
        ("replay", "process_wait", False),
        ("collect", "other_timeout", False),
    ],
)
def test_collection_retry_marker_requires_an_actual_collect_child_wait_timeout(
    tmp_path, monkeypatch, stage, source, retryable
):
    import sys

    from ocr_service.profile import MODEL_NAMES, RuntimeSettings

    module = qualification()
    real_run = module.run_stage
    runtime = RuntimeSettings(
        accelerator="cuda", gpu_index=0, model_dirs={name: tmp_path / name for name in MODEL_NAMES},
        cache_dir=tmp_path, qualification_manifest=tmp_path / "manifest.json", expected_packages={},
        expected_paddle_commit="1234567", image_identity="sha256:" + "a" * 64,
    )

    class Process:
        args = ("native-child",)
        pid = 63001
        waited = False

        def wait(self, timeout):
            if not self.waited:
                self.waited = True
                raise module.subprocess.TimeoutExpired(self.args, timeout)
            return -9

    def run(name, payload, directory):
        if name != stage:
            return {"static_artifacts": {}, "device": {}}
        if source == "other_timeout":
            raise TimeoutError("Coordinator timeout rather than a child wait")
        monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: Process())
        monkeypatch.setattr(module.os, "killpg", lambda *_args: None)
        return real_run(name, payload, directory)

    case = module.QualificationCase(kind="upright", path=tmp_path / "upright", sha256="a" * 64)
    monkeypatch.setattr(module, "load_cases", lambda _path: [case])
    monkeypatch.setattr(module, "seal_dependencies", lambda **kwargs: ({}, {}))
    monkeypatch.setattr(module, "verify_dependencies", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "tree_identity", lambda _path: "model-hash")
    monkeypatch.setattr(module, "run_stage", run)
    monkeypatch.setattr(module, "qualification_is_supervised", lambda _fd: True)
    settings = tmp_path / "runtime.json"
    settings.write_text(runtime.model_dump_json())
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "qualify", "--accelerator", "cuda", "--runtime", str(settings),
        "--cases", str(tmp_path / "cases"), "--output", str(output),
    ])
    with pytest.raises(SystemExit) as failed:
        module.main()
    assert failed.value.code == (44 if retryable else 1)
    report = json.loads(output.read_bytes())
    assert report["failure"] == "TimeoutError"
    assert report["collection_wait_timed_out"] is retryable
