import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

PROFILE = Path(__file__).parents[1] / "paddleocr-service/profile.json"


def test_profile_is_exact():
    from ocr_service.profile import effective_pipeline_config, load_profile

    profile = load_profile(PROFILE)
    assert (profile.min_pixels, profile.max_pixels, profile.max_new_tokens) == (
        112896,
        2000000,
        4096,
    )
    assert profile.use_doc_orientation_classify is True
    assert profile.use_doc_unwarping is False
    assert profile.use_queues is False
    config = effective_pipeline_config(profile)
    assert config["use_doc_preprocessor"] is True
    assert config["SubPipelines"]["DocPreprocessor"]["use_doc_unwarping"] is False
    assert "DocUnwarping" not in config["SubPipelines"]["DocPreprocessor"]["SubModules"]
    assert config["SubPipelines"]["DocPreprocessor"]["use_doc_orientation_classify"] is True
    assert config["SubModules"]["VLRecognition"]["genai_config"] == {"backend": "native"}
    assert config["SubModules"]["LayoutDetection"]["model_name"] == "PP-DocLayoutV3"

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "batch_size":
                    assert item == 1
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(config)
    config["SubModules"]["LayoutDetection"]["batch_size"] = 100
    assert effective_pipeline_config(profile)["SubModules"]["LayoutDetection"]["batch_size"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_pixels", 1605632),
        ("precision", "bfloat16"),
        ("use_doc_unwarping", True),
        ("use_doc_orientation_classify", False),
        ("use_queues", True),
    ],
)
def test_profile_rejects_adaptive_changes(tmp_path, field, value):
    from ocr_service.profile import load_profile

    data = json.loads(PROFILE.read_text())
    data[field] = value
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValidationError):
        load_profile(path)


def test_runtime_only_supports_explicit_native_accelerators(tmp_path):
    from ocr_service.profile import RuntimeSettings

    with pytest.raises(ValidationError):
        RuntimeSettings(accelerator="cpu")


def test_identity_separates_effective_content_from_device():
    from ocr_service.profile import content_identity, runtime_identity

    config = {"a": 1, "b": {"c": 2}}
    assert content_identity(config) == content_identity({"b": {"c": 2}, "a": 1})
    changed = copy.deepcopy(config)
    changed["b"]["c"] = 3
    assert content_identity(config) != content_identity(changed)
    assert runtime_identity("profile", {"device": "a"}) != runtime_identity(
        "profile", {"device": "b"}
    )


def test_dependency_identity_hashes_installed_content_and_ignores_bytecode(tmp_path, monkeypatch):
    from ocr_service import profile

    code = tmp_path / "module.py"
    code.write_text("version = 1")
    bytecode = tmp_path / "module.pyc"
    bytecode.write_bytes(b"generated")
    distribution = SimpleNamespace(
        files=[Path("module.py"), Path("module.pyc")], locate_file=lambda path: tmp_path / path
    )
    monkeypatch.setattr(profile.importlib.metadata, "distribution", lambda name: distribution)
    first = profile.dependency_identity("package")
    bytecode.write_bytes(b"changed")
    assert profile.dependency_identity("package") == first
    code.write_text("version = 2")
    assert profile.dependency_identity("package") != first


@pytest.fixture
def native_dependencies(monkeypatch):
    from ocr_service import profile

    versions = {
        "paddleocr": "3.7.0",
        "paddlex": "3.7.2",
        "paddlepaddle-gpu": "3.3.1",
        "pillow": "12.3.0",
        "pillow-heif": "1.7.0",
        "numpy": "2.2.6",
        "pydantic": "2.13.4",
        "packaging": "26.2",
    }
    requirements = {name: [] for name in versions}
    contents = {name: "a" * 64 for name in versions}
    monkeypatch.setattr(
        profile.importlib.metadata,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": "paddlepaddle-gpu"})],
    )
    monkeypatch.setattr(profile.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(profile.importlib.metadata, "requires", requirements.__getitem__)
    monkeypatch.setattr(profile, "dependency_identity", contents.__getitem__)
    return versions, requirements, contents


def sealed_dependencies(versions, contents):
    return {
        name: {"version": version, "content_sha256": contents[name]}
        for name, version in versions.items()
    }


@pytest.mark.parametrize(
    "omitted",
    [
        "pillow",
        "pillow-heif",
        "numpy",
        "pydantic",
        "packaging",
        "paddleocr",
        "paddlex",
        "paddlepaddle-gpu",
    ],
)
def test_dependency_contract_requires_every_execution_root(native_dependencies, omitted):
    from ocr_service.profile import verify_dependencies

    versions, _, contents = native_dependencies
    expected = dict(versions)
    expected.pop(omitted)
    with pytest.raises(ValueError, match="required|Paddle"):
        verify_dependencies(expected, sealed_dependencies(expected, contents))


def test_dependency_contract_rejects_two_paddle_distributions(native_dependencies):
    from ocr_service.profile import verify_dependencies

    versions, _, contents = native_dependencies
    versions["paddlepaddle"] = "3.3.1"
    contents["paddlepaddle"] = "a" * 64
    with pytest.raises(ValueError, match="Paddle"):
        verify_dependencies(versions, sealed_dependencies(versions, contents))


@pytest.mark.parametrize("drift", ["version", "content"])
@pytest.mark.parametrize("package", ["pillow", "pydantic-core"])
def test_dependency_contract_rejects_installed_drift(native_dependencies, drift, package):
    from ocr_service.profile import verify_dependencies

    versions, requirements, contents = native_dependencies
    if package == "pydantic-core":
        versions[package] = "2.46.0"
        contents[package] = "a" * 64
        requirements[package] = []
        requirements["pydantic"] = ["pydantic-core==2.46.0"]
    expected = dict(versions)
    sealed = sealed_dependencies(expected, contents)
    if drift == "version":
        versions[package] = "99.0"
    else:
        contents[package] = "b" * 64
    with pytest.raises(ValueError, match="identity"):
        verify_dependencies(expected, sealed)


def test_dependency_contract_requires_matching_sealed_version(native_dependencies):
    from ocr_service.profile import verify_dependencies

    versions, _, contents = native_dependencies
    sealed = sealed_dependencies(versions, contents)
    sealed["numpy"]["version"] = "1.0"
    with pytest.raises(ValueError, match="identity"):
        verify_dependencies(versions, sealed)


def test_dependency_graph_activates_extras_markers_and_handles_cycles(native_dependencies):
    from ocr_service.profile import verify_dependencies

    versions, requirements, contents = native_dependencies
    versions.update({"layout-extra": "1.0", "leaf": "2.0"})
    contents.update({"layout-extra": "b" * 64, "leaf": "c" * 64})
    requirements["paddlex"] = [
        "Layout_Extra[fast]>=1; extra == 'ocr'",
        "never-activated; extra == 'other'",
        "obsolete; python_version < '2.0'",
    ]
    requirements["layout-extra"] = ["leaf>=2; extra == 'fast' and python_version >= '3.12'"]
    requirements["leaf"] = ["paddlex[ocr]==3.7.2"]
    actual_versions, actual_contents = verify_dependencies(
        versions, sealed_dependencies(versions, contents)
    )
    assert actual_versions == versions
    assert actual_contents == contents
    assert "never-activated" not in actual_versions
    assert "obsolete" not in actual_versions


@pytest.mark.parametrize("omitted_from", ["expected", "sealed"])
def test_active_transitive_dependency_cannot_be_omitted(native_dependencies, omitted_from):
    from ocr_service.profile import verify_dependencies

    versions, requirements, contents = native_dependencies
    versions["pydantic-core"] = "2.46.0"
    contents["pydantic-core"] = "b" * 64
    requirements["pydantic"] = ["pydantic-core==2.46.0"]
    requirements["pydantic-core"] = []
    sealed = sealed_dependencies(versions, contents)
    if omitted_from == "expected":
        versions.pop("pydantic-core")
    else:
        sealed.pop("pydantic-core")
    with pytest.raises(ValueError, match="dependency"):
        verify_dependencies(versions, sealed)


def test_dependency_contract_rejects_unrelated_manifest_entries(native_dependencies):
    from ocr_service.profile import verify_dependencies

    versions, requirements, contents = native_dependencies
    versions["unrelated-application"] = "7.0"
    requirements["unrelated-application"] = []
    contents["unrelated-application"] = "a" * 64
    with pytest.raises(ValueError, match="closure"):
        verify_dependencies(versions, sealed_dependencies(versions, contents))


def test_dependency_graph_rejects_versions_outside_active_requirements(native_dependencies):
    from ocr_service.profile import verify_dependencies

    versions, requirements, contents = native_dependencies
    requirements["paddleocr"] = ["paddlex>=4"]
    with pytest.raises(ValueError, match="requirement"):
        verify_dependencies(versions, sealed_dependencies(versions, contents))


def test_dependency_graph_revisits_node_when_later_extras_are_activated(native_dependencies):
    from ocr_service.profile import verify_dependencies

    versions, requirements, contents = native_dependencies
    versions.update({"shared": "1.0", "bridge": "1.0", "leaf": "1.0"})
    contents.update({name: "b" * 64 for name in ["shared", "bridge", "leaf"]})
    requirements["pydantic"] = ["shared"]  # Visited before paddlex's extras branch.
    requirements["paddlex"] = ["bridge; extra == 'ocr'"]
    requirements["bridge"] = ["shared[fast]"]
    requirements["shared"] = ["leaf; extra == 'fast'"]
    requirements["leaf"] = []
    actual_versions, _ = verify_dependencies(versions, sealed_dependencies(versions, contents))
    assert actual_versions == versions


def test_dependency_contract_canonicalizes_names(native_dependencies):
    from ocr_service.profile import verify_dependencies

    versions, _, contents = native_dependencies
    expected = dict(versions)
    expected["Pillow"] = expected.pop("pillow")
    sealed = sealed_dependencies(versions, contents)
    sealed["pillow_heif"] = sealed.pop("pillow-heif")
    actual_versions, _ = verify_dependencies(expected, sealed)
    assert actual_versions == versions
