"""Bounded subprocess runner, content-addressed cache and portable reports."""

import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .common import normalize_result

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("worker.py")
CONTRACT_VERSION = 1
DEFAULT_LIMITS = {"cpus": "4", "memory": "8g"}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def model_identity(models):
    extensions = {
        ".onnx",
        ".traineddata",
        ".pt",
        ".pth",
        ".pdiparams",
        ".pdparams",
        ".pdmodel",
        ".safetensors",
        ".json",
        ".yaml",
        ".yml",
        ".txt",
    }
    files = {
        str(p.relative_to(models)): digest(p.read_bytes())
        for p in sorted(models.rglob("*"))
        if p.is_file() and p.suffix in extensions and ".cache" not in p.relative_to(models).parts
    }
    return digest(json.dumps(files, sort_keys=True).encode())


def bounded_run(argv, timeout, log):
    with Path(log).open("wb") as stream:
        proc = subprocess.Popen(
            argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = proc.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise
    if code:
        raise RuntimeError(f"Process exited {code}; see {log}")


def image_identity(engine):
    if "image" not in engine:
        # Include scripts for custom command adapters, not merely their filenames.
        files = {
            s: digest(Path(s).read_bytes()) for s in engine.get("command", []) if Path(s).is_file()
        }
        return {"command_files": files}
    result = subprocess.run(
        ["docker", "image", "inspect", engine["image"], "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    return {"docker_image": result.stdout.strip()}


def execute(engine, image, target, models, timeout, log):
    container = None
    if "command" in engine:
        replacements = {"{image}": str(image), "{result}": str(target)}
        argv = [replacements.get(arg, arg) for arg in engine["command"]]
    elif engine["family"] == "http":
        argv = [
            sys.executable,
            str(WORKER),
            "http",
            "--options",
            json.dumps(engine["options"]),
            "--image",
            str(image),
            "--result",
            str(target),
        ]
    else:
        container = "vb-ocr-test-" + uuid.uuid4().hex[:16]
        argv = [
            "docker",
            "run",
            "--rm",
            "--name",
            container,
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--cpus",
            str(engine.get("limits", DEFAULT_LIMITS)["cpus"]),
            "--memory",
            engine.get("limits", DEFAULT_LIMITS)["memory"],
            "--pids-limit",
            "256",
            "-e",
            "HOME=/models",
            "-e",
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True",
            "-e",
            "OMP_NUM_THREADS=4",
            "-e",
            "DOCTR_CACHE_DIR=/models/doctr",
            "-e",
            "TORCH_HOME=/models/torch",
            "--mount",
            f"type=bind,src={models},dst=/models",
            "--mount",
            f"type=bind,src={image.parent},dst=/inputs,readonly",
            "--mount",
            f"type=bind,src={target.parent},dst=/results",
            "--mount",
            f"type=bind,src={WORKER},dst=/worker.py,readonly",
            "--entrypoint",
            "python",
            engine["image"],
            "/worker.py",
            engine["family"],
            "--options",
            json.dumps(engine.get("options", {})),
            "--image",
            "/inputs/" + image.name,
            "--result",
            "/results/" + target.name,
        ]
    try:
        bounded_run(argv, timeout, log)
    finally:
        if container:
            subprocess.run(
                ["docker", "rm", "-f", container], capture_output=True, timeout=15, check=False
            )


def run_matrix(engines, images, output, timeout=120, force=False, progress=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    models = output.parent / "models"
    models.mkdir(exist_ok=True)
    weights = model_identity(models)
    rows = []
    identities = {}
    for engine in engines:
        try:
            identities[engine["id"]] = image_identity(engine)
        except (OSError, subprocess.SubprocessError) as exc:
            identities[engine["id"]] = {"unavailable": str(exc)}
        for path in images:
            image = Path(path).resolve()
            identity = {
                "engine": engine,
                "runtime": identities[engine["id"]],
                "models": weights,
                "limits": engine.get("limits", DEFAULT_LIMITS),
                "input_sha256": digest(image.read_bytes()),
                "worker_sha256": digest(WORKER.read_bytes()),
                "normalizer_sha256": digest(Path(__file__).with_name("common.py").read_bytes()),
                "contract": CONTRACT_VERSION,
            }
            key = digest(json.dumps(identity, sort_keys=True).encode())
            job = output / "jobs" / key
            job.mkdir(parents=True, exist_ok=True)
            record = job / "record.json"
            row = {
                "engine": engine["id"],
                "input": str(image),
                "identity": identity,
                "status": "error",
                "cached": False,
                "accuracy": None,
                "artifact": str(job / "ocr.json"),
                "error": None,
            }
            start = time.monotonic()
            try:
                if record.exists() and not force:
                    saved = json.loads(record.read_text())
                    normalize_result(json.loads((job / "ocr.json").read_text()))
                    row.update(saved)
                    row["input"] = str(image)
                    row["cached"] = True
                else:
                    if "unavailable" in identity["runtime"]:
                        raise RuntimeError(identity["runtime"]["unavailable"])
                    raw = job / "raw.json"
                    raw.unlink(missing_ok=True)
                    execute(engine, image, raw, models, timeout, job / "worker.log")
                    if raw.stat().st_size > 8_000_000:
                        raise ValueError("OCR response exceeds 8 MB compact-result limit")
                    result = normalize_result(json.loads(raw.read_text()))
                    (job / "ocr.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
                    row.update(
                        status="ok",
                        elapsed_ms=round((time.monotonic() - start) * 1000, 2),
                        characters=len(result["text"]),
                        blocks=len(result["blocks"]),
                        warnings=result["warnings"],
                        versions=result["versions"],
                    )
                    record.write_text(json.dumps(row, indent=2))
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                row.update(
                    status="timeout" if isinstance(exc, subprocess.TimeoutExpired) else "error",
                    error=f"{type(exc).__name__}: {exc}",
                    elapsed_ms=round((time.monotonic() - start) * 1000, 2),
                )
            rows.append(row)
            write_reports(output, rows)
            if progress:
                print(
                    f"{row['engine']}: {image.name}: {row['status']} "
                    f"{row.get('elapsed_ms', 0) / 1000:.2f}s cached={row['cached']}",
                    flush=True,
                )
    return rows


def write_reports(output, rows):
    (output / "report.json").write_text(
        json.dumps({"contract_version": CONTRACT_VERSION, "results": rows}, indent=2)
    )
    fields = [
        "engine",
        "input",
        "status",
        "elapsed_ms",
        "characters",
        "blocks",
        "cached",
        "error",
        "llm_status",
        "llm_cached",
        "accuracy",
        "llm_error",
    ]
    with (output / "report.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    text = [
        "# OCR comparison",
        "",
        "Timing includes worker/container and model startup. Cached rows retain original timing.",
        "No accuracy score is claimed without expected data. See JSON artifacts for text/boxes and warnings.",
        "",
        "| Engine | Image | Status | Seconds | Characters | Cached | JSON stage | Score |",
        "|---|---|---|---:|---:|---|---|---:|",
    ]
    for row in rows:
        name = Path(row["input"]).name.replace("|", "\\|")
        text.append(
            f"| {row['engine']} | {name} | {row['status']} | {row.get('elapsed_ms', 0) / 1000:.2f} | "
            f"{row.get('characters', '')} | {row['cached']} | {row.get('llm_status', '')} | {row.get('accuracy', '')} |"
        )
    (output / "report.md").write_text("\n".join(text) + "\n")
    from .step_report import render_report

    render_report(output, rows)
