import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from .runner import WORKER, bounded_run, run_matrix


def load_engines(path, only=None):
    engines = json.loads(Path(path).read_text())
    ids = [e["id"] for e in engines]
    if len(ids) != len(set(ids)):
        raise ValueError("Engine IDs must be unique")
    if only:
        selected = only.split(",")
        if set(selected) - set(ids):
            raise ValueError(f"Unknown engines: {set(selected) - set(ids)}")
        engines = [e for e in engines if e["id"] in selected]
    else:
        engines = [e for e in engines if e.get("enabled", True)]
    if not engines:
        raise ValueError("No engines selected")
    return engines


def prepare(engines, work, timeout):
    models = work / "models"
    models.mkdir(parents=True, exist_ok=True)
    records = []
    for engine in engines:
        if "command" in engine:
            records.append(
                {"engine": engine["id"], "status": "ok", "note": "Externally managed command"}
            )
            continue
        log = work / f"prepare-{engine['id']}.log"
        container = "vb-ocr-prepare-" + uuid.uuid4().hex[:16]
        try:
            if engine["family"] == "http":
                argv = [
                    sys.executable,
                    str(WORKER),
                    "http",
                    "--options",
                    json.dumps(engine["options"]),
                ]
            else:
                argv = [
                    "docker",
                    "run",
                    "--rm",
                    "--name",
                    container,
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "--cpus",
                    "4",
                    "--memory",
                    "4g",
                    "-e",
                    "HOME=/models",
                    "-e",
                    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True",
                    "-e",
                    "DOCTR_CACHE_DIR=/models/doctr",
                    "-e",
                    "TORCH_HOME=/models/torch",
                    "-e",
                    "OMP_NUM_THREADS=4",
                    "--mount",
                    f"type=bind,src={models},dst=/models",
                    "--mount",
                    f"type=bind,src={WORKER},dst=/worker.py,readonly",
                    "--entrypoint",
                    "python",
                    engine["image"],
                    "/worker.py",
                    engine["family"],
                    "--options",
                    json.dumps(engine.get("options", {})),
                ]
            bounded_run(argv, timeout, log)
            record = {"engine": engine["id"], "status": "ok"}
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            record = {"engine": engine["id"], "status": "error", "error": str(exc), "log": str(log)}
        finally:
            if engine["family"] != "http":
                subprocess.run(
                    ["docker", "rm", "-f", container], capture_output=True, timeout=15, check=False
                )
        records.append(record)
        (work / "preparation.json").write_text(json.dumps(records, indent=2))
        print(f"prepare {record['engine']}: {record['status']} ({log})", flush=True)
    return records


def build(engines, work, timeout):
    seen = set()
    errors = []
    for engine in engines:
        if "target" not in engine or engine["image"] in seen:
            continue
        seen.add(engine["image"])
        log = work / f"build-{engine['target']}.log"
        try:
            if engine["target"] == "paddle":
                probe = subprocess.run(
                    ["docker", "image", "inspect", "vision-bill-paddleocr-service:latest"],
                    capture_output=True,
                    check=False,
                )
                if probe.returncode:
                    bounded_run(
                        [
                            "docker",
                            "build",
                            "-t",
                            "vision-bill-paddleocr-service:latest",
                            str(WORKER.parents[2] / "paddleocr-service"),
                        ],
                        timeout,
                        work / "build-paddle-base.log",
                    )
            bounded_run(
                [
                    "docker",
                    "build",
                    "--target",
                    engine["target"],
                    "-t",
                    engine["image"],
                    str(WORKER.parent),
                ],
                timeout,
                log,
            )
            print(f"build {engine['image']}: ok", flush=True)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            errors.append(str(exc))
            print(f"build {engine['image']}: error; see {log}", flush=True)
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Offline, cached OCR comparison for receipt images"
    )
    parser.add_argument("action", choices=["build", "prepare", "run", "all", "report"])
    parser.add_argument("--engines", type=Path, default=WORKER.with_name("engines.json"))
    parser.add_argument("--only", help="Comma-separated engine IDs")
    parser.add_argument("--work-dir", type=Path, default=Path("server_data/ocr-harness"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120, help="Seconds per OCR engine/image")
    parser.add_argument("--setup-timeout", type=float, default=900)
    parser.add_argument("--force", action="store_true", help="Rerun successful cached OCR jobs")
    parser.add_argument(
        "--refresh-llm",
        action="store_true",
        help="Rerun JSON extraction and capture raw responses; OCR remains cached",
    )
    parser.add_argument(
        "--llm-host", help="Optional local Ollama endpoint; no LLM calls by default"
    )
    parser.add_argument("--llm-model", default="gemma4-12b:latest")
    args = parser.parse_args()
    if args.timeout <= 0 or args.setup_timeout <= 0 or args.limit < 1:
        parser.error("timeouts and limit must be positive")
    os.umask(0o077)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.action == "report":
        from .runner import write_reports

        saved = json.loads((work / "results" / "report.json").read_text())
        write_reports(work / "results", saved["results"])
        print(f"Step-by-step report: {work / 'results' / 'report.html'}")
        return 0
    engines = load_engines(args.engines, args.only)
    if args.action in ("build", "all"):
        errors = build(engines, work, args.setup_timeout)
        if args.action == "build":
            return bool(errors)
    if args.action in ("prepare", "all"):
        records = prepare(engines, work, args.setup_timeout)
        if args.action == "prepare":
            return any(r["status"] != "ok" for r in records)
    if args.data_dir is None or not args.data_dir.is_dir():
        parser.error("run/all requires an existing --data-dir")
    images = sorted(
        p
        for p in args.data_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    )[: args.limit]
    if not images:
        parser.error("No receipt images found")
    rows = run_matrix(engines, images, work / "results", args.timeout, args.force, progress=True)
    if args.llm_host:
        import asyncio

        from .extraction import extract_results

        try:
            asyncio.run(
                extract_results(
                    rows, work / "results", args.llm_host, args.llm_model, refresh=args.refresh_llm
                )
            )
        except Exception as exc:  # noqa: BLE001 - preserve OCR report on model-stage setup failure
            from .runner import write_reports

            for row in rows:
                if row["status"] == "ok":
                    row.update(llm_status="error", llm_error=f"{type(exc).__name__}: {exc}")
            write_reports(work / "results", rows)
    print(f"Reports: {work / 'results' / 'report.html'}", flush=True)
    return any(r["status"] != "ok" or r.get("llm_status") == "error" for r in rows)


if __name__ == "__main__":
    sys.exit(main())
