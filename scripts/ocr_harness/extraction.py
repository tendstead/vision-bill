"""Optional local Ollama stage; OCR-only execution does not import app dependencies."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ollama import AsyncClient

    from vision_bill.model.receipt import Receipt
    from vision_bill.provider.llm.base import LLMProvider
    from vision_bill.provider.llm.ollama import OllamaProvider

import asyncio
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from .runner import digest, write_reports


def score_expected(path: Path, receipt: Receipt) -> dict[str, Any]:
    from vision_bill.model.receipt import Receipt
    from vision_bill.service.experiment_scoring import detailed_score

    return detailed_score(
        Receipt.model_validate_json(Path(path).read_text()), receipt, classification=False
    )


def score_path(directory: Path, expected: Path) -> Path:
    return Path(directory) / ("score-" + digest(Path(expected).read_bytes()) + ".json")


def make_provider(host: str) -> OllamaProvider:
    import httpx
    from ollama import AsyncClient

    from vision_bill.model.experiment import ExperimentOptions
    from vision_bill.provider.llm.ollama import OllamaProvider

    provider = OllamaProvider(host, temperature=0, analysis_timeout_seconds=600)
    # Never route receipt text through HTTP(S)_PROXY, even for a LAN hostname.
    provider._client = AsyncClient(
        host=host, timeout=httpx.Timeout(600, connect=5), trust_env=False, follow_redirects=False
    )
    return cast(OllamaProvider, provider.for_experiment(
        ExperimentOptions(
            temperature=0, think=False, include_classification=False, allow_ocr_fallback=False
        )
    ))


@contextmanager
def capture_chat(client: AsyncClient, path: Path) -> Iterator[None]:
    original = client.chat
    trace: dict[str, Any] = {"version": 1, "attempts": []}
    path.write_text(json.dumps(trace))

    async def observed(*args: Any, **kwargs: Any) -> Any:
        attempt: dict[str, Any] = {
            "messages": [
                {"role": m.get("role"), "content": m.get("content")}
                for m in kwargs.get("messages", [])
            ]
        }
        trace["attempts"].append(attempt)
        started = perf_counter()
        try:
            response = await original(*args, **kwargs)
            attempt["response"] = response.message.content
            return response
        except Exception as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            attempt["elapsed_ms"] = (perf_counter() - started) * 1000
            path.write_text(json.dumps(trace, ensure_ascii=False, indent=2))

    method = "chat"
    setattr(client, method, observed)
    try:
        yield
    finally:
        setattr(client, method, original)


async def service_extract(provider: LLMProvider, model: str, row: dict[str, Any], text: str, directory: Path) -> tuple[Receipt, dict[str, Any]]:
    from .comparison import compare_case
    from .historical import load_replay
    case = {"images": [Path(row["input"]).resolve()]}
    pages, _ = load_replay(case, [row], row["engine"])
    result = await compare_case(provider, model, case, "ocr", pages, directory, 600)
    return result.receipt, result.metadata


async def extract_results(rows: list[dict[str, Any]], output: Path, host: str, model: str, refresh: bool = False) -> None:
    from .worker import require_local_endpoint

    require_local_endpoint(host)
    # Isolated settings file, not the deployed configuration or database.
    os.environ.setdefault("VISION_BILL_CONFIG_PATH", str(output.parent / "harness-config.yaml"))
    from vision_bill.model.receipt import Receipt

    logging.disable(logging.CRITICAL)  # Receipt text stays in private artifacts, not terminal logs.
    provider = make_provider(host)
    tags = await provider._client.list()
    model_digest = next((m.digest for m in tags.models if m.model == model), None)
    if not model_digest:
        raise ValueError(f"Cannot resolve local model digest for {model}")
    for row in rows:
        if row["status"] != "ok":
            continue
        try:
            ocr = json.loads(Path(row["artifact"]).read_text())
            text = ocr["row_text"]
            from .comparison import job_identity
            from .historical import load_replay
            case = {"images": [Path(row["input"]).resolve()]}
            pages, _ = load_replay(case, [row], row["engine"])
            identity = {"host": host, "model": model, "digest": model_digest, "pages": pages}
            key = job_identity({"host": host, "model": model, "digest": model_digest},
                               case, "ocr", pages, 600, provider.max_output_tokens)
            directory = output / "extractions" / key
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / "receipt.json"
            cache_path = directory / "cache.json"
            cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
            valid = bool(cache) and all(Path(p).is_file() and digest(Path(p).read_bytes()) == hashed
                                       for p, hashed in cache.items())
            row["llm_cached"] = valid and not refresh
            if valid and not refresh:
                receipt = Receipt.model_validate_json(target.read_text())
                metadata = json.loads((directory / "metadata.json").read_text())
            else:
                target.unlink(missing_ok=True)
                row["llm_cached"] = False
                provider.experiment_trace = []
                row["trace_artifact"] = str(directory / "trace.json")
                with capture_chat(provider._client, directory / "trace.json"):
                    async with asyncio.timeout(600):
                        receipt, metadata = await service_extract(
                            provider, model, row, text, directory
                        )
                target.write_text(receipt.model_dump_json(indent=2))
                (directory / "metadata.json").write_text(
                    json.dumps({"identity": identity, **metadata}, indent=2)
                )
                cache_path.write_text(json.dumps({str(p): digest(p.read_bytes()) for p in
                    (target, directory / "metadata.json", directory / "trace.json")}))
            row["metadata"] = metadata
            row["cache_mode"] = "ocr_replay"
            row["execution_path"] = metadata.get("execution_path", "provider")
            if (directory / "trace.json").exists():
                row["trace_artifact"] = str(directory / "trace.json")
            row.update(llm_status="ok", receipt_artifact=str(target))
            expected = Path(row["input"]).with_suffix(".json")
            if expected.exists():
                score = score_expected(expected, receipt)
                score_file = score_path(directory, expected)
                score_file.write_text(json.dumps(score, indent=2))
                row["accuracy"] = score["scores"]["normalized"]
                row["score_artifact"] = str(score_file)
            else:
                row["accuracy"] = None
        except Exception as exc:  # noqa: BLE001 - record provider failures and continue the matrix
            row.update(llm_status="error", llm_error=f"{type(exc).__name__}: {exc}")
        write_reports(output, rows)
        print(
            f"Gemma stage {row['engine']}: {row['llm_status']} cached={row.get('llm_cached', False)}",
            flush=True,
        )
