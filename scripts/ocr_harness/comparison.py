"""Paired local vision/OCR experiments using existing service and scoring paths."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

from vision_bill.provider.ocr.base import OcrProvider

if TYPE_CHECKING:
    from vision_bill.model.ocr import OcrCapabilities, OcrResponse
    from vision_bill.provider.llm.base import LLMProvider
    from vision_bill.provider.llm.ollama import OllamaProvider
    from vision_bill.provider.llm.openai import OpenAIProvider
    from vision_bill.service.experiment_execution import ExperimentResult

# Provider imports must stay lazy until bootstrap_settings has isolated config.

import argparse
import asyncio
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from .runner import digest, write_reports
from .worker import require_local_endpoint


def load_cases(manifest: Path) -> list[dict[str, Any]]:
    manifest = Path(manifest).resolve()
    cases = json.loads(manifest.read_text())["cases"]
    ids = [c["id"] for c in cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("Dataset must contain cases with unique IDs")
    result = []
    for case in cases:
        item = {
            **case,
            "images": [(manifest.parent / p).resolve() for p in case["images"]],
            "expected": (manifest.parent / case["expected"]).resolve(),
        }
        if not item["images"] or any(not p.is_file() for p in [*item["images"], item["expected"]]):
            raise ValueError("Each case requires existing ordered images and expected JSON")
        if "preview" in case:
            item["preview"] = (manifest.parent / case["preview"]).resolve()
        result.append(item)
    return result


def cached_ocr(case: dict[str, Any], rows: list[dict[str, Any]], engine: str) -> tuple[list[dict[str, Any]], float]:
    from vision_bill.model.ocr import OcrResponse

    pages: list[dict[str, Any]] = []
    elapsed = 0.0
    for i, image in enumerate(case["images"]):
        found = [r for r in rows if r["engine"] == engine and Path(r["input"]).resolve() == image]
        if len(found) != 1 or found[0]["status"] != "ok":
            raise ValueError(f"Missing successful {engine} OCR for photo {i + 1}")
        row = found[0]
        source_hash = digest(image.read_bytes())
        if row["identity"]["input_sha256"] != source_hash:
            raise ValueError(f"Source photo {i + 1} changed since OCR")
        raw_path = Path(row.get("raw_artifact", row["artifact"]))
        raw = raw_path.read_bytes()
        data = json.loads(raw)
        if data.get("contract_version") != 2:
            raise ValueError("Typed v2 per-photo OCR artifact required; rerun this engine")
        page = OcrResponse.model_validate(data)
        if page.source_sha256 != source_hash:
            raise ValueError("OCR artifact source changed")
        # A single-photo engine job records index zero; the dataset assigns capture order.
        page.photo_index = i
        payload = page.model_dump(mode="json")
        payload["artifact_identity"] = {
            "raw_sha256": digest(raw), "engine_identity": row["identity"],
            "recorded_photo_index": data["photo_index"],
        }
        pages.append(payload)
        elapsed += row.get("elapsed_ms", 0)
    return pages, elapsed


class ReplayOCR(OcrProvider):
    """Typed recorded pages pass through the same all-photo policy as live OCR."""

    def __init__(self, pages: list[dict[str, Any]]) -> None:
        from vision_bill.model.ocr import OcrResponse
        self.pages = [OcrResponse.model_validate({k: v for k, v in page.items()
                                                if k != "artifact_identity"}) for page in pages]

    async def check_connection(self) -> bool:
        return bool(self.pages)

    async def extract_text(self, image: Path) -> dict[str, Any]:
        raise NotImplementedError("replay requires explicit source and photo identities")

    async def get_capabilities(self) -> OcrCapabilities:
        from vision_bill.model.ocr import OcrCapabilities
        first = self.pages[0]
        return OcrCapabilities(enabled=True, configured=True, reachable=True, ready=True,
                               profile_id=first.profile_id, runtime_fingerprint=first.runtime_fingerprint,
                               ocr_max_new_tokens=first.telemetry.get("max_new_tokens"))

    async def extract_page(self, image: Path, *, photo_index: int, source_sha256: str, deadline: float, expected_profile_id: str | None = None) -> OcrResponse:
        page = self.pages[photo_index]
        if page.photo_index != photo_index or page.source_sha256 != source_sha256:
            raise ValueError("replay_source_identity_mismatch")
        return page.model_copy(deep=True)


def job_identity(target: dict[str, Any], case: dict[str, Any], pipeline: str, text: object, timeout: float, max_tokens: int) -> str:
    root = Path(__file__).resolve().parents[2]
    source = [
        *sorted((root / "src/vision_bill").rglob("*.py")),
        *sorted(Path(__file__).parent.glob("*.py")),
    ]
    return digest(
        json.dumps(
            {
                "target": target,
                "images": [digest(p.read_bytes()) for p in case["images"]],
                "expected": digest(case["expected"].read_bytes()) if case.get("expected") else None,
                "pipeline": pipeline,
                "ocr": text,
                "timeout": timeout,
                "max_tokens": max_tokens,
                "classification": False,
                "fallback": False,
                "source_projection_version": "2",
                "preprocessing": "native-v2-working-view-or-first-frame-exif-rgb",
                "source": {str(p.relative_to(root)): digest(p.read_bytes()) for p in source},
            },
            sort_keys=True,
        ).encode()
    )


def safe_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = [
                {"type": "text", "text": p.get("text", "")}
                if p.get("type") == "text"
                else {"type": "text", "text": "[image omitted; see dataset input hashes]"}
                for p in content
            ]
        result.append(
            {
                "role": message.get("role"),
                "content": content,
                "image_count": len(message.get("images", [])),
            }
        )
    return result


@contextmanager
def capture_requests(provider: OllamaProvider | OpenAIProvider, kind: str, path: Path, max_tokens: int) -> Iterator[None]:
    from vision_bill.provider.llm.ollama import OllamaProvider
    from vision_bill.provider.llm.openai import OpenAIProvider

    endpoint = (cast(OllamaProvider, provider)._client if kind == "ollama"
                else cast(OpenAIProvider, provider)._client.chat.completions)
    name = "chat" if kind == "ollama" else "create"
    original = getattr(endpoint, name)
    trace: dict[str, Any] = {"version": 1, "attempts": []}

    async def observed(*args: Any, **kwargs: Any) -> Any:
        if kind == "ollama":
            kwargs["options"] = {**kwargs.get("options", {}), "num_predict": max_tokens}
        else:
            kwargs["max_tokens"] = max_tokens
        attempt: dict[str, Any] = {"messages": safe_messages(kwargs.get("messages", [])), "max_tokens": max_tokens}
        trace["attempts"].append(attempt)
        started = perf_counter()
        try:
            response = await original(*args, **kwargs)
            attempt["response"] = (
                response.message.content
                if kind == "ollama"
                else response.choices[0].message.content
            )
            return response
        except BaseException as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            attempt["elapsed_ms"] = (perf_counter() - started) * 1000
            path.write_text(json.dumps(trace, ensure_ascii=False, indent=2))

    setattr(endpoint, name, observed)
    try:
        yield
    finally:
        setattr(endpoint, name, original)
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2))


async def compare_case(provider: LLMProvider, model: str, case: dict[str, Any], pipeline: str, text: list[dict[str, Any]] | None, directory: Path, timeout: float) -> ExperimentResult:
    from vision_bill.config import ImageSettings, PGSettings
    from vision_bill.model.experiment import ExperimentOptions, ExtractionPipeline
    from vision_bill.service.receipt_service import ReceiptService

    if pipeline not in {"ocr", "direct_vision"}:
        raise ValueError("Choose ocr or direct_vision")
    if pipeline == "ocr" and (not isinstance(text, list) or len(text) != len(case["images"])):
        raise ValueError("Typed OCR text pages for every photo are required; vision fallback is disabled")

    if pipeline == "ocr" and text and any("replay_kind" in page for page in text):
        from .historical import execute
        return await execute(provider, model, text, timeout)

    service = ReceiptService(
        ImageSettings(save_dir=str(directory / "storage"), tmp_dir=str(directory / "tmp")),
        PGSettings(user="harness", password="unused"),
        provider,
        ReplayOCR(text) if pipeline == "ocr" and text is not None else None,
    )
    options = ExperimentOptions(
        pipeline=ExtractionPipeline(pipeline),
        temperature=0,
        think=False if provider.supports_thinking_control else None,
        include_classification=False,
        allow_ocr_fallback=False,
    )
    async with asyncio.timeout(timeout):
        result = await service.analyse_submission_with_metadata(
            model, case["images"], options, []
        )
    result.metadata.update(
        execution_path="receipt_service",
        ocr_replayed=pipeline == "ocr",
        source_image_count=len(case["images"]),
        multi_photo_ocr="typed pages through common complete pipeline" if pipeline == "ocr" else None,
    )
    return result


def summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for label in dict.fromkeys(r["engine"] for r in rows):
        cases = [r for r in rows if r["engine"] == label]
        scores = []
        scoring_failed = full_correct = 0
        for row in cases:
            if row.get("llm_status") != "ok":
                continue
            try:
                score = row.get("score") or json.loads(Path(row["score_artifact"]).read_text())
                full_correct += int(bool(score.get("full_receipt_correct")))
            except (OSError, ValueError, KeyError):
                scoring_failed += 1
                continue
            payment_ok = not any(
                d["path"] == "payment_method" and not d["equivalent"] for d in score["differences"]
            )
            header = score["scores"]["header"]
            adjusted = score["scores"]["normalized"] + 0.2 * (
                (header * 7 - payment_ok) / 6 - header
            )
            scores.append(adjusted)
        result.append(
            {
                "engine": label,
                "cases": len(cases),
                "valid": len(scores),
                "extracted": sum(r.get("llm_status") == "ok" for r in cases),
                "scoring_failed": scoring_failed,
                "full_receipt_correct_rate": full_correct / len(cases),
                "score_excluding_payment": sum(scores) / len(cases),
                "score_successes_excluding_payment": sum(scores) / len(scores) if scores else None,
                "exact_totals": sum(bool(r.get("total_exact")) for r in cases),
                "same_item_count": sum(bool(r.get("item_count_exact")) for r in cases),
            }
        )
    return result


def make_target_provider(target: dict[str, Any], timeout: float) -> OllamaProvider | OpenAIProvider:
    import httpx2

    require_local_endpoint(target["host"])
    if target["provider"] == "ollama":
        from ollama import AsyncClient

        from vision_bill.provider.llm.ollama import OllamaProvider

        provider = OllamaProvider(target["host"], temperature=0, analysis_timeout_seconds=timeout)
        provider._client = AsyncClient(
            host=target["host"], timeout=timeout, trust_env=False, follow_redirects=False
        )
    elif target["provider"] == "openai":
        from openai import AsyncOpenAI

        from vision_bill.provider.llm.openai import OpenAIProvider

        openai_provider = OpenAIProvider(
            target["host"], "local-unused", temperature=0, analysis_timeout_seconds=timeout
        )
        openai_provider._client = AsyncOpenAI(
            base_url=target["host"],
            api_key="local-unused",
            max_retries=0,
            http_client=httpx2.AsyncClient(trust_env=False, follow_redirects=False, timeout=timeout),
            timeout=timeout,
        )
        return openai_provider
    else:
        raise ValueError("Supported local providers: ollama, openai")
    return provider


async def run_comparison(cases: list[dict[str, Any]], targets: list[dict[str, Any]], ocr_rows: list[dict[str, Any]], engine: str, output: Path, timeout: float = 600, max_tokens: int = 8192) -> list[dict[str, Any]]:
    from decimal import Decimal

    from vision_bill.model.receipt import Receipt
    from vision_bill.provider.llm.ollama import OllamaProvider
    from vision_bill.provider.llm.openai import OpenAIProvider
    from vision_bill.service.experiment_scoring import detailed_score

    from .historical import CONTRACT, load_replay

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    rows = []
    for target in targets:
        provider = make_target_provider(target, timeout)
        provider.max_output_tokens = max_tokens
        if target["provider"] == "ollama":
            models = await cast(OllamaProvider, provider)._client.list()
            actual_digest = next(
                (m.digest for m in models.models if m.model == target["model"]), None
            )
            if not actual_digest:
                raise ValueError(f"Model unavailable: {target['model']}")
            target = {**target, "resolved_digest": actual_digest}
        elif not target.get("revision"):
            raise ValueError(
                "OpenAI-compatible targets require an explicit weights/server revision"
            )
        for case in cases:
            for pipeline in ("direct_vision", "ocr"):
                label = target["id"] + "/" + (engine + "+llm" if pipeline == "ocr" else "vision")
                text: list[dict[str, Any]] | None = None
                ocr_ms = 0.0
                loader_error: Exception | None = None
                if pipeline == "ocr":
                    try:
                        text, ocr_ms = load_replay(case, ocr_rows, engine)
                    except (ValueError, OSError, KeyError) as exc:
                        loader_error = exc
                key = job_identity(target, case, pipeline, text, timeout, max_tokens)
                directory = output / "jobs" / key
                directory.mkdir(parents=True, exist_ok=True)
                preview = case.get("preview", case["images"][0])
                row: dict[str, Any] = {
                    "engine": label,
                    "case": case["id"],
                    "pipeline": pipeline,
                    "input": str(preview),
                    "identity": {
                        "input_sha256": digest(preview.read_bytes()),
                        "job": key,
                        "target": target,
                    },
                    "status": "error" if loader_error else "ok",
                    "cached": pipeline == "ocr" and loader_error is None,
                    "cache_mode": "ocr_replay" if pipeline == "ocr" else "live",
                    "warm_state": "unknown",
                    "artifact": str(directory / "ocr.json"),
                    "preview_note": "Display overview only; inference uses every original photo or its validated transcript in capture order.",
                    "source_images": [
                        {"path": str(p), "sha256": digest(p.read_bytes())} for p in case["images"]
                    ],
                    "elapsed_ms": ocr_ms,
                    "accuracy": None,
                    "trace_artifact": str(directory / "trace.json"),
                    "execution_path": "historical_cpu_transcript" if text and text[0].get("replay_kind") == CONTRACT else "receipt_service",
                }
                if text:
                    Path(row["artifact"]).write_text(
                        json.dumps(
                            {
                                "text": "\n\n".join(page["text"] for page in text),
                                "row_text": "\n\n".join(page["text"] for page in text),
                                "blocks": [],
                                "warnings": [
                                    "Display transcript only; inference receives all ordered pages under their declared replay contract."
                                ],
                            }
                        )
                    )
                started = perf_counter()
                try:
                    if loader_error:
                        row["error"] = str(loader_error)
                        raise loader_error
                    saved = directory / "success.json"
                    cached = json.loads(saved.read_text()) if saved.exists() else {}
                    valid_cache = bool(cached.get("artifact_hashes")) and all(
                        Path(path).is_file() and digest(Path(path).read_bytes()) == hashed
                        for path, hashed in cached.get("artifact_hashes", {}).items()
                    )
                    if valid_cache:
                        row.update(cached)
                        row["llm_cached"] = True
                        Receipt.model_validate_json(Path(row["receipt_artifact"]).read_text())
                    else:
                        with capture_requests(
                            provider, target["provider"], Path(row["trace_artifact"]), max_tokens
                        ):
                            result = await compare_case(
                                provider, target["model"], case, pipeline, text, directory, timeout
                            )
                        actual = result.receipt
                        receipt_path = directory / "receipt.json"
                        receipt_path.write_text(actual.model_dump_json(indent=2))
                        row["llm_status"] = "ok"
                        expected = Receipt.model_validate_json(case["expected"].read_text())
                        score = detailed_score(expected, actual, classification=False)
                        score_path = directory / "score.json"
                        score_path.write_text(json.dumps(score, indent=2))
                        (directory / "metadata.json").write_text(
                            json.dumps(result.metadata, indent=2)
                        )
                        row.update(
                            metadata=result.metadata,
                            llm_status="ok",
                            llm_cached=False,
                            receipt_artifact=str(receipt_path),
                            score_artifact=str(score_path),
                            accuracy=score["scores"]["normalized"],
                            total_exact=Decimal(actual.total) == Decimal(expected.total),
                            item_count_exact=len(actual.line_items) == len(expected.line_items),
                            llm_elapsed_ms=(perf_counter() - started) * 1000,
                        )
                        row["artifact_hashes"] = {str(path): digest(path.read_bytes()) for path in
                            (receipt_path, score_path, directory / "metadata.json", Path(row["trace_artifact"]))}
                        saved.write_text(json.dumps(row, indent=2))
                except Exception as exc:  # noqa: BLE001 - record failures and continue the matrix
                    if hasattr(exc, "metadata"):
                        row["metadata"] = exc.metadata
                    row.update(
                        score_status="error" if row.get("llm_status") == "ok" else "not_run",
                        llm_status=row.get("llm_status", "error"),
                        llm_error=f"{type(exc).__name__}: {exc}",
                        llm_elapsed_ms=(perf_counter() - started) * 1000,
                    )
                row["job_wall_ms"] = (perf_counter() - started) * 1000
                rows.append(row)
                write_reports(output, rows)
                (output / "summary.json").write_text(json.dumps(summary(rows), indent=2))
                print(
                    f"{label} {case['id']}: {row['llm_status']} cached={row.get('llm_cached', False)}",
                    flush=True,
                )
        if target["provider"] == "ollama":
            await cast(OllamaProvider, provider)._client._client.aclose()
        else:
            await cast(OpenAIProvider, provider)._client.close()
    return rows


def bootstrap_settings(output: Path) -> Path:
    """Satisfy config imports without initializing the application's config file."""
    import secrets

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = output / "bootstrap-settings.json"
    path.write_text(
        json.dumps(
            {
                "llm": {
                    "provider": "ollama",
                    "host": "http://127.0.0.1:1",
                    "api_key": "unused",
                    "model_name": "benchmark-unused",
                    "temperature": 0,
                },
                "pg": {"user": "harness", "password": "unused", "host": "127.0.0.1", "port": 1},
                "auth": {"secret_key": secrets.token_hex(32)},
            }
        )
    )
    path.chmod(0o600)
    os.environ["VISION_BILL_CONFIG_PATH"] = str(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--ocr-report", type=Path, required=True)
    parser.add_argument("--ocr-engine", default="pp-medium")
    parser.add_argument(
        "--output", type=Path, default=Path("server_data/ocr-harness/paired/results")
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.limit < 1 or args.timeout <= 0 or args.max_tokens < 1:
        parser.error("Limits must be positive")
    os.umask(0o077)
    logging.disable(logging.CRITICAL)
    cases = load_cases(args.dataset)[: args.limit]
    targets = json.loads(args.targets.read_text())["targets"]
    if not targets or len({t["id"] for t in targets}) != len(targets):
        parser.error("Targets require unique IDs")
    bootstrap_settings(args.output)
    rows = asyncio.run(
        run_comparison(
            cases,
            targets,
            json.loads(args.ocr_report.read_text())["results"],
            args.ocr_engine,
            args.output,
            args.timeout,
            args.max_tokens,
        )
    )
    return int(any(r["llm_status"] != "ok" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
