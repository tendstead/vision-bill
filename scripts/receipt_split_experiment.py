"""Experimental local-only split extraction; never used by uploads or persistence.

Run with PYTHONPATH=src .venv/bin/python scripts/receipt_split_experiment.py --help.
Private inputs and artifacts belong under ignored server_data/. This is a spike,
not a production merger: every overlap decision requires human review.
"""

import argparse
import asyncio
import base64
import hashlib
import html
import json
import os
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Protocol, cast
from urllib.parse import urlsplit

from ollama import AsyncClient
from PIL import Image
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from vision_bill.helper.duplicates import verify_duplicate
from vision_bill.model.receipt import LineItem, Receipt
from vision_bill.model.receipt_extraction import MultiImageReceipt
from vision_bill.provider.llm.experiment import extraction_prompt
from vision_bill.service.experiment_scoring import detailed_score


class PartialObservation(Protocol):
    # Dynamic schema clones Receipt fields, retaining their original constraints.
    line_items: list[LineItem] | None
    confidence: float | None

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]: ...


# Clone each FieldInfo: nullable observations retain every original constraint.
primary_fields: dict[str, Any] = {
    name: (field.annotation, deepcopy(field))
    for name, field in LineItem.model_fields.items()
    if name not in {"spending_category", "tags"}
}
PrimaryItem = create_model("PrimaryItem", **primary_fields)


def nullable_field(name: str, field: FieldInfo) -> tuple[Any, FieldInfo]:
    cloned = deepcopy(field)
    annotation: Any = list.__class_getitem__(PrimaryItem) if name == "line_items" else field.annotation
    # Constraints belong to the non-null branch, never to the optional union.
    constrained = Annotated[annotation, *cloned.metadata] if cloned.metadata else annotation
    cloned.metadata = []
    cloned.default = None
    cloned.default_factory = None
    return constrained | None, cloned


partial_fields: dict[str, Any] = {
    name: nullable_field(name, field)
    for name, field in Receipt.model_fields.items() if name != "category"
}
PartialReceipt = create_model("PartialReceipt", **partial_fields)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def merge_parts(parts: list[BaseModel]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Suffix/prefix matches only; preserve singleton ambiguity and log conflicts."""
    payload: dict[str, Any] = {}
    decisions: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    items: list[LineItem] = []
    for index, observation in enumerate(parts):
        part = cast(PartialObservation, observation)
        values = part.model_dump(exclude_none=True)
        incoming = part.line_items or []
        overlap = 0
        for count in range(min(len(items), len(incoming)), 1, -1):
            if all(
                verify_duplicate(a, b) and a.unit_price == b.unit_price
                for a, b in zip(items[-count:], incoming[:count], strict=True)
            ):
                overlap = count
                break
        singleton = bool(items and incoming and verify_duplicate(items[-1], incoming[0]))
        decisions.append(
            {
                "part": index,
                "rows_in": len(incoming),
                "removed_prefix": overlap,
                "ambiguous_singleton_retained": singleton and not overlap,
                "requires_review": bool(overlap or singleton),
            }
        )
        items.extend(incoming[overlap:])
        for key, value in values.items():
            if key == "line_items" or (key == "taxes" and not value):
                continue
            if key in payload and payload[key] != value:
                conflicts.append(
                    {"field": key, "part": index, "previous": payload[key], "incoming": value}
                )
            # Header first; totals/taxes last. Never sum repeated receipt totals.
            if key not in payload or key in {
                "subtotal",
                "discount_total",
                "tax_total",
                "tip",
                "total",
                "taxes",
            }:
                payload[key] = value
        # Use least claimed confidence rather than the confidence of the first part.
        if part.confidence is not None:
            payload["confidence"] = min(payload.get("confidence", 100), part.confidence)
    payload["line_items"] = [item.model_dump(mode="json") for item in items]
    payload.setdefault("confidence", 0)
    return payload, {
        "boundaries": decisions,
        "conflicts": conflicts,
        "requires_review": True,
        "warning": "Experimental boundary matching does not prove physical overlap.",
    }


def partial_prompt(kind: str) -> str:
    return (
        f"Extract only what is actually visible in this {kind}, a PART of ONE receipt. "
        "Return a populated JSON object, not the schema. Fields absent from this piece MUST be null. "
        "Do not fill header/date/totals from assumptions. line_items contains ONLY fully readable "
        "purchased rows, in printed order; keep genuinely repeated purchases. A cut-off row must "
        "not be invented. Copy abbreviated descriptions literally. No categories or tags. "
        "Negative prices represent returns/refunds. discount_total is a nonnegative magnitude. "
        "Totals are only printed receipt-level totals, never calculate totals for this piece. "
        "Payment card type must not be guessed. Return only JSON matching: "
        + json.dumps(PartialReceipt.model_json_schema())
    )


def source_parts(case: dict[str, Any], mode: str, output: Path, tile_height: int) -> list[dict[str, Any]]:
    parts = []
    for photo, path in enumerate(case["images"]):
        if mode != "vision_tiles":
            parts.append({"path": path, "photo": photo})
            continue
        with Image.open(path) as image:
            height = image.height
            top = 0
            while True:
                bottom = min(top + tile_height, height)
                target = output / f"{case['id']}-photo{photo}-y{top}-{bottom}.png"
                image.crop((0, top, image.width, bottom)).save(target)
                parts.append(
                    {"path": target, "photo": photo, "crop": [0, top, image.width, bottom]}
                )
                if bottom == height:
                    break
                top = bottom - min(150, tile_height // 5)
    return parts


def ocr_parts(case: dict[str, Any], report: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for path in case["images"]:
        matches = [
            r
            for r in report
            if r["engine"] == "pp-medium"
            and Path(r["input"]).resolve() == path
            and r["status"] == "ok"
        ]
        if len(matches) != 1 or matches[0]["identity"]["input_sha256"] != digest(path.read_bytes()):
            raise ValueError("Missing PP-medium OCR with matching source hash")
        row = matches[0]
        data = json.loads(Path(row["artifact"]).read_text())
        result.append(
            {
                "text": data.get("row_text") or data["text"],
                "ocr_ms": row["elapsed_ms"],
                "artifact": row["artifact"],
            }
        )
    return result


def chunk_ocr_parts(parts: list[dict[str, Any]], rows: int) -> list[dict[str, Any]]:
    """Bound text by recognized source lines, preserving three lines of context."""
    if rows < 4:
        raise ValueError("OCR chunks need at least four rows")
    result = []
    for photo, part in enumerate(parts):
        lines = part["text"].splitlines()
        start = 0
        while start < len(lines):
            end = min(start + rows, len(lines))
            result.append(
                {
                    **part,
                    "text": "\n".join(lines[start:end]),
                    "photo": photo,
                    "row_range": [start, end],
                    "ocr_ms": part["ocr_ms"] if start == 0 else 0,
                }
            )
            if end == len(lines):
                break
            start = end - 3
    return result


async def request(client: AsyncClient, model: str, revision: str, prompt: str, images: list[Path], schema: type[BaseModel], cache: Path, timeout: float, step: dict[str, Any]) -> tuple[BaseModel, dict[str, Any], Path, bool]:
    image_bytes = [p.read_bytes() for p in images]
    identity = {
        "model": model,
        "revision": revision,
        "prompt": prompt,
        "images": [digest(data) for data in image_bytes],
        "source_sha256": digest(Path(__file__).read_bytes()),
        "temperature": 0,
        "think": False,
        "num_predict": 8192,
        "timeout_seconds": timeout,
    }
    key = digest(json.dumps(identity, sort_keys=True).encode())
    path = cache / f"{key}.json"
    step["trace"] = str(path)
    record: dict[str, Any]
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("status") == "ok":
            parsed = schema.model_validate(record["parsed"])
            if isinstance(parsed, MultiImageReceipt):
                parsed.reconciled(len(images))
            return parsed, record, path, True
    record = {"identity": identity, "attempts": [], "status": "error"}
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": prompt,
            "images": [base64.b64encode(data).decode() for data in image_bytes],
        }
    ]
    started = perf_counter()
    try:
        async with asyncio.timeout(timeout):
            for _ in range(3):
                attempt: dict[str, Any] = {"prompt": messages[-1]["content"]}
                record["attempts"].append(attempt)
                response = await client.chat(
                    model=model,
                    messages=messages,
                    think=False,
                    stream=False,
                    options={"temperature": 0, "num_predict": 8192},
                    keep_alive="1m",
                )
                content = response.message.content or ""
                attempt.update(
                    response=content,
                    telemetry={
                        name: getattr(response, name, None)
                        for name in (
                            "prompt_eval_count",
                            "eval_count",
                            "load_duration",
                            "prompt_eval_duration",
                            "eval_duration",
                            "done_reason",
                        )
                    },
                )
                try:
                    parsed = schema.model_validate_json(
                        content.strip().removeprefix("```json").removesuffix("```").strip()
                    )
                    if isinstance(parsed, MultiImageReceipt):
                        parsed.reconciled(len(images))
                    if schema is PartialReceipt and not any(
                        value is not None for value in parsed.model_dump().values()
                    ):
                        raise ValueError("Empty partial extraction")
                    record.update(status="ok", parsed=parsed.model_dump(mode="json"))
                    return parsed, record, path, False
                except ValueError as exc:
                    attempt["validation_error"] = str(exc)
                    messages.extend(
                        [
                            {"role": "assistant", "content": content},
                            {
                                "role": "user",
                                "content": f"Correct the populated JSON, not the schema. {exc}",
                            },
                        ]
                    )
            raise ValueError("Three invalid responses")
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["elapsed_ms"] = (perf_counter() - started) * 1000
        write_json(path, record)


def score(expected: Receipt, actual: Receipt) -> dict[str, Any]:
    detail = detailed_score(expected, actual, classification=False)
    payment_ok = not any(
        d["path"] == "payment_method" and not d["equivalent"] for d in detail["differences"]
    )
    header = detail["scores"]["header"]
    adjusted = detail["scores"]["normalized"] + 0.2 * ((header * 7 - payment_ok) / 6 - header)
    return {
        "adjusted": adjusted,
        "detail": detail,
        "exact_total": expected.total == actual.total,
        "same_item_count": len(expected.line_items) == len(actual.line_items),
        "ordered_item_numbers_exact": len(expected.line_items) == len(actual.line_items)
        and all(
            all(
                getattr(a, field) == getattr(b, field)
                for field in ("quantity", "unit_price", "total_price")
            )
            for a, b in zip(expected.line_items, actual.line_items, strict=True)
        ),
    }


def render(output: Path, rows: list[dict[str, Any]], cases: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        case = next(c for c in cases if c["id"] == row["case"])
        images = "".join(
            '<img style="max-width:200px;max-height:300px" src="data:image/png;base64,'
            + base64.b64encode(p.read_bytes()).decode()
            + '">'
            for p in case["images"]
        )
        traces = [
            json.loads(Path(s["trace"]).read_text())
            for s in row["steps"]
            if s.get("trace") and Path(s["trace"]).exists()
        ]
        cards.append(
            f"<article><h2>{html.escape(row['case'])} / {row['mode']}</h2>{images}"
            f"<pre>{html.escape(json.dumps(row, indent=2, ensure_ascii=False, default=str))}</pre>"
            f"<details><summary>Raw model requests and responses</summary><pre>{html.escape(json.dumps(traces, indent=2, ensure_ascii=False))}</pre></details></article>"
        )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" '
        "content=\"default-src 'none'; img-src data:; style-src 'unsafe-inline'\">"
        "<title>Private split receipt experiment</title><style>body{font:16px sans-serif;max-width:1100px;"
        "margin:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border-bottom:2px solid #888}</style>"
        + "".join(cards)
    )


async def run(args: argparse.Namespace) -> None:
    endpoint = urlsplit(args.host)
    if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Only same-machine Ollama HTTP endpoints are supported")
    os.umask(0o077)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    cache = output / "calls"
    cache.mkdir(exist_ok=True)
    (output / "crops").mkdir(exist_ok=True)
    manifest = args.dataset.resolve()
    cases = json.loads(manifest.read_text())["cases"]
    for case in cases:
        case["images"] = [(manifest.parent / p).resolve() for p in case["images"]]
        case["expected"] = (manifest.parent / case["expected"]).resolve()
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases]
    if not cases:
        raise ValueError("No selected cases")
    ocr = json.loads(args.ocr_report.read_text())["results"] if args.ocr_report else []
    client = AsyncClient(
        host=args.host, timeout=args.request_timeout, trust_env=False, follow_redirects=False
    )
    models = await client.list()
    revision = next((m.digest for m in models.models if m.model == args.model), None)
    if not revision:
        raise ValueError("Requested local model not available")
    rows = []
    for case in cases:
        for mode in args.modes:
            row: dict[str, Any] = {
                "case": case["id"],
                "mode": mode,
                "model": args.model,
                "revision": revision,
                "status": "error",
                "steps": [],
                "request_timeout": args.request_timeout,
                "job_timeout": args.request_timeout if mode.endswith("all") else args.job_timeout,
                "source_images": [
                    {"path": str(p), "sha256": digest(p.read_bytes())} for p in case["images"]
                ],
            }
            started = perf_counter()
            print(f"START {case['id']} {mode}", flush=True)
            try:
                async with asyncio.timeout(row["job_timeout"]):
                    is_ocr = mode.startswith("ocr")
                    chunks = (
                        ocr_parts(case, ocr)
                        if is_ocr
                        else source_parts(case, mode, output / "crops", args.tile_height)
                    )
                    if mode == "ocr_rows":
                        chunks = chunk_ocr_parts(chunks, args.ocr_rows)
                    row["replayed_ocr_ms"] = sum(p.get("ocr_ms", 0) for p in chunks)
                    if mode.endswith("all"):
                        prompt = extraction_prompt(len(case["images"]))
                        if is_ocr:
                            # OCR has no photo indices: one complete receipt, overlap removed by model.
                            prompt = extraction_prompt(1).replace(
                                "1 ordered photos", "ordered OCR transcripts"
                            )
                            prompt += (
                                "\nThese transcripts may overlap. Merge only overlapping rows.\n"
                            )
                            prompt += "\n\n".join(
                                f"PHOTO {i + 1}\n{p['text']}" for i, p in enumerate(chunks)
                            )
                        chunks = [{"prompt": prompt, "paths": [] if is_ocr else case["images"]}]
                    parts = []
                    for index, chunk in enumerate(chunks):
                        partial = not mode.endswith("all")
                        schema = (
                            PartialReceipt
                            if partial
                            else (
                                MultiImageReceipt
                                if len(case["images"]) > 1 and not is_ocr
                                else Receipt
                            )
                        )
                        prompt = chunk.get("prompt") or partial_prompt(
                            "OCR transcript" if is_ocr else "photo/crop"
                        )
                        if partial and is_ocr:
                            prompt += "\nOCR TEXT:\n" + chunk["text"]
                        paths = (
                            chunk["paths"]
                            if "paths" in chunk
                            else ([] if is_ocr else [chunk["path"]])
                        )
                        step: dict[str, Any] = {
                            "index": index,
                            "cached": False,
                            "elapsed_ms": 0,
                            "source": {
                                k: str(v) if isinstance(v, Path) else v
                                for k, v in chunk.items()
                                if k not in {"prompt", "text"}
                            },
                        }
                        row["steps"].append(step)
                        value, trace, _artifact, cached = await request(
                            client,
                            args.model,
                            revision,
                            prompt,
                            paths,
                            schema,
                            cache,
                            args.request_timeout,
                            step,
                        )
                        step.update(
                            cached=cached,
                            elapsed_ms=trace["elapsed_ms"],
                            parsed=value.model_dump(mode="json"),
                        )
                        parts.append(value)
                        print(
                            f"  PART {index + 1}/{len(chunks)} ok {trace['elapsed_ms'] / 1000:.1f}s cached={cached}",
                            flush=True,
                        )
                    if mode.endswith("all"):
                        first = parts[0]
                        actual = (first.reconciled(len(case["images"])) if isinstance(first, MultiImageReceipt)
                                  else Receipt.model_validate(first.model_dump()))
                    else:
                        payload, decisions = merge_parts(parts)
                        row.update(merge=decisions, candidate=payload)
                        actual = Receipt.model_validate(payload)
                    row["returned_candidate"] = actual.model_dump(mode="json")
                    # References are opened ONLY after extraction and merging.
                    expected = Receipt.model_validate_json(case["expected"].read_text())
                    row.update(status="ok", score=score(expected, actual))
            except Exception as exc:  # noqa: BLE001 -- record one failed case and continue the experiment
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["wall_ms"] = (perf_counter() - started) * 1000
            for step in row["steps"]:
                if step.get("trace") and Path(step["trace"]).exists():
                    step["elapsed_ms"] = json.loads(Path(step["trace"]).read_text())["elapsed_ms"]
            row["model_ms"] = sum(s["elapsed_ms"] for s in row["steps"])
            row["cached_steps"] = sum(s["cached"] for s in row["steps"])
            print(
                f"DONE {case['id']} {mode} {row['status']} wall={row['wall_ms'] / 1000:.1f}s score={row.get('score', {}).get('adjusted')} {row.get('error', '')}",
                flush=True,
            )
            rows.append(row)
            write_json(output / "results.json", rows)
            render(output, rows, cases)
    summaries = []
    for mode in args.modes:
        group = [r for r in rows if r["mode"] == mode]
        good = [r for r in group if r["status"] == "ok"]
        summaries.append(
            {
                "mode": mode,
                "cases": len(group),
                "valid": len(good),
                "score": sum(r["score"]["adjusted"] for r in good) / len(group),
                "exact_total": sum(r["score"]["exact_total"] for r in good),
                "same_item_count": sum(r["score"]["same_item_count"] for r in good),
                "ordered_item_numbers_exact": sum(
                    r["score"]["ordered_item_numbers_exact"] for r in good
                ),
                "completed_within_180s_uncached": sum(
                    r["wall_ms"] <= 180000 and not r["cached_steps"] for r in good
                ),
            }
        )
    write_json(output / "summary.json", summaries)
    print(json.dumps(summaries, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ocr-report", type=Path)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="gemma4-12b:latest")
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--job-timeout", type=float, default=600)
    parser.add_argument("--tile-height", type=int, default=1000)
    parser.add_argument("--ocr-rows", type=int, default=16)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["vision_all", "vision_parts", "vision_tiles", "ocr_all", "ocr_parts", "ocr_rows"],
        default=["vision_all", "vision_parts", "ocr_all", "ocr_parts"],
    )
    args = parser.parse_args()
    if (
        args.tile_height < 300
        or args.ocr_rows < 4
        or min(args.request_timeout, args.job_timeout) <= 0
    ):
        parser.error("tile-height must be >=300, ocr-rows >=4, and deadlines must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
