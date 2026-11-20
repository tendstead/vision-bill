#!/usr/bin/env python3
"""Estimate (really: measure) the token budget for receipt analysis on an Ollama model.

For a given model it reports:
  * the model's context window,
  * the exact token cost of the base prompt (instructions + JSON schema),
  * the exact tokens-per-tag for the tag vocabulary,
  * the exact image-token cost for a sample image (optional),
  * a table showing how the prompt grows as the tag list grows,
  * how much "payload" (image + JSON output) still fits, and the tag count at
    which the prompt reaches a given fraction of the context window.

Token counts are EXACT: they come straight from Ollama's `prompt_eval_count`
(the number of tokens the model actually evaluated for the prompt), so no
tokenizer library or chars-per-token guesswork is needed. The only reservation
that is an estimate is the model's JSON *output* (see --output-budget).

Examples:
  uv run python scripts/context_budget.py
  uv run python scripts/context_budget.py --model glm-ocr:latest
  uv run python scripts/context_budget.py --model glm-ocr:latest \
      --image tests/data/very-long-hit.png --app-url http://localhost:8080
  # Statistical distribution of the image cost over real receipts in the DB:
  PG__PORT=54321 uv run python scripts/context_budget.py --model glm-ocr:latest --db-sample 24
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import sys
from pathlib import Path
from statistics import mean
from time import monotonic

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_bill.model.handoff import HandoffWait
from vision_bill.provider.llm.base import LLMProvider
from vision_bill.service.activation_admission import direct_inference_call, inference_admission

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_APP_URL = os.environ.get("VB_APP_URL", "http://localhost:8080")

# A realistic pool of grocery tags (multi-word on purpose) used to measure the
# average token cost per tag without needing the live database.
_TAG_POOL = [
    "alcohol",
    "beverage",
    "cheese",
    "coffee",
    "deposit",
    "electronics",
    "food",
    "fresh",
    "gift",
    "household",
    "household goods",
    "hygiene",
    "meat",
    "meat product",
    "office",
    "other",
    "pet",
    "poultry",
    "service",
    "subscription",
    "supplement",
    "travel",
    "bakery",
    "dairy",
    "frozen",
    "snack",
    "spice",
    "canned goods",
    "detergent",
    "paper products",
]


def _provider() -> LLMProvider:
    """A minimal concrete provider so we can reuse the real build_prompt()."""

    class _P(LLMProvider):
        async def check_connection(self) -> bool:
            return True

        async def get_available_models(self, capability="vision"):
            return []

        async def send_message(self, *a, **k):
            raise NotImplementedError("Prompt-only diagnostic provider")

        async def analyse_receipt_from_model(self, *a, **k):
            raise NotImplementedError("Prompt-only diagnostic provider")

        async def analyse_receipt_from_ocr_text(self, *a, **k):
            raise NotImplementedError("Prompt-only diagnostic provider")

    return _P()


def build_tags_prompt(provider: LLMProvider, count: int) -> str:
    tags = [
        _TAG_POOL[i % len(_TAG_POOL)] if count <= len(_TAG_POOL) else f"tag_{i}"
        for i in range(count)
    ]
    return provider.build_prompt(tags)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (len(sorted_vals) - 1) * (pct / 100.0)
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return sorted_vals[int(rank)]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (rank - lo)


def sample_image_paths(dsn: str, n: int) -> list[dict]:
    """Return up to `n` randomly-ordered image rows from the `images` table."""
    import asyncpg

    async def _run() -> list[dict]:
        conn = await asyncpg.connect(dsn)
        try:
            rows = await conn.fetch(
                "SELECT id, image_path, size_bytes, media_type, status FROM images "
                "WHERE image_path IS NOT NULL ORDER BY random() LIMIT $1",
                n,
            )
        finally:
            await conn.close()
        return [dict(r) for r in rows]

    return asyncio.run(_run())


def resolve_image_path(raw: str, extra_dirs: list[str]) -> Path | None:
    """Resolve a stored image_path to a local file.

    Stored paths may be container-absolute (e.g. ``/app/uploads/x.png``), so we
    also try the bare filename under the configured/local image directories.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    candidates = [raw]
    name = Path(raw).name
    for d in extra_dirs:
        if d:
            candidates.append(str(Path(d) / name))
    for cand in candidates:
        p = Path(cand)
        if p.is_file():
            return p
    return None


def report_db_image_sample(
    b: Budget,
    provider: LLMProvider,
    dsn: str | None,
    n: int,
    current_tags: list[str],
    output_budget: int,
) -> None:
    print(f"Image token cost from the database (random sample of up to {n}):")
    extra_dirs: list[str] = []
    try:
        from vision_bill.config import settings as _settings

        if not dsn:
            dsn = _settings.pg.pg_dsn
        extra_dirs = [
            _settings.images.save_dir,
            _settings.images.tmp_dir,
            "uploads",
            "uploads_tmp",
        ]
    except Exception as e:  # noqa: BLE001
        if not dsn:
            print(f"  Could not derive a Postgres DSN from settings ({e}); pass --db-dsn.\n")
            return
    if not dsn:
        return
    try:
        rows = sample_image_paths(dsn, n)
    except Exception as e:  # noqa: BLE001
        print(f"  Could not query the images table: {e}\n")
        return
    if not rows:
        print("  No images with a path found in the database yet.\n")
        return
    measured: list[int] = []
    unresolvable = 0
    for row in rows:
        path = resolve_image_path(row["image_path"], extra_dirs)
        if path is None:
            unresolvable += 1
            continue
        b64 = base64.b64encode(path.read_bytes()).decode()
        measured.append(b.count(provider.build_prompt([]), image_b64=b64) - b.base_tokens)
    if not measured:
        print(
            f"  {unresolvable} rows found but none resolved to a local file "
            "(stored paths may be container-absolute).\n"
        )
        return
    vals = sorted(measured)
    tail = f", {unresolvable} unresolvable" if unresolvable else ""
    print(f"  sampled {len(rows)} rows, measured {len(vals)} images{tail}")
    print(
        f"  min={vals[0]:.0f}  median={_percentile(vals, 50):.0f}  "
        f"mean={mean(vals):.0f}  p90={_percentile(vals, 90):.0f}  max={vals[-1]:.0f}  tokens"
    )
    base_plus_tags = b.base_tokens + round(len(current_tags) * b.tokens_per_tag)
    for label, img in (("median", _percentile(vals, 50)), ("p90", _percentile(vals, 90))):
        total = base_plus_tags + int(img) + output_budget
        print(
            f"  base + {len(current_tags)} tags + {label} image + output = {total} tokens "
            f"({pct(total, b.ctx)} of ctx)"
        )
    print()


class Budget:
    def __init__(self, host: str, model: str, num_ctx_cap: int, timeout: int):
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx_cap = num_ctx_cap
        self.timeout = timeout
        self.ctx = 0
        self.base_tokens = 0
        self.tokens_per_tag = 0.0
        self.image_tokens = 0

    def _post(self, path: str, payload: dict) -> dict:
        r = httpx.post(f"{self.host}{path}", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str) -> dict:
        r = httpx.get(f"{self.host}{path}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def context_length(self) -> int:
        data = self._get("/api/tags")
        for m in data.get("models", []):
            if m.get("model") == self.model or m.get("name") == self.model:
                self.ctx = int((m.get("details") or {}).get("context_length") or 0)
                if self.ctx > 0:
                    return self.ctx
                break
        # Fall back to model_info.
        data = self._post("/api/show", {"name": self.model})
        mi = data.get("model_info") or {}
        for key, value in mi.items():
            if key.endswith("context_length"):
                self.ctx = int(value)
                if self.ctx > 0:
                    return self.ctx
        return 0

    def count(self, prompt: str, image_b64: str | None = None, headroom: int = 4096) -> int:
        """Exact token count of `prompt` (plus an optional image) via prompt_eval_count."""
        if self.ctx <= 0 or self.num_ctx_cap <= 1:
            raise ValueError("Known positive context and diagnostic cap are required")
        num_ctx = min(self.ctx, self.num_ctx_cap)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": num_ctx, "num_predict": 1},
        }
        if image_b64:
            payload["images"] = [image_b64]
        from vision_bill.config import settings
        if settings.ocr.enabled and settings.ocr.manage_model_lifecycle and settings.ocr.shared_gpu:
            data = asyncio.run(self._managed_generate(payload))
        else:
            with inference_admission(), direct_inference_call():
                data = self._post("/api/generate", payload)
        count = data.get("prompt_eval_count")
        if type(count) is not int or count <= 0:
            raise ValueError("Prompt token count is unavailable")
        if count >= num_ctx - 1 or data.get("truncated") is True:
            raise ValueError("Prompt may be truncated at the diagnostic context cap; no complete count")
        return count

    async def _managed_generate(self, payload: dict) -> dict:
        from vision_bill.provider.llm.ollama import OllamaProvider
        provider = OllamaProvider(self.host, analysis_timeout_seconds=self.timeout)
        with inference_admission(provider):
            async with provider.handoff.session(monotonic() + min(600, self.timeout)) as owner:
                await owner.llm_phase(self.model)
                result: dict = await owner.model_call(asyncio.to_thread(self._post, "/api/generate", payload))
                return result

    def measure(self, provider: LLMProvider, tag_probe: int, image_b64: str | None) -> None:
        self.context_length()
        if tag_probe < 0:
            raise ValueError("Tag probe must be nonnegative")
        self.base_tokens = self.count(provider.build_prompt([]))
        big = self.count(build_tags_prompt(provider, tag_probe))
        if big < self.base_tokens:
            raise ValueError("Inconsistent prompt token count")
        self.tokens_per_tag = (big - self.base_tokens) / tag_probe if tag_probe else 0.0
        if image_b64:
            self.image_tokens = (
                self.count(provider.build_prompt([]), image_b64=image_b64) - self.base_tokens
            )
            if self.image_tokens < 0:
                raise ValueError("Inconsistent image token count")


def pct(part: int, whole: int) -> str:
    if not whole:
        return "n/a"
    return f"{100.0 * part / whole:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", help="Ollama model name (default: first vision-capable model)")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama host (default: {DEFAULT_HOST})")
    ap.add_argument("--image", help="path to a sample image to measure image-token cost")
    ap.add_argument(
        "--app-url", default=DEFAULT_APP_URL, help="vision-bill app URL to read the live tag count"
    )
    ap.add_argument("--tags-file", help="JSON file of the current tag list (overrides --app-url)")
    ap.add_argument(
        "--output-budget",
        type=int,
        default=2000,
        help="tokens reserved for the model's JSON receipt output (default 2000)",
    )
    ap.add_argument(
        "--tag-probe",
        type=int,
        default=250,
        help="tag count used to measure the average tokens-per-tag (default 250)",
    )
    ap.add_argument(
        "--num-ctx-cap",
        type=int,
        default=16384,
        help="max num_ctx to request during measurement (default 16384)",
    )
    ap.add_argument("--timeout", type=int, default=180, help="per-call HTTP timeout in seconds")
    ap.add_argument(
        "--db-sample",
        type=int,
        nargs="?",
        const=16,
        default=0,
        metavar="N",
        help="sample N random images from the `images` table and report a token-cost "
        "distribution (default N=16 when the flag is given alone)",
    )
    ap.add_argument(
        "--db-dsn",
        default=None,
        help="Postgres DSN for --db-sample (default: the app's settings.pg.pg_dsn)",
    )
    args = ap.parse_args()
    if args.output_budget < 0 or args.tag_probe < 0 or args.num_ctx_cap <= 1:
        ap.error("output/tag budgets must be nonnegative and context cap greater than one")

    host = args.host.rstrip("/")
    client = httpx.Client(timeout=30)
    try:
        # Resolve the model: explicit, or the first vision-capable one.
        model = args.model
        if not model:
            tags = client.get(f"{host}/api/tags").json().get("models", [])
            for m in tags:
                name = m.get("model")
                if not name:
                    continue
                caps = (m.get("details") or {}).get("capabilities")
                if caps and "vision" in caps:
                    model = name
                    break
            if not model:
                # details.capabilities isn't always present; probe via /api/show.
                for m in tags:
                    name = m.get("model")
                    if not name:
                        continue
                    show = client.post(f"{host}/api/show", json={"name": name}).json()
                    if "vision" in (show.get("capabilities") or []):
                        model = name
                        break
        if not model:
            print("No vision-capable model found. Pass --model explicitly.", file=sys.stderr)
            return 1
    except httpx.HTTPError as e:
        print(f"Could not reach Ollama at {host}: {e}", file=sys.stderr)
        return 1
    finally:
        client.close()

    # Live / provided tag list.
    current_tags: list[str] = []
    if args.tags_file:
        current_tags = json.loads(Path(args.tags_file).read_text())
    else:
        try:
            current_tags = httpx.get(f"{args.app_url.rstrip('/')}/api/v1/tags", timeout=5).json()
        except (httpx.HTTPError, ValueError):
            current_tags = []

    image_b64 = None
    if args.image:
        image_b64 = base64.b64encode(Path(args.image).read_bytes()).decode()

    provider = _provider()
    b = Budget(host, model, args.num_ctx_cap, args.timeout)
    print(f"Model: {model}\n")
    try:
        b.measure(provider, args.tag_probe, image_b64)
    except (HandoffWait, httpx.HTTPError, ValueError) as e:
        print(f"Measurement failed (is the model loaded / reachable?): {e}", file=sys.stderr)
        return 1

    print(f"Context window:          {b.ctx} tokens")
    print(
        f"Base prompt (instr+schema): {b.base_tokens} tokens  ({pct(b.base_tokens, b.ctx)} of ctx)"
    )
    print(f"Tokens per tag (sample average; growth estimates): {b.tokens_per_tag:.2f}")
    print(f"Diagnostic context cap: {min(b.ctx, b.num_ctx_cap)}; diagnostic output limit: 1 token")
    if image_b64:
        print(f"Image tokens ({Path(args.image).name}): {b.image_tokens} tokens")
    if current_tags:
        print(f"Current tags (live):   {len(current_tags)}")
    print(f"Reserved output budget:  {args.output_budget} tokens\n")

    if args.db_sample > 0:
        try:
            report_db_image_sample(
                b=b,
                provider=provider,
                dsn=args.db_dsn,
                n=args.db_sample,
                current_tags=current_tags,
                output_budget=args.output_budget,
            )
        except HandoffWait as e:
            print(f"Measurement held: {e}", file=sys.stderr)
            return 1

    def prompt_tokens(tag_count: int) -> int:
        return (
            b.base_tokens
            + round(tag_count * b.tokens_per_tag)
            + (b.image_tokens if image_b64 else 0)
        )

    print("Tag vocabulary growth (base + tags" + (" + image" if image_b64 else "") + "):")
    print(f"  {'tags':>6}  {'prompt':>8}  {'%ctx':>6}  {'payload headroom':>16}")
    counts = sorted(
        {0, 25, 50, 100, 250, 500, 1000} | ({len(current_tags)} if current_tags else set())
    )
    for n in counts:
        p = prompt_tokens(n)
        headroom = max(0, b.ctx - p - args.output_budget)
        print(f"  {n:>6}  {p:>8}  {pct(p, b.ctx):>6}  {headroom:>16}")

    # Tag count at which the prompt reaches 50/80% of context (ignoring image).
    if b.tokens_per_tag > 0:
        for frac in (0.5, 0.8):
            limit = int((frac * b.ctx - b.base_tokens) / b.tokens_per_tag)
            print(
                f"\nPrompt hits {frac * 100:.0f}% of context at ~{limit} tags"
                + (" (before image)" if image_b64 else "")
                + "."
            )

    # The sample image and reserved output are each counted exactly once.
    n_cur = len(current_tags)
    p_cur = prompt_tokens(n_cur)
    headroom_cur = b.ctx - p_cur - args.output_budget
    print(
        f"\nAt {n_cur} tags: prompt = {p_cur} tokens, leaving {headroom_cur} tokens "
        f"({pct(headroom_cur, b.ctx)}) remaining after prompt, sample image and reserved output."
    )
    if b.image_tokens:
        fits = headroom_cur >= 0
        print(
            f"  -> sample image costs {b.image_tokens} tokens; "
            f"{'fits' if fits else 'DOES NOT FIT'} with {headroom_cur} tokens of headroom."
        )
        if not fits:
            print("  The selected sample and output reservation exceed this model context; no app limits were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
