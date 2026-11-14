"""Self-contained, offline HTML inspection of real extraction artifacts."""

import base64
import hashlib
import html
import json
import mimetypes
from pathlib import Path


def read_json(path):
    try:
        return json.loads(Path(path).read_text()) if path else None
    except (OSError, ValueError):
        return None


def pre(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    return "<pre>" + html.escape(text) + "</pre>"


def details(title, value):
    return "<details><summary>" + html.escape(title) + "</summary>" + pre(value) + "</details>"


def render_report(output, rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["input"], []).append(row)
    parts = [
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Receipt extraction — step by step</title><style>
:root{font:16px/1.5 system-ui;color:#24212b;background:#f6f4f8}
body{max-width:1400px;margin:auto;padding:24px}h1,h2,h3{line-height:1.25}
a,summary{color:#6750a4}summary{cursor:pointer;font-weight:600;padding:8px 0}
.case{display:grid;grid-template-columns:minmax(240px,360px) minmax(0,1fr);gap:24px;margin:32px 0}
.image{align-self:start;position:sticky;top:16px}.image img{width:100%;max-height:80vh;object-fit:contain;background:white}
article,.image{background:white;border:1px solid #ddd7e4;border-radius:12px;padding:20px}
article{margin-bottom:20px;min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f4f8;padding:12px;border-radius:6px;max-height:32rem;overflow:auto}
.status{font-size:.9rem;color:#625c6b}.error{color:#9b2633}section.stage{border-top:1px solid #e7e1ec;padding-top:8px;margin-top:16px}
@media(max-width:760px){body{padding:12px}.case{grid-template-columns:1fr}.image{position:static}.image img{max-height:55vh}}
</style><h1>Receipt extraction — step by step</h1>
<p>Local report. Images and receipt text are embedded; keep this file private. No scripts or external resources.</p>
<p>The harness does not call the HTTP API or persist receipts. Cached stages retain their original artifacts; missing historical data is labeled explicitly.</p>"""
    ]
    for input_path, cases in groups.items():
        image = Path(input_path)
        parts.append(
            '<div class="case"><aside class="image"><h2>1. Input image</h2><p>'
            + html.escape(image.name)
            + "</p>"
        )
        try:
            raw = image.read_bytes()
            expected = cases[0].get("identity", {}).get("input_sha256")
            mime = mimetypes.guess_type(image.name)[0]
            if expected and hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("Image changed since this run; current file is not shown.")
            if mime not in ("image/png", "image/jpeg", "image/webp", "image/bmp"):
                raise ValueError("Image format cannot be embedded safely.")
            parts.append(
                f'<img alt="Tested receipt image" src="data:{mime};base64,{base64.b64encode(raw).decode()}">'
            )
        except (OSError, ValueError) as exc:
            parts.append('<p class="error">' + html.escape(str(exc)) + "</p>")
        parts.append("</aside><div>")
        for row in cases:
            parts.append(
                "<article><h2>"
                + html.escape(row["engine"])
                + '</h2><p class="status">'
                + html.escape(
                    f"OCR: {row['status']} · cached: {row.get('cached', False)} · JSON: {row.get('llm_status', 'not run')}"
                )
                + "</p>"
            )
            ocr = read_json(row.get("artifact")) if row["status"] == "ok" else None
            parts.append('<section class="stage"><h3>2. OCR output</h3>')
            if ocr:
                parts.append(pre(ocr.get("text", "")))
                parts.append(
                    details(
                        "Text sent to Gemma (coordinate-grouped rows)",
                        ocr.get("row_text", ocr.get("text", "")),
                    )
                )
                parts.append(details("OCR boxes, confidence, versions and warnings", ocr))
            else:
                parts.append(pre(row.get("error") or "OCR result not captured."))
            parts.append('</section><section class="stage"><h3>3. LLM responses</h3>')
            trace = read_json(row.get("trace_artifact"))
            if trace and trace.get("attempts"):
                for index, attempt in enumerate(trace["attempts"], 1):
                    parts.append(f"<h4>Attempt {index}</h4>")
                    parts.append(details("Request / prompt", attempt.get("messages", [])))
                    parts.append(
                        pre(
                            attempt.get("response")
                            if "response" in attempt
                            else attempt.get("error", "No response")
                        )
                    )
            else:
                parts.append(
                    "<p>Raw LLM response not captured in this cache, or generation has not run. Use --refresh-llm to capture a fresh JSON stage without rerunning OCR.</p>"
                )
            if row.get("llm_error"):
                parts.append('<p class="error">' + html.escape(row["llm_error"]) + "</p>")
            parts.append('</section><section class="stage"><h3>4. Final extraction result</h3>')
            receipt = (
                read_json(row.get("receipt_artifact")) if row.get("llm_status") == "ok" else None
            )
            if receipt is not None:
                label = (
                    "ReceiptService return"
                    if row.get("execution_path") == "receipt_service"
                    else "Validated provider result (older cache; service return not recorded)"
                )
                parts.append(
                    "<p>"
                    + html.escape(label)
                    + ". No HTTP response or database persistence is implied.</p>"
                    + pre(receipt)
                )
            else:
                parts.append("<p>No validated receipt returned.</p>")
            score = read_json(row.get("score_artifact"))
            if score:
                parts.append(details("Expected values, field differences and score", score))
            parts.append("</section></article>")
        parts.append("</div></div>")
    parts.append("</html>")
    (Path(output) / "report.html").write_text("\n".join(parts))
