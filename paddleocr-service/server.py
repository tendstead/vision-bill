"""PaddleOCR-VL-1.6 HTTP service for vision-bill.

Provides a /ocr endpoint that accepts an image and returns structured OCR
results using PaddleOCR-VL-1.6.

Design notes
------------
* ``/ocr`` is a *synchronous* endpoint, so FastAPI runs it in a worker thread.
  The blocking ``predict()`` call therefore never blocks the asyncio event
  loop, which keeps ``/health`` responsive while an image is being processed.
* The actual inference runs in a **separate child process**. PaddleOCR-VL can
  raise native (C++) fatal errors that abort the whole process. Isolating the
  inference means such a crash only kills the disposable worker, which the
  server transparently respawns. The HTTP server itself stays up and healthy.
"""

import logging
import multiprocessing as mp
import os
import queue as queue_mod
import tempfile
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="PaddleOCR-VL-1.6 Service", version="1.1.0")

ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"]

# Inference timeout (seconds); overridable via environment. CPU inference of
# PaddleOCR-VL is slow (minutes per image), so the default is generous. Tune it
# to your hardware, and keep the vision-bill client's OCR__TIMEOUT_SECONDS >= it.
OCR_TIMEOUT_SECONDS = int(os.environ.get("OCR_TIMEOUT_SECONDS", "900"))

# Global worker manager. Created in startup(); None until then.
ocr_worker = None

# Model instance. Lives only in the worker (child) process.
_instance = None


# --------------------------------------------------------------------------- #
# Result normalisation (runs inside the worker process)
# --------------------------------------------------------------------------- #
def _to_plain(obj):
    """Recursively convert numpy scalars/arrays to plain JSON-safe Python."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    # Primitives (and None) pass through untouched; this also narrows the type
    # so the numpy checks below are only reached for non-primitive objects.
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj
    # numpy scalar (has .item) or numpy array (has .tolist)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, AttributeError, TypeError):
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except (ValueError, AttributeError, TypeError):
            pass
    return obj


def _normalize_bbox(coordinate):
    """Normalise a box coordinate to a list of ``[x, y]`` points.

    PaddleOCR may return either a flat list ``[x1, y1, x2, y2, ...]`` or a list
    of points ``[[x1, y1], [x2, y2], ...]``. Both are handled.
    """
    coordinate = _to_plain(coordinate)
    if not coordinate:
        return []
    first = coordinate[0]
    if isinstance(first, (int, float)) or hasattr(first, "item"):
        nums = [float(v) for v in coordinate]
        return [[nums[i], nums[i + 1]] for i in range(0, len(nums) - 1, 2)]
    return [[float(x) for x in point] for point in coordinate]


def process_ocrvl_result(result):
    """Return recognized content in reading order, without image/model internals."""
    res_dict = result.res if hasattr(result, "res") else result
    blocks = []
    for block in (res_dict or {}).get("parsing_res_list", []) or []:
        # Native PaddleX results contain ParsingBlock objects; exported JSON
        # contains dictionaries. Both preserve the pipeline's reading order.
        if isinstance(block, dict):
            text = block.get("block_content", "")
            label = block.get("block_label", "text")
            coordinate = block.get("block_bbox", [])
        else:
            text = getattr(block, "content", "")
            label = getattr(block, "label", "text")
            coordinate = getattr(block, "bbox", [])
        if not isinstance(text, str) or not text.strip():
            continue
        blocks.append(
            {
                "label": str(label),
                "text": text.strip(),
                "bbox": _normalize_bbox(coordinate),
            }
        )
    if not blocks:
        raise ValueError("OCR returned no recognized text; layout boxes are not transcription")
    return {"text": "\n".join(block["text"] for block in blocks), "blocks": blocks}


# --------------------------------------------------------------------------- #
# Isolated inference worker (child process)
# --------------------------------------------------------------------------- #
def _worker_main(in_q, out_q, ready_evt):
    """Run in the child process: load the model once, then serve predict calls.

    If the native Paddle code aborts this process, the parent detects the dead
    worker and spawns a fresh one. Nothing here can take down the HTTP server.
    """
    global _instance
    try:
        from paddleocr import PaddleOCRVL

        logger.info("Worker: initializing PaddleOCR-VL-1.6...")
        _instance = PaddleOCRVL(pipeline_version="v1.6")
        logger.info("Worker: PaddleOCR-VL-1.6 ready")
    except Exception as e:
        logger.exception("Worker: failed to initialize PaddleOCR-VL")
        out_q.put({"ok": False, "error": f"model init failed: {e}"})
        ready_evt.set()
        return

    ready_evt.set()
    while True:
        item = in_q.get()
        if item is None:  # shutdown sentinel
            break
        req_id, tmp_path = item
        try:
            output = _instance.predict(tmp_path)
            result = process_ocrvl_result(output[0] if output else {})
            out_q.put({"req_id": req_id, "ok": True, "result": result})
        except Exception as e:
            logger.exception("Worker: predict failed")
            out_q.put({"req_id": req_id, "ok": False, "error": str(e)})


class OcrWorkerManager:
    """Owns the isolated inference process and respawns it if it crashes."""

    def __init__(self):
        # "spawn" gives the worker a clean process (no inherited native state).
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._in_q = None
        self._out_q = None
        self._ready = None
        self._lock = threading.Lock()
        self._req_counter = 0
        self._spawn()

    def _spawn(self):
        self._teardown()
        self._in_q = self._ctx.Queue()
        self._out_q = self._ctx.Queue()
        self._ready = self._ctx.Event()
        self._proc = self._ctx.Process(
            target=_worker_main,
            args=(self._in_q, self._out_q, self._ready),
            daemon=True,
        )
        self._proc.start()
        if not self._ready.wait(timeout=600):
            raise RuntimeError("OCR worker did not become ready in time")

    def _teardown(self):
        proc, in_q = self._proc, self._in_q
        self._proc = None
        self._in_q = None
        self._out_q = None
        self._ready = None
        if proc is None:
            return
        try:
            if proc.is_alive() and in_q is not None:
                in_q.put(None)  # ask it to exit cleanly
                proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
        except OSError as exc:
            logger.debug("Error during worker teardown: %s", exc)

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def predict(self, tmp_path: str, timeout_seconds: int):
        """Run one inference, respawning the worker once if it crashes.

        Requests are correlated by a monotonically increasing id so that a
        stale result from a previously timed-out request is discarded rather
        than delivered to the next caller.
        """
        with self._lock:
            attempt = 0
            while True:
                attempt += 1
                self._req_counter += 1
                req_id = self._req_counter
                self._in_q.put((req_id, tmp_path))
                deadline = time.monotonic() + timeout_seconds
                msg = None
                crashed = False
                while msg is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"OCR inference timed out after {timeout_seconds}s")
                    try:
                        candidate = self._out_q.get(timeout=min(2.0, remaining))
                    except queue_mod.Empty:
                        if not self._proc.is_alive():
                            crashed = True
                            break
                        continue  # worker still churning, keep waiting
                    if candidate.get("req_id") == req_id:
                        msg = candidate
                    # else: stale result from a timed-out request; discard it
                if crashed:
                    if attempt >= 2:
                        raise RuntimeError("OCR worker crashed during inference")
                    logger.warning("OCR worker died mid-inference; respawning and retrying")
                    self._spawn()
                    continue  # retry the same request on the fresh worker
                if msg.get("ok"):
                    return msg["result"]
                raise RuntimeError(msg.get("error", "OCR failed"))

    def stop(self):
        with self._lock:
            self._teardown()


# --------------------------------------------------------------------------- #
# FastAPI application
# --------------------------------------------------------------------------- #
def _init_worker():
    """Initialize the OCR worker in a background thread.

    Loading the PaddleOCR-VL model takes a while; doing it off the event loop
    keeps /health responsive from the moment the HTTP server starts.
    """
    global ocr_worker
    try:
        ocr_worker = OcrWorkerManager()
        logger.info("OCR worker ready")
    except Exception:
        logger.exception("Failed to initialize OCR worker")


@app.on_event("startup")
async def startup():
    thread = threading.Thread(target=_init_worker, name="ocr-worker-init", daemon=True)
    thread.start()
    logger.info("OCR worker initialization started in background")


@app.on_event("shutdown")
async def shutdown():
    if ocr_worker is not None:
        ocr_worker.stop()


@app.get("/health")
async def health():
    """Health check. Always fast: inference runs off the event loop."""
    worker_ok = ocr_worker is not None and ocr_worker.is_alive()
    return JSONResponse({"status": "ok", "worker": worker_ok})


@app.post("/ocr")
def ocr_image(image: UploadFile = File(...)):  # noqa: B008 - FastAPI dependency idiom
    """Extract text from an uploaded image using PaddleOCR-VL-1.6.

    Synchronous on purpose: FastAPI dispatches it to a worker thread, so the
    blocking inference never stalls the event loop (or ``/health``).
    """
    if ocr_worker is None:
        raise HTTPException(status_code=503, detail="OCR service not initialized")

    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type: {image.content_type}. Allowed: {ALLOWED_TYPES}",
        )

    image_data = image.file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_data)
        tmp_path = tmp.name

    try:
        result = ocr_worker.predict(tmp_path, OCR_TIMEOUT_SECONDS)
        return JSONResponse(result)
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        logger.exception("OCR processing failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    port = int(os.environ.get("OCR_PORT", "8082"))
    logger.info("Starting PaddleOCR service on port %s", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
