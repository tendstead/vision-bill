"""Independent CPU Donut receipt-parsing reference service.

Returns the checkpoint's own schema, not Vision Bill's Receipt schema.
No receipt images or pixel arrays are included in the response.
"""

import io
import logging
import os
import threading
import time
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from decoding import clean_sequence

logger = logging.getLogger(__name__)
MODEL = os.getenv("DONUT_MODEL", "naver-clova-ix/donut-base-finetuned-cord-v2")
REVISION = os.getenv("DONUT_REVISION") or None
TASK_PROMPT = os.getenv("DONUT_TASK_PROMPT", "<s_cord-v2>")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
inference_lock = threading.Lock()


class DonutEngine:
    def __init__(self):
        # Lazy imports keep boundary tests independent of PyTorch/model downloads.
        import torch
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        threads = int(os.getenv("DONUT_CPU_THREADS", "2"))
        self.max_new_tokens = int(os.getenv("DONUT_MAX_NEW_TOKENS", "768"))
        if threads < 1 or self.max_new_tokens < 1:
            raise ValueError("CPU threads and generation token limit must be positive")
        torch.set_num_threads(threads)
        self.torch = torch
        kwargs = {"revision": REVISION, "local_files_only": os.getenv("HF_HUB_OFFLINE") == "1"}
        self.processor = DonutProcessor.from_pretrained(MODEL, use_fast=False, **kwargs)
        self.model = VisionEncoderDecoderModel.from_pretrained(MODEL, **kwargs)
        self.model.to("cpu").eval()
        self.prompt_ids = self.processor.tokenizer(
            TASK_PROMPT, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        # Cap decoder output including the task prefix to its positional capacity.
        capacity = self.model.decoder.config.max_position_embeddings
        self.max_new_tokens = min(self.max_new_tokens, capacity - self.prompt_ids.shape[-1])
        if self.max_new_tokens < 1:
            raise ValueError("Task prompt leaves no decoder capacity")

    def extract(self, image):
        tokenizer = self.processor.tokenizer
        pixels = self.processor(image, return_tensors="pt").pixel_values
        kwargs = {}
        if tokenizer.unk_token_id is not None:
            kwargs["bad_words_ids"] = [[tokenizer.unk_token_id]]
        with self.torch.inference_mode():
            sequences = self.model.generate(
                pixels,
                decoder_input_ids=self.prompt_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **kwargs,
            )
        generated = sequences[0, self.prompt_ids.shape[-1]:].tolist()
        truncated = tokenizer.eos_token_id not in generated
        raw = clean_sequence(
            self.processor.batch_decode(sequences)[0], TASK_PROMPT,
            tokenizer.eos_token or "", tokenizer.pad_token or "",
        )
        return {
            "prediction": self.processor.token2json(raw),
            "raw_sequence": raw,
            "generated_tokens": len(generated),
            "truncated": truncated,
        }


@asynccontextmanager
async def lifespan(app):
    logger.info("Loading Donut checkpoint %s (CPU); first start downloads weights", MODEL)
    app.state.engine = DonutEngine()
    yield
    app.state.engine = None


app = FastAPI(title="Vision Bill Donut reference", lifespan=lifespan)
app.state.engine = None


@app.get("/health")
def health():
    ready = app.state.engine is not None
    return JSONResponse({"status": "ready" if ready else "loading", "model": MODEL},
                        status_code=200 if ready else 503)


@app.post("/ocr")
def ocr(image: UploadFile = File(...)):
    if app.state.engine is None:
        raise HTTPException(503, "Donut model not loaded")
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Use JPEG, PNG, WebP or BMP")
    if not inference_lock.acquire(blocking=False):
        raise HTTPException(429, "Donut is processing another image; retry when it finishes")
    started = time.perf_counter()
    try:
        data = image.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image exceeds the 25 MiB upload limit")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as source:
                    if source.width * source.height > MAX_IMAGE_PIXELS:
                        raise HTTPException(413, "Image exceeds the 32 megapixel limit")
                    if source.format not in {"JPEG", "PNG", "WEBP", "BMP"}:
                        raise HTTPException(415, "Unsupported image encoding")
                    receipt = ImageOps.exif_transpose(source).convert("RGB")
        except (Image.DecompressionBombError, Image.DecompressionBombWarning):
            raise HTTPException(413, "Image dimensions are too large") from None
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(422, "Cannot decode receipt image") from None
        with receipt:
            result = app.state.engine.extract(receipt)
            result.update({
                "model": MODEL,
                "revision": REVISION,
                "task_prompt": TASK_PROMPT,
                "device": "cpu",
                "image_size": list(receipt.size),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            })
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("Donut inference failed")
        raise HTTPException(500, "Donut inference failed; inspect service logs") from None
    finally:
        inference_lock.release()
