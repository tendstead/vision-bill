"""PaddleOCR-VL-1.6 HTTP service for vision-bill.

Provides a /ocr endpoint that accepts an image and returns structured
OCR results using PaddleOCR-VL-1.6.
"""

import logging
import os
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="PaddleOCR-VL-1.6 Service", version="1.0.0")

# Global OCR instance (initialized once at startup)
ocr_instance = None


def init_ocr():
    """Initialize the PaddleOCR-VL-1.6 instance."""
    global ocr_instance
    try:
        from paddleocr import PaddleOCRVL

        logger.info("Initializing PaddleOCR-VL-1.6...")
        ocr_instance = PaddleOCRVL(pipeline_version="v1.6")
        logger.info("PaddleOCR-VL-1.6 initialized successfully")
    except ImportError as e:
        logger.error("Failed to import paddleocr: %s", e)
        raise
    except Exception as e:
        logger.error("Failed to initialize PaddleOCR: %s", e)
        raise


@app.on_event("startup")
async def startup():
    """Initialize OCR on application startup."""
    init_ocr()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({"status": "ok"})


@app.post("/ocr")
async def ocr_image(image: UploadFile = File(...)):
    """Extract text from an uploaded image using PaddleOCR-VL-1.6."""
    if ocr_instance is None:
        raise HTTPException(status_code=503, detail="OCR service not initialized")

    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type: {image.content_type}. Allowed: {allowed_types}",
        )

    try:
        # Read image data
        image_data = await image.read()

        # Save to temporary file for PaddleOCR
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name

        try:
            # Run OCR using PaddleOCR-VL pipeline
            logger.info("Running OCR on image...")
            output = ocr_instance.predict(tmp_path)
            logger.info("OCR completed")

            # Process result - PaddleOCRVL returns a list of result objects
            if not output:
                return JSONResponse({"text": "", "blocks": []})

            # Get the first result (single image input)
            result = output[0]
            processed = process_ocrvl_result(result)

            return JSONResponse(processed)

        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        logger.exception("OCR processing failed")
        raise HTTPException(status_code=500, detail=str(e))


def process_ocrvl_result(result):
    """Process PaddleOCR-VL result object into structured format.

    The result object from PaddleOCRVL.predict() has a .res attribute
    containing the structured data with layout detection and recognition results.
    """
    # Extract the internal result dict
    res_dict = result.res if hasattr(result, "res") else result

    if not res_dict:
        return {"text": "", "blocks": []}

    # Extract layout detection results
    layout_res = res_dict.get("layout_det_res", {})
    boxes = layout_res.get("boxes", [])

    blocks = []
    for box in boxes:
        label = box.get("label", "")
        score = box.get("score", 0.0)
        coordinate = box.get("coordinate", [])

        # Convert numpy floats to regular floats
        if isinstance(score, (float, int)):
            score = float(score)

        blocks.append(
            {
                "label": label,
                "confidence": score,
                "bbox": [[float(x) for x in point] for point in coordinate] if coordinate else [],
            }
        )

    # For now, return layout info - the actual text recognition is done
    # by the VLM component which requires a separate service for production
    return {
        "text": "",  # Text recognition requires VLM service
        "blocks": blocks,
        "layout_info": layout_res,
    }


if __name__ == "__main__":
    port = int(os.environ.get("OCR_PORT", 8082))
    logger.info(f"Starting PaddleOCR service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
