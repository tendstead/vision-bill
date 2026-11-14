FROM python:3.12-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
ENV OMP_NUM_THREADS=4

FROM base AS tesseract
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/*

FROM base AS torch-cpu
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

FROM torch-cpu AS easyocr
RUN pip install --no-cache-dir easyocr==1.7.2

FROM torch-cpu AS doctr
RUN pip install --no-cache-dir python-doctr

# Reuse the project's pinned Paddle environment; no second Paddle dependency set.
FROM vision-bill-paddleocr-service:latest AS paddle
