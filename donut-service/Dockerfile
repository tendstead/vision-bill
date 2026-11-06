FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/models
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Boundary tests run without downloading a model or installing PyTorch.
FROM base AS test
RUN pip install --no-cache-dir httpx==0.28.1 ruff==0.12.0
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]

FROM base AS runtime
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
EXPOSE 8083
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8083", "--workers", "1"]
