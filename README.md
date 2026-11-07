# vision-bill — Your Private Invoice Tracker

### 🛡️ Privacy First
**vision-bill** is a local-first tool designed to process receipts without ever uploading your data to the cloud. Receipts contain highly sensitive personal and financial information (PII); this project ensures that everything—from image processing to LLM reasoning—stays on your machine by default.

### 🚀 Core Features
- **Multi-modal Local LLMs:** Uses Ollama by default to run vision-capable models (like Llama-3-Vision, Moondream, or Gemma) directly on your hardware; an OpenAI-compatible endpoint is optional.
- **Local Extraction:** No separate OCR engine is required; the LLM performs both image understanding and structured data extraction in one step.
- **Privacy-Preserving Architecture:** Built with FastAPI to serve as a local backend for personal use, with a bundled SvelteKit SPA frontend.
- **Multi-user & Admin:** Argon2-hashed passwords, stateless session cookies, per-user data ownership, an admin Settings page, and a registration toggle.
- **Model Benchmarking:** Admins can queue benchmark runs that re-examine your verified receipts with any vision model and compare the results field by field.
- **Insights:** Product search with price history and spending statistics (per currency, merchant, category, payment method, weekday, and week).
- **Non-sequential identifiers:** Users, receipts, images, line items, taxes, tags, and benchmark runs use UUIDs throughout the database and API.

### 🛠️ Quickstart (Docker)
Versioned container publishing is available through the manual Release workflow;
see [release setup, image references, and recovery](RELEASES.md).

The fastest way to get started is using Docker Compose:
```bash
cp .env.example .env   # then set AUTH__SECRET_KEY (required) and your model/DB values
docker compose up -d
```
*Note: You must have [Ollama](https://ollama.com/) installed and running on your host machine with vision-capable models downloaded.*

Sign in at `http://localhost:8080`. By default, self-service registration is open; alternatively set `AUTH__BOOTSTRAP_USERNAME` / `AUTH__BOOTSTRAP_PASSWORD` in `.env` to have the first admin created on an empty database.

### ⚙️ Configuration
The application persists its resolved settings in `/app/config/config.yaml`,
which Docker stores on the host at `./server_data/config`. The file is created
on first startup from environment defaults. Environment variables always take
precedence and are marked as environment-controlled in the admin Settings page.
The page can change the default model, provider connection settings, and
registration policy without exposing database credentials or secrets. Model,
temperature, and registration changes apply immediately; changing the
provider or host requires a server restart.

### 📊 Model Testing & Benchmarking
Since different LLMs perform differently at image extraction, the project includes a benchmark system:
- **Benchmark Runs:** Admins queue runs via the API (`POST /api/v1/benchmarks`) or `make script-create-benchmark`. Each run re-extracts your **verified** receipts with the selected vision models — verified receipts serve as the ground truth.
- **Scoring:** Extracted data is compared against the verified values field by field; runs report per-model summaries (score, confidence, attempts, latency) and are durable across restarts.
- **Customization:** Swap the provider, model, or temperature from the admin Settings page (or via `LLM__*` environment variables); `make script-context-budget` measures prompt token usage of a model.

### 📦 Tech Stack
- **Backend:** Python, FastAPI, Pydantic, PostgreSQL (Alembic migrations)
- **Frontend:** SvelteKit SPA (TypeScript, Tailwind CSS)
- **Inference:** Ollama (local LLM execution; optional OpenAI-compatible endpoint)
- **Authentication:** Argon2id password hashing, stateless session cookies
- **Deployment:** Docker, UV (dependency management)
- **Testing:** Pytest (API), Vitest + Playwright (frontend)
