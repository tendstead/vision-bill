# vision-bill development workflow.
.DEFAULT_GOAL := help
.PHONY: ocr-compare receipt-compare
receipt-compare: ## Compare direct vision and cached OCR with local LLMs (ARGS="--dataset ... --targets ... --ocr-report ...")
	uv run python -m scripts.ocr_harness.comparison $(ARGS)

ocr-compare: ## Compare local OCR engines (ARGS="run --data-dir /path/to/receipts")
	uv run python -m scripts.ocr_harness $(ARGS)
SHELL := /bin/bash

.PHONY: help install-uv setup migrate run docker-up docker-down docker-logs test lint context-budget \
	script-context-budget script-create-benchmark db-configure-roles \
	e2e-up e2e-down test-e2e \
	fe-install fe-dev fe-build fe-sync fe-check fe-test fe-test-e2e fe-verify fe

help: ## Show this help
	@echo "vision-bill — available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

setup: ## Install project + dev extra (ruff, mypy, pytest) via uv
	uv sync --extra dev

migrate: ## Apply database migrations
	uv run alembic upgrade head

run: ## Run the API locally with auto-reload
	uv run uvicorn src.vision_bill.main:app --host 0.0.0.0 --port 8080 --reload --reload-dir ./src/vision_bill

docker-up: ## Build image + start app and postgres
	docker compose up --build

docker-down: ## Stop app and postgres
	docker compose down

docker-logs: ## Follow vision_bill container logs
	docker compose logs -f vision_bill

test: ## Run unit tests (e2e excluded; see test-e2e)
	uv run --extra dev pytest tests/ -m "not e2e"

e2e-up: ## Start a private stack (requires E2E_STATE=/private/directory/state.json)
	uv run python e2e/run.py up --state "$(E2E_STATE)"

e2e-down: ## Remove only the explicitly owned stack
	uv run python e2e/run.py down --state "$(E2E_STATE)"

E2E_BUILD_OVERRIDE ?=
E2E_OVERRIDE_ARG = $(if $(E2E_BUILD_OVERRIDE),--override "$(E2E_BUILD_OVERRIDE)")

test-e2e: ## Run backend E2E in a unique owned stack
	uv run --extra dev python e2e/run.py run $(E2E_OVERRIDE_ARG) -- uv run --extra dev pytest tests/e2e/

fe-test-e2e: ## Run Playwright in a unique owned stack
	$(NODE) uv run python e2e/run.py run --policy fallback $(E2E_OVERRIDE_ARG) -- npm --prefix frontend run test:e2e

PYTHON_CHECK_PATHS := src paddleocr-service/ocr_contract paddleocr-service/ocr_service paddleocr-service/server.py \
	paddleocr-service/build e2e scripts/ocr_harness scripts/ocr_build_rocm.py scripts/ocr_cuda_probe.py scripts/ocr_source_up.py scripts/ocr_deferred.py scripts/ocr_native_reuse.py scripts/ocr_stopped_recovery.py \
	scripts/receipt_split_experiment.py scripts/standalone_benchmark.py scripts/test_ocr_service.py

.PHONY: compose-check
compose-check: ## Validate GPU Compose overlays with private synthetic settings (no services)
	uv run python e2e/check_compose.py

lint: ## Lint (ruff) and type-check (mypy)
	uv run --extra dev ruff check $(PYTHON_CHECK_PATHS)
	uv run --extra dev mypy $(PYTHON_CHECK_PATHS)

# --- Operational scripts -------------------------------------------------
# Values loaded here become real environment variables and therefore override
# the application's .env files. Override with: make ... SCRIPTS_ENV=path/to/file
SCRIPTS_ENV ?= .env.scripts
WITH_SCRIPTS_ENV = set -a; if [[ -f "$(SCRIPTS_ENV)" ]]; then source "$(SCRIPTS_ENV)"; fi; set +a;

script-context-budget: ## Measure Ollama context usage (ARGS="...")
	@$(WITH_SCRIPTS_ENV) uv run python scripts/context_budget.py $(ARGS)

script-create-benchmark: ## Queue a benchmark API run (ARGS="...")
	@$(WITH_SCRIPTS_ENV) uv run python scripts/create_benchmark_run.py $(ARGS)

script-standalone-benchmark: ## Run benchmark directly without API (ARGS="...")
	@$(WITH_SCRIPTS_ENV) uv run python scripts/standalone_benchmark.py $(ARGS)

script-test-ocr: ## Test the PaddleOCR service with local images (ARGS="...")
	@$(WITH_SCRIPTS_ENV) uv run python scripts/test_ocr_service.py $(ARGS)

db-configure-roles: ## Create/update PostgreSQL runtime and read-only group roles
	@$(WITH_SCRIPTS_ENV) test -n "$$MIGRATION_DATABASE_URL" || { echo "Set MIGRATION_DATABASE_URL in $(SCRIPTS_ENV)"; exit 2; }; \
		psql "$$MIGRATION_DATABASE_URL" -v database_name="$${PG__DB:-vision_bill}" -f scripts/configure_db_roles.sql

.PHONY: release-test
release-test: ## Test manual release allocation and publishing safeguards (no publication)
	$(NODE) node --test scripts/release.test.mjs

context-budget: script-context-budget ## Alias for script-context-budget

# --- Frontend (SvelteKit SPA) -------------------------------------------
# Node/npm come from nvm when present (local dev, pinned Node version
# independent of the caller's global default); otherwise the system Node is
# used (e.g. GitHub Actions runners, where actions/setup-node provides it).

NVM_DIR ?= $(HOME)/.nvm
ifeq ($(wildcard $(NVM_DIR)/nvm.sh),)
NODE :=
else
NODE := . "$(NVM_DIR)/nvm.sh" >/dev/null 2>&1 &&
endif
FE := frontend

fe-install: ## Install locked frontend npm dependencies
	$(NODE) (cd $(FE) && npm ci)

fe-dev: ## Run the Vite dev server on :5173 (proxies /api -> :8080)
	$(NODE) (cd $(FE) && npm run dev)

fe-build: ## Build the SPA into frontend/out (run fe-install once after dependency changes)
	$(NODE) (cd $(FE) && npm run build)

fe-sync: fe-build ## Copy the built SPA into src/vision_bill/static for FastAPI/Docker
	rm -rf src/vision_bill/static
	mkdir -p src/vision_bill/static
	cp -r $(FE)/out/. src/vision_bill/static/
	touch src/vision_bill/static/.gitkeep

fe-check: ## Frontend static checks (svelte-check + eslint + prettier)
	$(NODE) (cd $(FE) && npm run check && npm run lint && npm run format:check)

fe-test: ## Run frontend unit tests (vitest)
	$(NODE) (cd $(FE) && npm run test)

fe-typecheck-e2e: ## Type-check the Playwright e2e suite
	$(NODE) (cd $(FE) && npm run typecheck:e2e)

fe-verify: fe-check fe-test fe-typecheck-e2e fe-build ## Run frontend checks, unit tests, and a production build

fe: ## Run an npm script in frontend/ (make fe CMD=<script> [ARGS="..."])
	@test -n "$(CMD)" || { echo "usage: make fe CMD=<npm-script> [ARGS=\"arg1 arg2\"]"; exit 1; }
	$(NODE) (cd $(FE) && npm run $(CMD) $(ARGS))

.PHONY: ocr-lock-cuda ocr-build-cuda ocr-build-rocm ocr-qualify-cuda ocr-qualify-rocm
ocr-lock-cuda ocr-build-cuda: export OCR_CUDA_CHANNEL ?= cu129
ocr-lock-cuda: ## Resolve official CUDA lock (default cu129, optional OCR_CUDA_CHANNEL=cu126)
	SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PADDLEX=3.7.2 SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PADDLEOCR=3.7.0 uv run --no-project --with packaging==26.2 python paddleocr-service/build/resolve_cuda.py --channel="$${OCR_CUDA_CHANNEL:-cu129}"

ocr-build-cuda: ## Build candidate CUDA image after official lock resolution
	OCR_CUDA_WHEEL_CONTEXT="$${VISION_BILL_VERIFIED_CUDA_WHEEL_CONTEXT:-$$(uv run --no-project --with packaging==26.2 python paddleocr-service/build/resolve_cuda.py --check paddleocr-service/requirements.cuda.lock --channel="$${OCR_CUDA_CHANNEL:-cu129}" --build-context)}" && \
	export OCR_CUDA_WHEEL_CONTEXT OCR_CUDA_LOCK_CONTEXT="$$OCR_CUDA_WHEEL_CONTEXT" && \
	docker compose --profile ocr -f docker-compose.yml -f docker-compose.cuda.yml build paddleocr-service

ocr-build-rocm: ## Build the native Radeon SDK candidate
	python3 -m scripts.ocr_build_rocm

ocr-qualify-cuda ocr-qualify-rocm: ## Run an explicit supervised CUDA/ROCm source qualification
	@set -eu; \
		accelerator="$(@:ocr-qualify-%=%)"; \
		intent="$${OCR_MAINTENANCE_INTENT:?Set OCR_MAINTENANCE_INTENT for this invocation}"; \
		handoff="$${OCR_APP_HANDOFF_LOCK:?Set OCR_APP_HANDOFF_LOCK to the original /app/config lock path}"; \
		compose=(docker compose --profile ocr -f docker-compose.yml -f "docker-compose.$$accelerator.yml"); \
		running="$$("$${compose[@]}" ps --status running --status restarting --quiet vision_bill paddleocr-service)"; \
		if test -n "$$running"; then echo "Stop the source app and OCR sidecar before qualification" >&2; exit 1; fi; \
		compose+=(-f docker-compose.ocr-qualification.yml); \
		if [[ $$accelerator == rocm ]] && grep -qi microsoft /proc/sys/kernel/osrelease; then \
			compose+=(-f docker-compose.rocm-wsl.yml); \
		fi; \
		"$${compose[@]}" run --rm --no-deps --pull never \
			-v "$$PWD/server_data/ocr-runtime:/var/lib/ocr" -v "$$PWD/server_data/config:/app/config" \
			paddleocr-service python -m ocr_service.source_maintenance --accelerator "$$accelerator" \
			--runtime /var/lib/ocr/candidate.runtime.json --cases /var/lib/ocr/fixtures.toml \
			--output /var/lib/ocr/qualification.json --maintenance-intent "$$intent" --handoff-lock "$$handoff"

.PHONY: distribution-test
distribution-test: ## Test prebuilt bundle generation/Compose and release gates on CPU (no publication)
	$(NODE) node --test scripts/distribution.test.mjs scripts/distribution-start-linux.test.mjs

.PHONY: distribution-pid1-test
distribution-pid1-test: ## Run native coordinator/provision software tests as CPU container PID1 (no GPU or weights)
	$(NODE) node scripts/distribution-pid1-test.mjs
