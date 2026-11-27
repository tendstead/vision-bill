# vision-bill development workflow.
.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help run build test lint fmt docker-up docker-down docker-logs \
	script-context-budget script-create-benchmark script-thumbnails context-budget db-configure-roles \
	e2e-up e2e-down test-e2e release-test \
	fe-install fe-dev fe-build fe-sync fe-check fe-test fe-test-e2e fe-verify fe

GO_PKGS := ./cmd/... ./internal/...

help: ## Show this help
	@echo "vision-bill — available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'

run: ## Run the server locally on :8080 (config from env / .env; needs a built SPA via fe-sync)
	VB_LISTEN_ADDR=:8080 go run ./cmd/server

build: ## Build the server and tools into bin/ (CGO disabled)
	CGO_ENABLED=0 go build -trimpath -o bin/vision-bill ./cmd/server
	CGO_ENABLED=0 go build -trimpath -o bin/vbctl ./cmd/vbctl

fmt: ## Format Go sources
	gofmt -w cmd internal e2e

lint: ## gofmt check + go vet
	@test -z "$$(gofmt -l cmd internal e2e)" || { gofmt -l cmd internal e2e; echo "gofmt needed (make fmt)"; exit 1; }
	go vet $(GO_PKGS)
	go vet -tags e2e ./e2e/...

test: ## Go tests with the race detector (starts a throwaway PostgreSQL unless VB_TEST_DATABASE_URL is set)
	scripts/go-test.sh

docker-up: ## Build image + start app and postgres
	docker compose up --build

docker-down: ## Stop app and postgres
	docker compose down

docker-logs: ## Follow vision_bill container logs
	docker compose logs -f vision_bill

e2e-up: ## Build and start the e2e stack (manual; the test targets manage it)
	$(E2E_COMPOSE) up -d --build --wait

e2e-down: ## Stop the e2e stack and delete its volumes (manual)
	$(E2E_COMPOSE) down -v

# The e2e test targets own the stack lifecycle: start it for the run and
# always tear it down on exit (the EXIT trap covers failures and Ctrl-C).
E2E_COMPOSE := docker compose -f docker-compose.e2e.yml
test-e2e: ## Run the black-box API e2e suite (starts the e2e stack, always tears it down)
	@bash -c 'trap "$(E2E_COMPOSE) down -v" EXIT; \
		$(E2E_COMPOSE) up -d --build --wait && \
		go test -tags e2e -count=1 -p 1 ./e2e/suite/...'

fe-test-e2e: ## Playwright e2e (starts the e2e stack, always tears it down)
	@bash -c 'trap "$(E2E_COMPOSE) down -v" EXIT; \
		$(E2E_COMPOSE) up -d --build --wait && \
		$(NODE) (cd $(FE) && npm run test:e2e)'

# --- Operational tools (vbctl) -------------------------------------------
# Values loaded here become real environment variables and therefore override
# the application's .env files. Override with: make ... SCRIPTS_ENV=path/to/file
SCRIPTS_ENV ?= .env.scripts
WITH_SCRIPTS_ENV = set -a; if [[ -f "$(SCRIPTS_ENV)" ]]; then source "$(SCRIPTS_ENV)"; fi; set +a;

script-context-budget: ## Measure Ollama context usage (ARGS="...")
	@$(WITH_SCRIPTS_ENV) go run ./cmd/vbctl context-budget $(ARGS)

script-create-benchmark: ## Queue a benchmark API run (ARGS="...")
	@$(WITH_SCRIPTS_ENV) go run ./cmd/vbctl benchmark $(ARGS)

script-thumbnails: ## Create thumbnails for images without one (ARGS="--dry-run")
	@$(WITH_SCRIPTS_ENV) go run ./cmd/vbctl thumbnails $(ARGS)

db-configure-roles: ## Create/update PostgreSQL runtime and read-only group roles
	@$(WITH_SCRIPTS_ENV) test -n "$$MIGRATION_DATABASE_URL" || { echo "Set MIGRATION_DATABASE_URL in $(SCRIPTS_ENV)"; exit 2; }; \
		psql "$$MIGRATION_DATABASE_URL" -v database_name="$${PG__DB:-vision_bill}" -f scripts/configure_db_roles.sql

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
FE := web

fe-install: ## Install locked frontend npm dependencies
	$(NODE) (cd $(FE) && npm ci)

fe-dev: ## Run the Vite dev server on :5173 (proxies /api -> :8080)
	$(NODE) (cd $(FE) && npm run dev)

fe-build: ## Build the SPA into web/out (run fe-install once after dependency changes)
	$(NODE) (cd $(FE) && npm run build)

fe-sync: fe-build ## Copy the built SPA into cmd/server/static (embedded into the binary)
	find cmd/server/static -mindepth 1 ! -name .gitkeep -delete
	cp -r $(FE)/out/. cmd/server/static/

fe-check: ## Frontend static checks (svelte-check + eslint + prettier)
	$(NODE) (cd $(FE) && npm run check && npm run lint && npm run format:check)

fe-test: ## Run frontend unit tests (vitest)
	$(NODE) (cd $(FE) && npm run test)

fe-typecheck-e2e: ## Type-check the Playwright e2e suite
	$(NODE) (cd $(FE) && npm run typecheck:e2e)

fe-verify: fe-check fe-test fe-typecheck-e2e fe-build ## Run frontend checks, unit tests, and a production build

fe: ## Run an npm script in web/ (make fe CMD=<script> [ARGS="..."])
	@test -n "$(CMD)" || { echo "usage: make fe CMD=<npm-script> [ARGS=\"arg1 arg2\"]"; exit 1; }
	$(NODE) (cd $(FE) && npm run $(CMD) $(ARGS))
