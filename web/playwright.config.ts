import { defineConfig } from '@playwright/test';

// The E2E suite runs against a real, self-started vision-bill stack
// (docker-compose.e2e.yml: app + postgres + deterministic LLM stub).
// Set VB_E2E_BASE_URL to point at an already-running stack instead.
const baseURL = process.env.VB_E2E_BASE_URL ?? 'http://localhost:53625';

export default defineConfig({
	testDir: './e2e',
	testMatch: /.*\.e2e\.ts$/,
	timeout: 60_000,
	// The LLM stub's mode is global state shared by every test, so the suite
	// must run serially — parallel workers would interleave mode changes.
	// (~60 fast tests against a local stack: a few minutes, not a bottleneck.)
	fullyParallel: false,
	workers: 1,
	/* Retry on CI only */
	retries: process.env.CI ? 2 : 0,
	reporter: [['list']],
	use: {
		baseURL,
		// The app is i18n (en/de); selectors use English strings, so pin the
		// locale to keep text-based locators deterministic on any host.
		locale: 'en-US',
		// Keep traces on failure to aid debugging of the live-app flow.
		trace: 'retain-on-failure'
	},
	webServer: {
		// Fallback for running `playwright test` directly: `make fe-test-e2e`
		// starts the stack itself and tears it down after the run, in which
		// case this only reuses it. Running bare `playwright test` leaves the
		// stack up (stop it with `make e2e-down`).
		// Runs from rootDir (web/), so ../ = repo root.
		command: 'docker compose -f ../docker-compose.e2e.yml up -d --build',
		url: `${baseURL}/api/v1/system/ui-config`,
		reuseExistingServer: true,
		// First run builds the multi-stage app image (Node SPA + Python) — can
		// take several minutes; cached afterwards.
		timeout: 600_000
	}
});
