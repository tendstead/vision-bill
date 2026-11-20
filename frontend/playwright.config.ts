import { defineConfig } from '@playwright/test';

// The runner owns the real stack and supplies its explicit external endpoints.
// Playwright never adopts or starts an operator service.
const baseURL = process.env.VB_E2E_BASE_URL;
if (!baseURL || process.env.VB_E2E_EXTERNAL_STACK !== '1') {
	throw new Error('Run make fe-test-e2e or supply an explicitly owned external E2E stack');
}

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
		launchOptions: process.env.VB_E2E_CHROMIUM_EXECUTABLE
			? { executablePath: process.env.VB_E2E_CHROMIUM_EXECUTABLE }
			: undefined,
		// The app is i18n (en/de); selectors use English strings, so pin the
		// locale to keep text-based locators deterministic on any host.
		locale: 'en-US',
		// Keep traces on failure to aid debugging of the live-app flow.
		trace: 'retain-on-failure'
	}
});
