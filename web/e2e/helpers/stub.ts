import { test } from '@playwright/test';

const STUB_BASE = 'http://localhost:9123';

export type StubMode = 'ok' | 'down' | 'repair_first' | 'broken' | 'unreadable' | 'slow';

export async function setStubMode(mode: StubMode): Promise<void> {
	const response = await fetch(`${STUB_BASE}/__mode`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ mode })
	});
	if (!response.ok) {
		throw new Error(`stub mode ${mode} failed: ${response.status}`);
	}
}

/**
 * The stub mode is GLOBAL state shared by every test (the suite runs with
 * workers: 1, so files never interleave). Call once per file that mutates the
 * mode: it pins 'ok' before and after the whole file.
 */
export function stubGuard(): void {
	test.beforeAll(async () => {
		await setStubMode('ok');
	});
	test.afterAll(async () => {
		await setStubMode('ok');
	});
}
