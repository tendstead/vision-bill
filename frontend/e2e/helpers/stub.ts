import { testEnvironment } from './environment';
import { test } from '@playwright/test';

// Submission polling lives in helpers/api.ts; it never changes this global mode.
const STUB_BASE = testEnvironment('VB_E2E_STUB_URL');

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

export async function setStubScenario(
	scenario: 'complete' | 'partial' | 'policy_unknown' | 'overlap' | 'direct_merchant_mismatch'
): Promise<void> {
	const response = await fetch(`${STUB_BASE}/__scenario`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ scenario })
	});
	if (!response.ok) throw new Error(`stub scenario failed: ${response.status}`);
}

export function stubGuard(): void {
	test.beforeEach(async () => {
		await setStubMode('ok');
		await setStubScenario('complete');
	});
	test.afterEach(async () => {
		await setStubMode('ok');
		await setStubScenario('complete');
	});
}
