import { expect, test } from '@playwright/test';
import { sha256 } from '@noble/hashes/sha256';
import { bytesToHex, utf8ToBytes } from '@noble/hashes/utils';
import { registerUser } from '../helpers/auth';
import { awaitSubmission } from '../helpers/api';
import { testEnvironment } from '../helpers/environment';
import { loadFixture } from '../helpers/fixtures';
import { setStubMode, stubGuard } from '../helpers/stub';

// The canonical uv runner supplies its Python on PATH; use only runner-owned endpoints.
async function history(action: 'seed' | 'inspect', id: string, ownerId: string) {
	const module = 'node:child_process';
	const { execFileSync } = (await import(module)) as {
		execFileSync: (file: string, args: string[], options: { encoding: string }) => string;
	};
	return JSON.parse(
		execFileSync('python', ['../tests/e2e/policy_history.py', action, id, ownerId], {
			encoding: 'utf8'
		})
	);
}

test.describe('explicit current policy adoption', () => {
	stubGuard();
	test('held history adopts once after lost acceptance; all formatter, repair and classification calls remain text only', async ({
		page,
		context,
		browser
	}) => {
		test.setTimeout(120_000);
		test.skip(testEnvironment('VB_E2E_POLICY') === 'off', 'Requires configured OCR');
		await registerUser(context, 'policy-owner');
		const ownerId = (await (await context.request.get('/api/v1/auth/me')).json()).id;
		const fixture = await loadFixture('a');
		await setStubMode('down');
		const form = new FormData();
		form.append(
			'receipt',
			new File([new Uint8Array(fixture.bytes)], 'a.png', { type: fixture.mimeType })
		);
		const accepted = await context.request.post(
			'/api/v1/images?model_id=e2e-text&allow_ocr_fallback=true',
			{
				headers: { 'Idempotency-Key': 'historical-policy-upload' },
				multipart: form
			}
		);
		expect(accepted.status()).toBe(202);
		const { id } = await accepted.json();
		const unseeded = await history('inspect', id, ownerId);
		const wrongOwner = crypto.randomUUID();
		for (const action of ['inspect', 'seed'] as const) {
			await expect
				.soft(history(action, id, wrongOwner))
				.rejects.toThrow('owned submission not found');
			expect(await history('inspect', id, ownerId)).toEqual(unseeded);
		}
		const before = await history('seed', id, ownerId);
		expect(before.user_id).toBe(ownerId);
		await setStubMode('ok');
		expect((await context.request.post('/api/v1/images/analyze')).status()).toBe(200);
		const url = `/api/v1/images/${id}`;
		await expect
			.poll(async () => (await (await context.request.get(url)).json()).wait_reason)
			.toBe('execution_policy_changed');
		const held = await (await context.request.get(url)).json();
		expect(held.attempt_count).toBe(0);
		expect(held.receipt_id).toBeNull();
		expect(
			(
				await context.request.post(`${url}/retry`, {
					data: {
						expected_generation: held.generation + 1,
						request_key: 'stale-generation',
						policy: 'current'
					}
				})
			).status()
		).toBe(409);
		expect(await (await context.request.get(url)).json()).toEqual(held);
		expect(
			(await (await fetch(`${testEnvironment('VB_E2E_STUB_URL')}/__requests`)).json()).requests
		).toEqual([]);
		const original = await (await context.request.get(`${url}/file`)).body();
		expect(Array.from(original)).toEqual(Array.from(fixture.bytes));
		const outsider = await browser.newContext({ baseURL: testEnvironment('VB_E2E_BASE_URL') });
		try {
			await registerUser(outsider, 'policy-outsider');
			expect(
				(
					await outsider.request.post(`${url}/retry`, {
						data: {
							expected_generation: held.generation,
							request_key: 'foreign',
							policy: 'current'
						}
					})
				).status()
			).toBe(404);
		} finally {
			await outsider.close();
		}
		await page.goto(`/upload?job=${id}`);
		await expect(page.getByText(/execution_policy_changed/)).toBeVisible();
		const adopt = page.getByRole('button', { name: 'Retry using current server settings' });
		await expect(adopt).toBeVisible();
		await setStubMode('repair_first');
		const requests: { body: Record<string, unknown>; result: Record<string, unknown> }[] = [];
		let release!: () => void;
		const pendingGets = new Promise<void>((resolve) => {
			release = resolve;
		});
		// Delay polling after acceptance so the explicit retry can replay its original intent.
		await page.route(`**${url}`, async (route) => {
			await pendingGets;
			await route.continue();
		});
		await page.route(`**${url}/retry`, async (route) => {
			const response = await route.fetch();
			expect(response.status()).toBe(202);
			requests.push({ body: route.request().postDataJSON(), result: await response.json() });
			if (requests.length === 1)
				await route.fulfill({ status: 503, json: { detail: 'Lost accepted response' } });
			else {
				await route.fulfill({ response });
				release();
			}
		});
		try {
			await adopt.click();
			await expect(page.getByRole('alert')).toContainText('Lost accepted response');
			await adopt.click();
			await expect.poll(() => requests.length).toBe(2);
			expect(requests[1]).toEqual(requests[0]);
			expect(requests[0].body).toEqual({
				expected_generation: held.generation,
				request_key: expect.any(String),
				policy: 'current'
			});
			expect(requests[0].body.request_key).not.toBe('');
		} finally {
			release();
		}
		const completed = await awaitSubmission(context, id);
		expect(completed.status).toBe('analyzed');
		expect(completed.generation).toBe(held.generation + 1);
		expect(completed.attempt_count).toBe(1);
		const receiptUrl = `/api/v1/receipts/${completed.receipt_id}`;
		const receipt = await (await context.request.get(receiptUrl)).json();
		for (const body of [
			{ ...requests[0].body, policy: 'preserve' },
			{ expected_generation: completed.generation, request_key: 'completed', policy: 'current' }
		])
			expect((await context.request.post(`${url}/retry`, { data: body })).status()).toBe(409);
		expect(await (await context.request.get(`${url}/file`)).body()).toEqual(original);
		expect(await (await context.request.get(receiptUrl)).json()).toEqual(receipt);
		const after = await history('inspect', id, ownerId);
		expect(after.user_id).toBe(ownerId);
		expect(after.generation).toBe(completed.generation);
		for (const key of [
			'source_manifest',
			'requested_options',
			'request_fingerprint',
			'idempotency_key'
		])
			expect(after[key]).toEqual(before[key]);
		expect(after.requested_options.allow_ocr_fallback).toBe(true);
		expect(after.effective_options).toMatchObject({
			execution_policy_version: 4,
			managed_binding_sha256: null,
			allow_ocr_fallback: false,
			ocr: { enabled: true, allow_vision_fallback: false }
		});
		// Use the app's Compose-side stub endpoint, which differs from the published test URL.
		const inferenceIdentity = `openai:${bytesToHex(sha256(utf8ToBytes('http://llm-stub:9123/v1')))}`;
		expect(after.effective_options.inference_identity).toBe(inferenceIdentity);
		expect(after.inference_identity).toBe(inferenceIdentity);
		expect(after.checkpoint_id).toMatch(/^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/);
		const checkpoint = {
			checkpoint_id: after.checkpoint_id,
			producer_generation: completed.generation,
			producer_attempt_number: 1,
			inference_identity: inferenceIdentity
		};
		expect(after.legacy_events).toEqual([
			{
				event: 'execution_policy_adopted',
				generation: held.generation,
				effective_options: before.effective_options,
				model_id: 'e2e-text',
				requested_options: before.requested_options,
				request_fingerprint: before.request_fingerprint
			},
			{ event: 'unmanaged_classification_started', ...checkpoint },
			{
				event: 'unmanaged_classification_settled',
				...checkpoint,
				settlement: 'provider_returned'
			}
		]);
		const calls = (await (await fetch(`${testEnvironment('VB_E2E_STUB_URL')}/__requests`)).json())
			.requests as { kind: string; image_sha256: string | null; served: string }[];
		expect(calls.filter((call) => call.kind === 'ocr').length).toBeGreaterThanOrEqual(2);
		expect(calls.some((call) => call.kind === 'ocr' && call.served === 'broken')).toBe(true);
		expect(calls.some((call) => call.kind === 'classification')).toBe(true);
		expect(calls.some((call) => call.served === 'broken')).toBe(true);
		for (const call of calls) expect(call.image_sha256).toBeNull();
	});
});
