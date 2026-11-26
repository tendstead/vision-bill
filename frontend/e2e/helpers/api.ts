import { expect, type BrowserContext } from '@playwright/test';

export interface LlmSettings {
	provider: string;
	host: string;
	model_name: string;
	temperature: number;
}

export interface SettingsBody {
	llm: LlmSettings;
	allow_registration: boolean;
}

export async function getSettings(context: BrowserContext): Promise<SettingsBody> {
	const response = await context.request.get('/api/v1/system/settings');
	if (response.status() !== 200) throw new Error(`get settings failed: ${response.status()}`);
	const body = (await response.json()) as { llm: LlmSettings; allow_registration: boolean };
	return { llm: body.llm, allow_registration: body.allow_registration };
}

export async function putSettings(
	context: BrowserContext,
	changes: { temperature?: number; allow_registration?: boolean }
): Promise<SettingsBody> {
	const current = await getSettings(context);
	const next: SettingsBody = {
		llm:
			changes.temperature === undefined
				? current.llm
				: { ...current.llm, temperature: changes.temperature },
		allow_registration: changes.allow_registration ?? current.allow_registration
	};
	const response = await context.request.put('/api/v1/system/settings', { data: next });
	if (response.status() !== 200) {
		throw new Error(`put settings failed: ${response.status()} ${await response.text()}`);
	}
	return next;
}

/** Create a collection through the API; returns its id. */
export async function createCollection(context: BrowserContext, name: string): Promise<string> {
	// Pass session cookie via explicit header — APIRequestContext.setCookie
	// is untyped in @types/playwright even though it works at runtime.
	const cookies = await context.cookies();
	const sessionCookie = cookies.find((c) => c.name === 'vb_session');
	const headers: Record<string, string> = {};
	if (sessionCookie) {
		headers.Cookie = `vb_session=${sessionCookie.value}`;
	}
	const response = await context.request.post('/api/v1/collections', {
		data: { name, color: '#2563EB', start_date: null, end_date: null },
		headers: Object.keys(headers).length > 0 ? headers : undefined
	});
	if (response.status() !== 201) {
		throw new Error(`create collection failed: ${response.status()} ${await response.text()}`);
	}
	const body = (await response.json()) as { id: string };
	return body.id;
}

/** Upload raw bytes through the API; returns status + parsed body. */
export async function uploadViaApi(
	context: BrowserContext,
	name: string,
	bytes: Uint8Array,
	mimeType = 'image/png',
	options: {
		key?: string;
		wait?: boolean;
		photos?: { name: string; bytes: Uint8Array; mimeType: string }[];
		allow_ocr_fallback?: boolean;
	} = {}
): Promise<{ status: number; body: Record<string, unknown> }> {
	const form = new FormData();
	for (const photo of options.photos ?? [{ name, bytes, mimeType }]) {
		form.append(
			'receipt',
			new File([new Uint8Array(photo.bytes)], photo.name, { type: photo.mimeType })
		);
	}
	const response = await context.request.post(
		'/api/v1/images' +
			(options.allow_ocr_fallback === undefined
				? ''
				: `?allow_ocr_fallback=${options.allow_ocr_fallback}`),
		{
			headers: { 'Idempotency-Key': options.key ?? crypto.randomUUID() },
			multipart: form
		}
	);
	const body = (await response.json()) as Record<string, unknown>;
	if (response.status() !== 202) return { status: response.status(), body };
	expect(typeof body.id).toBe('string');
	return {
		status: response.status(),
		body: options.wait === false ? body : await awaitSubmission(context, body.id as string)
	};
}

/** Await the durable outcome, never treat acceptance as a usable receipt. */
export async function awaitSubmission(
	context: BrowserContext,
	id: string,
	budget = 120_000
): Promise<Record<string, unknown>> {
	let outcome: Record<string, unknown> = {};
	await expect
		.poll(
			async () => {
				const response = await context.request.get(`/api/v1/images/${id}`);
				if (!response.ok()) throw new Error(`get submission failed: ${response.status()}`);
				outcome = await response.json();
				return outcome.terminal;
			},
			{ timeout: budget, intervals: [200, 500, 1000] }
		)
		.toBe(true);
	return outcome;
}

/** Calling tests must enumerate expected findings and their review comments. */
export async function verifyReceiptViaApi(
	context: BrowserContext,
	id: string,
	expected: { id?: string; code: string; comment: string }[] = []
): Promise<void> {
	const response = await context.request.get(`/api/v1/receipts/${id}`);
	expect(response.ok()).toBeTruthy();
	const { receipt } = await response.json();
	const findings = receipt.findings as { id: string; code: string; severity: string }[];
	expect(findings.filter((finding) => finding.severity === 'integrity')).toEqual([]);
	const reviews = findings.filter((finding) => finding.severity === 'review');
	expect(reviews.map((finding) => finding.code).sort()).toEqual(
		expected.map((finding) => finding.code).sort()
	);
	const remaining = [...reviews];
	const acknowledgements = expected.map((finding) => {
		expect(finding.comment.trim()).not.toBe('');
		const index = remaining.findIndex(
			(actual) =>
				actual.code === finding.code && (finding.id === undefined || actual.id === finding.id)
		);
		expect(index, `Expected review finding ${finding.code}`).toBeGreaterThanOrEqual(0);
		const [actual] = remaining.splice(index, 1);
		return { finding_id: actual.id, comment: finding.comment };
	});
	const verified = await context.request.post(`/api/v1/receipts/${id}/verify`, {
		data: {
			expected_operation_revision: receipt.operation_revision,
			expected_review_revision: receipt.review_revision,
			acknowledgements
		}
	});
	expect(verified.status()).toBe(200);
	expect((await verified.json()).verified).toBe(true);
}

export async function qualifyGold(context: BrowserContext, id: string): Promise<void> {
	const response = await context.request.get(`/api/v1/benchmarks/references/${id}`);
	expect(response.status()).toBe(200);
	const preview = await response.json();
	const qualified = await context.request.post(`/api/v1/benchmarks/references/${id}`, {
		data: {
			expected_review_revision: preview.review_revision,
			expected_document_sha256: preview.document_sha256
		}
	});
	expect(qualified.status()).toBe(200);
}
