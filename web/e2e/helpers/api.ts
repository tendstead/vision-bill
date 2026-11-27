import type { BrowserContext } from '@playwright/test';

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
	mimeType = 'image/png'
): Promise<{ status: number; body: Record<string, unknown> }> {
	const response = await context.request.post('/api/v1/images', {
		multipart: { receipt: { name, mimeType, buffer: bytes } }
	});
	return { status: response.status(), body: (await response.json()) as Record<string, unknown> };
}
