import type { BrowserContext } from '@playwright/test';

export interface Creds {
	username: string;
	password: string;
}

/** Bootstrap admin seeded by the e2e stack (AUTH__BOOTSTRAP_USERNAME/PASSWORD). */
export const ADMIN: Creds = { username: 'admin', password: 'admin-e2e-pass' };

export function uniqueUsername(prefix = 'e2e'): string {
	return `${prefix}-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
}

/**
 * Register a fresh isolated user through the real API. The session cookie is
 * set on the given context, so its pages are immediately authenticated.
 */
export async function registerUser(context: BrowserContext, prefix = 'e2e'): Promise<Creds> {
	const creds: Creds = { username: uniqueUsername(prefix), password: 'e2e-pw-12345' };
	const response = await context.request.post('/api/v1/auth/register', { data: creds });
	if (response.status() !== 201) {
		throw new Error(
			`register ${creds.username} failed: ${response.status()} ${await response.text()}`
		);
	}
	return creds;
}

/** Log in with existing credentials (e.g. the bootstrap admin). */
export async function loginAs(context: BrowserContext, creds: Creds): Promise<void> {
	const response = await context.request.post('/api/v1/auth/login', { data: creds });
	if (response.status() !== 200) {
		throw new Error(`login ${creds.username} failed: ${response.status()}`);
	}
}

export async function loginAdmin(context: BrowserContext): Promise<void> {
	await loginAs(context, ADMIN);
}
