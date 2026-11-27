import { expect, type BrowserContext, type Page } from '@playwright/test';
import { registerUser } from './auth';
import type { Fixture } from './fixtures';

const SNACKBAR_TIMEOUT = 15_000;

/** Snackbar renders role="status" (success/info) or role="alert" (error). */
export async function waitForSnackbar(page: Page, text: string | RegExp): Promise<void> {
	await expect(
		page.locator('[role="status"], [role="alert"]').filter({ hasText: text }).first()
	).toBeVisible({ timeout: SNACKBAR_TIMEOUT });
}

export async function confirmDialog(
	page: Page,
	title: string,
	confirmLabel: string
): Promise<void> {
	const dialog = page.getByRole('dialog', { name: title });
	await expect(dialog).toBeVisible();
	await dialog.getByRole('button', { name: confirmLabel }).click();
}

export async function cancelDialog(page: Page, title: string): Promise<void> {
	const dialog = page.getByRole('dialog', { name: title });
	await expect(dialog).toBeVisible();
	await dialog.getByRole('button', { name: 'Cancel' }).click();
	await expect(dialog).toBeHidden();
}

export function receiptIdFromUrl(page: Page): string {
	return page.url().split('/').pop() ?? '';
}

export async function ensureSession(context: BrowserContext): Promise<void> {
	const me = await context.request.get('/api/v1/auth/me');
	if (me.status() === 200) return; // reuse the context's existing user
	await registerUser(context, 'e2e-up');
}

/**
 * Authenticates the context (fresh user if needed), uploads `fixture` through
 * the real /upload page and — unless opts.verify === false — confirms the
 * verify dialog. Returns the receipt id (read from the URL).
 */
export async function uploadFixture(
	page: Page,
	context: BrowserContext,
	fixture: Fixture,
	opts: { bypass?: boolean; verify?: boolean } = {}
): Promise<string> {
	await ensureSession(context);
	await page.goto('/upload');
	if (opts.bypass) {
		await page.getByLabel(/Skip review/).check();
	}
	await page
		.locator('input[type="file"]')
		.first()
		.setInputFiles({
			name: `${fixture.name}.png`,
			mimeType: fixture.mimeType,
			buffer: fixture.bytes
		});
	await page.getByRole('button', { name: 'Upload' }).click();
	await expect(page.getByRole('heading', { name: fixture.expected.merchant_name })).toBeVisible({
		timeout: 30_000
	});
	if (opts.bypass) {
		return receiptIdFromUrl(page);
	}
	if (opts.verify !== false) {
		await page.getByRole('button', { name: 'Verify' }).click();
		await confirmDialog(page, 'Verify receipt?', 'Verify');
		await waitForSnackbar(page, 'Receipt verified');
	}
	return receiptIdFromUrl(page);
}
