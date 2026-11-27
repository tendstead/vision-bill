import { expect, test, type BrowserContext, type Page } from '@playwright/test';
import { registerUser } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { setStubMode, stubGuard } from '../helpers/stub';
import { confirmDialog } from '../helpers/ui';

/** Upload one image while the stub is in `mode`; land on the queue page. */
async function queueAnImage(
	page: Page,
	_context: BrowserContext,
	mode: 'down' | 'broken'
): Promise<void> {
	const fixture = await loadFixture('a');
	await setStubMode(mode);
	await page.goto('/upload');
	await page.locator('input[type="file"]').first().setInputFiles({
		name: 'a.png',
		mimeType: fixture.mimeType,
		buffer: fixture.bytes
	});
	await page.getByRole('button', { name: 'Upload' }).click();
	if (mode === 'down') {
		await expect(page.locator('p:has-text("Added to the analysis queue")')).toBeVisible();
	} else {
		await expect(page.getByText('Upload failed')).toBeVisible({ timeout: 30_000 });
	}
	await page.goto('/queue');
}

test.describe('queue page', () => {
	stubGuard();

	// Reset the stub to ok before each test — tests leave it in various modes.
	test.beforeEach(async () => {
		await setStubMode('ok');
	});

	test('empty state disables the analyze action', async ({ page, context }) => {
		await registerUser(context, 'e2e-queue');
		await page.goto('/queue');
		await expect(page.getByText('The queue is empty')).toBeVisible();
		await expect(page.getByRole('button', { name: 'Analyze now' })).toBeDisabled();
	});

	test('pending image can be analyzed', async ({ page, context }) => {
		test.setTimeout(120_000);
		await registerUser(context, 'e2e-queue');
		await queueAnImage(page, context, 'down');
		await expect(page.getByText('Pending', { exact: true })).toBeVisible();

		await setStubMode('ok');
		await page.getByRole('button', { name: 'Analyze now' }).click();
		// Wait for the queue to actually empty (more reliable than the snackbar).
		await expect(page.getByText('The queue is empty')).toBeVisible({ timeout: 60_000 });
	});

	test('analyze now with provider down keeps the image pending', async ({ page, context }) => {
		test.setTimeout(120_000);
		await registerUser(context, 'e2e-queue');
		await queueAnImage(page, context, 'down');
		// Contract: a down provider is NOT an error — the cycle returns zero
		// results and the image stays pending for the next cycle.
		// (The queue page only shows a snackbar when results.length > 0,
		// so we verify no error appeared and the image stays pending.)
		await page.getByRole('button', { name: 'Analyze now' }).click();
		await expect(page.getByText('Pending', { exact: true })).toBeVisible({ timeout: 30_000 });
	});

	test('failed image shows the error and can be deleted', async ({ page, context }) => {
		test.setTimeout(150_000);
		await registerUser(context, 'e2e-queue');
		await queueAnImage(page, context, 'broken');
		await expect(page.getByText('Failed', { exact: true })).toBeVisible();

		await page.getByRole('button', { name: 'Remove' }).click();
		await confirmDialog(page, 'Delete image?', 'Delete');
		await expect(page.getByText('Image deleted')).toBeVisible({ timeout: 15_000 });
		await expect(page.getByText('The queue is empty')).toBeVisible();
	});
});
