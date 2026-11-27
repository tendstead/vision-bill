import { expect, test } from '@playwright/test';
import { registerUser } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { setStubMode, stubGuard } from '../helpers/stub';

test.describe('queued upload with provider down', () => {
	stubGuard();

	test('provider down -> queued -> analyze now recovers', async ({ page, context }) => {
		test.setTimeout(150_000);
		const fixture = await loadFixture('a');
		await setStubMode('down');
		await registerUser(context, 'e2e-queued');

		await page.goto('/upload');
		await page.locator('input[type="file"]').first().setInputFiles({
			name: 'a.png',
			mimeType: fixture.mimeType,
			buffer: fixture.bytes
		});
		await page.getByRole('button', { name: 'Upload' }).click();
		await expect(page.locator('p.font-medium.text-on-secondary-container')).toBeVisible();

		// The header queue badge appears after a navigation (refetch).
		await page.goto('/');
		await expect(page.locator('a[title="Queue"]')).toHaveText('1');

		await page.goto('/queue');
		await expect(page.getByText('a.png')).toBeVisible();
		await expect(page.getByText('Pending', { exact: true })).toBeVisible();

		await setStubMode('ok');
		await page.getByRole('button', { name: 'Analyze now' }).click();
		await expect(page.getByText('1 image(s) processed')).toBeVisible({ timeout: 30_000 });
		await expect(page.getByText('The queue is empty')).toBeVisible();

		// The receipt exists and is still unverified (the queue never verifies).
		await page.goto('/receipts');
		await expect(page.getByText('Aldi Test')).toBeVisible();
		await expect(
			page.getByRole('link', { name: 'Aldi Test' }).locator('..').getByText('Unverified')
		).toBeVisible();
	});
});
