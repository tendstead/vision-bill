import { expect, test } from '@playwright/test';
import { uniqueUsername } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';

// The golden journey for a brand-new user: register via the form -> empty
// dashboard -> upload -> review -> verify -> every surface (dashboard,
// receipts list, search, statistics) reflects the receipt. One long test on
// purpose: the value is the continuity.
test.describe('first run', () => {
	stubGuard();

	test('register, upload, verify and see the receipt everywhere', async ({ page, baseURL }) => {
		test.setTimeout(180_000);
		const username = uniqueUsername('e2e-first');
		await page.goto('/login');
		await page.getByRole('button', { name: 'Create one' }).click();
		await page.getByLabel('Username').fill(username);
		await page.getByLabel('Password').fill('e2e-pw-12345');
		await page.getByRole('button', { name: 'Create account' }).click();
		await expect(page).toHaveURL(baseURL + '/');
		await expect(page.getByText('No verified receipts yet')).toBeVisible();

		const fixture = await loadFixture('a');
		await page.goto('/upload');
		await page.locator('input[type="file"]').first().setInputFiles({
			name: 'a.png',
			mimeType: fixture.mimeType,
			buffer: fixture.bytes
		});
		await page.getByRole('button', { name: 'Upload' }).click();
		await expect(page.getByRole('heading', { name: 'Aldi Test' })).toBeVisible({
			timeout: 30_000
		});

		await page.getByRole('button', { name: 'Verify' }).click();
		await page
			.getByRole('dialog', { name: 'Verify receipt?' })
			.getByRole('button', { name: 'Verify' })
			.click();
		await expect(page.getByText('Receipt verified')).toBeVisible({ timeout: 15_000 });
		await expect(page.getByText('Verified', { exact: true })).toBeVisible();

		// Dashboard reflects the verified receipt.
		await page.goto('/');
		await expect(page.getByText('Verified receipts', { exact: true })).toBeVisible();
		await expect(page.getByText('Last 12 weeks')).toBeVisible();
		await expect(page.getByText('€3.86')).toBeVisible();
		await expect(page.getByText('Aldi Test · 1')).toBeVisible();

		// Receipts list shows it as verified.
		await page.goto('/receipts');
		await expect(page.getByText('Aldi Test')).toBeVisible();
		await expect(page.getByRole('link', { name: 'Aldi Test' }).getByText('Verified')).toBeVisible();

		// Search finds the line item with a price card.
		await page.goto('/search');
		await page.getByRole('searchbox').fill('Milk');
		await expect(page.getByText('€2.00').first()).toBeVisible({ timeout: 15_000 });
		await expect(page.getByRole('link', { name: 'Milk' })).toBeVisible();

		// Statistics are no longer empty.
		await page.goto('/statistics');
		await expect(page.getByText('Total spend')).toBeVisible();
		await expect(page.getByText('No statistics yet')).toHaveCount(0);
	});
});
