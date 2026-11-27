import { expect, test } from '@playwright/test';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';
import { uploadFixture } from '../helpers/ui';

test.describe('search price history', () => {
	stubGuard();

	test('same product at two stores shows latest, cheapest and average', async ({
		page,
		context
	}) => {
		test.setTimeout(150_000);
		// Both uploads share ONE user (ensureSession reuses the session), so
		// the price history spans both receipts.
		await uploadFixture(page, context, await loadFixture('a'));
		await uploadFixture(page, context, await loadFixture('b'));

		await page.goto('/search');
		await page.getByRole('searchbox').fill('Milk');
		await expect(page.getByText('Latest')).toBeVisible({ timeout: 15_000 });
		await expect(page.getByRole('paragraph').filter({ hasText: '€2.40' })).toBeVisible(); // Billa, 2026-06-15
		await expect(page.getByText('Cheapest')).toBeVisible();
		await expect(page.getByRole('paragraph').filter({ hasText: '€2.00' })).toBeVisible(); // Aldi, 2026-05-01
		await expect(page.getByText('Average')).toBeVisible();
		await expect(page.getByRole('paragraph').filter({ hasText: '€2.20' })).toBeVisible();
		await expect(page.getByText('Aldi Test')).toBeVisible();
		await expect(page.getByText('Billa Test')).toBeVisible();

		// A product links into its receipt.
		await page.getByRole('link', { name: 'Milk' }).first().click();
		await expect(page).toHaveURL(/\/receipts\/[0-9a-f-]{36}/i);
	});

	test('unverified receipts are not searched', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('c'), { verify: false });
		await page.goto('/search');
		await page.getByRole('searchbox').fill('Coffee');
		await expect(page.getByText('No purchases match “Coffee”.')).toBeVisible({
			timeout: 15_000
		});
	});
});
