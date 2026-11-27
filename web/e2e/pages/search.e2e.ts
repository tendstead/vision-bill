import { expect, test } from '@playwright/test';
import { registerUser } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';
import { uploadFixture } from '../helpers/ui';

test.describe('search page', () => {
	stubGuard();

	test('empty state', async ({ page, context }) => {
		await registerUser(context, 'e2e-search');
		await page.goto('/search');
		await expect(page.getByText('Search your purchases')).toBeVisible();
	});

	test('no results message includes the query', async ({ page, context }) => {
		await registerUser(context, 'e2e-search');
		await page.goto('/search');
		await page.getByRole('searchbox').fill('nothing-here');
		await expect(page.getByText('No purchases match “nothing-here”.')).toBeVisible({
			timeout: 15_000
		});
	});

	test('results show price cards, table and detail link', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('a'));
		await page.goto('/search');
		await page.getByRole('searchbox').fill('Milk');
		await expect(page.getByText('Latest')).toBeVisible({ timeout: 15_000 });
		await expect(page.getByText('€2.00').first()).toBeVisible();
		await expect(page.getByRole('link', { name: 'Milk' })).toBeVisible();

		await page.getByRole('link', { name: 'Milk' }).click();
		await expect(page).toHaveURL(/\/receipts\/[0-9a-f-]{36}/i);
		await expect(page.getByRole('heading', { name: 'Aldi Test' })).toBeVisible();
	});
});
