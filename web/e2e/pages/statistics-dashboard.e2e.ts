import { expect, test } from '@playwright/test';
import { registerUser } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';
import { uploadFixture } from '../helpers/ui';

test.describe('statistics + dashboard', () => {
	stubGuard();

	test('empty states on both pages', async ({ page, context }) => {
		await registerUser(context, 'e2e-stats');
		await page.goto('/');
		await expect(page.getByText('No verified receipts yet')).toBeVisible();
		await page.goto('/statistics');
		await expect(page.getByText('No statistics yet')).toBeVisible();
		await expect(page.getByText('No collections yet')).toBeVisible();
	});

	test('populated statistics render all sections', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('a'));
		await page.goto('/statistics');
		for (const label of [
			'Verified receipts',
			'Total spend',
			'Spend by category',
			'Spend by merchant',
			'Spend by payment method',
			'Spend by weekday',
			'Weekly spending'
		]) {
			await expect(page.getByText(label).first()).toBeVisible();
		}
	});

	test('collection filter changes the data', async ({ page, context }) => {
		test.setTimeout(150_000);
		await registerUser(context, 'e2e-stats-coll');
		// Navigate to authenticate the page (cookie is shared between
		// context.request and the browser context).
		await page.goto('/collections');

		// Use page.request so the session cookie is available.
		const collResp = await page.request.post('/api/v1/collections', {
			data: { name: 'Stats Coll', color: '#2563EB', start_date: null, end_date: null }
		});
		const collBody = (await collResp.json()) as { id: string };
		const collectionId = collBody.id;

		const receiptId = await uploadFixture(page, context, await loadFixture('a'));

		await page.goto('/statistics');
		const picker = page.getByRole('combobox', { name: 'All collections' });
		await picker.click();
		await page.getByRole('option', { name: 'Stats Coll' }).click();
		// The receipt is not (yet) in the collection: filtered stats are empty.
		await expect(page.getByText('No statistics yet')).toBeVisible();

		// Assign the receipt via page.request (has session cookie).
		await page.request.post(`/api/v1/collections/${collectionId}/receipts/${receiptId}`);
		// Re-navigate to pick up the updated data with the collection filter.
		await page.goto('/statistics');
		await picker.click();
		await page.getByRole('option', { name: 'Stats Coll' }).click();
		await expect(page.getByText('Total spend')).toBeVisible();
	});

	test('multi-currency groups (EUR + USD)', async ({ page, context }) => {
		test.setTimeout(150_000);
		await uploadFixture(page, context, await loadFixture('a'));
		await uploadFixture(page, context, await loadFixture('c'));
		await page.goto('/');
		await expect(page.getByText('Last 12 weeks').first()).toBeVisible();
		await expect(page.getByText('€3.86')).toBeVisible();
		await expect(page.getByText('$5.50').first()).toBeVisible();
	});
});
