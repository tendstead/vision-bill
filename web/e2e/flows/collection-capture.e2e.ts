import { expect, test } from '@playwright/test';
import { createCollection } from '../helpers/api';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';
import { ensureSession, uploadFixture } from '../helpers/ui';

test.describe('collection capture', () => {
	stubGuard();

	test('capturing collection receives new verified receipts automatically', async ({
		page,
		context
	}) => {
		test.setTimeout(180_000);
		await ensureSession(context);
		const collectionId = await createCollection(context, 'Capture Test');

		// Start capturing via the dashboard picker.
		await page.goto('/');
		await page.getByRole('combobox', { name: 'No collection is capturing.' }).click();
		await page.getByRole('option', { name: 'Capture Test' }).click();
		await expect(page.getByRole('button', { name: 'Stop capturing' })).toBeVisible();
		// Header capture dot is visible while capturing.
		await expect(page.locator('a[title="Capturing"]')).toBeVisible();

		// A new verified receipt lands in the collection automatically.
		await uploadFixture(page, context, await loadFixture('a'));
		await page.goto(`/collections/${collectionId}`);
		await expect(page.getByText('Aldi Test', { exact: true })).toBeVisible();

		// Stop capturing: the header dot goes away...
		await page.goto('/');
		await page.getByRole('button', { name: 'Stop capturing' }).click();
		await expect(page.locator('a[title="Capturing"]')).toHaveCount(0);
		// ...and the next receipt is NOT assigned.
		await uploadFixture(page, context, await loadFixture('b'));
		await page.goto(`/collections/${collectionId}`);
		await expect(page.getByText('Billa Test', { exact: true })).toHaveCount(0);
	});
});
