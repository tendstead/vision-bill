import { expect, test } from '@playwright/test';
import { registerUser } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';
import { uploadFixture } from '../helpers/ui';

test.describe('receipts list', () => {
	stubGuard();

	test('empty state', async ({ page, context }) => {
		await registerUser(context, 'e2e-list');
		await page.goto('/receipts');
		await expect(page.getByText('No receipts yet.')).toBeVisible();
	});

	test('lists receipts with status badges and totals', async ({ page, context }) => {
		test.setTimeout(150_000);
		await uploadFixture(page, context, await loadFixture('a')); // verified
		await uploadFixture(page, context, await loadFixture('b'), { verify: false }); // unverified
		await page.goto('/receipts');
		await expect(page.getByText('Aldi Test')).toBeVisible();
		await expect(page.getByText('€3.86')).toBeVisible();
		// Target the receipt link whose accessible name includes the badge.
		await expect(page.getByRole('link', { name: /Aldi Test.*Verified/ })).toBeVisible();
		await expect(page.getByText('Billa Test')).toBeVisible();
		await expect(page.getByRole('link', { name: /Billa Test.*Unverified/ })).toBeVisible();
	});

	test('search filter narrows results and shows no-results', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('a'));
		await page.goto('/receipts');
		const search = page.getByRole('searchbox', { name: 'Search receipts…' });
		await search.fill('Aldi');
		await expect(page.getByText('Aldi Test')).toBeVisible();
		await search.fill('zzz-no-match');
		await expect(page.getByText('No receipts match your search.')).toBeVisible({
			timeout: 15_000
		});
	});

	test('status filters', async ({ page, context }) => {
		test.setTimeout(150_000);
		await uploadFixture(page, context, await loadFixture('a'));
		await uploadFixture(page, context, await loadFixture('b'), { verify: false });
		await page.goto('/receipts');
		await page.getByRole('button', { name: 'Verified', exact: true }).click();
		await expect(page.getByText('Aldi Test')).toBeVisible();
		await expect(page.getByText('Billa Test')).toHaveCount(0);
		await page.getByRole('button', { name: 'Unverified', exact: true }).click();
		await expect(page.getByText('Billa Test')).toBeVisible();
		await expect(page.getByText('Aldi Test')).toHaveCount(0);
	});

	test('delete receipt via confirm dialog', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('a'));
		await page.goto('/receipts');
		await page.getByRole('button', { name: 'Remove' }).click();
		const dialog = page.getByRole('dialog', { name: 'Delete receipt?' });
		await expect(dialog).toBeVisible();
		await expect(dialog.getByText('Aldi Test')).toBeVisible();
		await dialog.getByRole('button', { name: 'Delete' }).click();
		await expect(page.getByText('Receipt deleted')).toBeVisible({ timeout: 15_000 });
		await expect(page.getByText('No receipts yet.')).toBeVisible();
	});
});
