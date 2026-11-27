import { expect, test } from '@playwright/test';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';
import { uploadFixture } from '../helpers/ui';

test.describe('upload, edit, save, verify', () => {
	stubGuard();

	test('editor is pre-filled with ground truth and plain save persists', async ({
		page,
		context
	}) => {
		test.setTimeout(120_000);
		const fixture = await loadFixture('a');
		await uploadFixture(page, context, fixture, { verify: false });

		// Ground-truth pre-fill.
		await expect(page.locator('#re-merchant')).toHaveValue('Aldi Test');
		await expect(page.locator('#re-date')).toHaveValue('2026-05-01');
		await expect(page.locator('#re-currency')).toHaveValue('EUR');
		// Line-item order is by DB UUID (non-deterministic), and a save
		// re-creates the line items with fresh UUIDs (so the order can change
		// across a reload). Address items by description, never by #li-0 index.
		const lineIndex = (desc: string) =>
			page
				.locator('input[placeholder="Description"]')
				.evaluateAll((els, d) => (els as HTMLInputElement[]).findIndex((e) => e.value === d), desc);
		expect(await lineIndex('Milk')).toBeGreaterThanOrEqual(0);
		expect(await lineIndex('Bread')).toBeGreaterThanOrEqual(0);
		await expect(page.locator('#tax-0-name')).toHaveValue('VAT');
		await expect(page.locator('#tax-0-amount')).toHaveValue('0.36');

		// Dirty state reveals the action bar.
		await page.locator('#re-merchant').fill('Aldi Test Edited');
		await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeVisible();

		// Edit category, currency, quantity, a new tag and the tax amount.
		await page.locator('#re-category').selectOption('electronics');
		await page.locator('#re-currency').fill('USD');
		const milkIdx = await lineIndex('Milk');
		await page.locator(`#li-${milkIdx}-qty`).fill('2');
		const milkTags = page.locator('details').filter({ hasText: 'Optional tags' }).nth(milkIdx);
		await milkTags.locator('summary').click();
		await milkTags.getByRole('button', { name: 'Tags' }).click();
		await milkTags.getByPlaceholder('Search or add tags…').fill('test tag');
		await milkTags.getByPlaceholder('Search or add tags…').press('Enter');
		await expect(
			page.locator('span.rounded-full.bg-primary-container').filter({ hasText: 'test tag' })
		).toBeVisible();
		await page.locator('#tax-0-amount').fill('0.50');

		await page.getByRole('button', { name: 'Save', exact: true }).click();
		await expect(page.getByText('Changes saved')).toBeVisible({ timeout: 15_000 });

		// Everything survived a reload.
		await page.reload();
		await expect(page.locator('#re-merchant')).toHaveValue('Aldi Test Edited');
		await expect(page.locator('#re-category')).toHaveValue('electronics');
		const milkIdx2 = await lineIndex('Milk');
		await expect(page.locator(`#li-${milkIdx2}-qty`)).toHaveValue('2');
		await page.getByText('Optional tags').nth(milkIdx2).click();
		await expect(
			page.locator('span.rounded-full.bg-primary-container').filter({ hasText: 'test tag' })
		).toBeVisible();
		await expect(page.locator('#tax-0-amount')).toHaveValue('0.50');
	});

	test('save & verify verifies the edited receipt', async ({ page, context }) => {
		test.setTimeout(120_000);
		const fixture = await loadFixture('b');
		await uploadFixture(page, context, fixture, { verify: false });
		await page.locator('#re-merchant').fill('Billa Test Edited');
		await page.getByRole('button', { name: 'Save & verify' }).click();
		await expect(page.getByText('Receipt verified')).toBeVisible({ timeout: 15_000 });
		await expect(page.getByText('Verified', { exact: true })).toBeVisible();
		await expect(page.locator('#re-merchant')).toHaveValue('Billa Test Edited');
	});
});
