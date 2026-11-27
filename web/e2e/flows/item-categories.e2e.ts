import { expect, test } from '@playwright/test';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';
import { uploadFixture } from '../helpers/ui';

test.describe('remembered item categories', () => {
	stubGuard();

	test('an explicit choice is remembered, while the following ordinary save does not teach again', async ({
		page,
		context
	}) => {
		const fixture = await loadFixture('a');
		const receiptId = await uploadFixture(page, context, fixture, { verify: false });
		await page.getByLabel('Spending category', { exact: true }).first().selectOption('milk');
		await page.getByRole('button', { name: 'Save', exact: true }).click();
		await expect(page.getByText('Changes saved')).toBeVisible();
		await expect(page.getByRole('button', { name: 'Save', exact: true })).toHaveCount(0);

		await page.getByLabel('Qty', { exact: true }).first().fill('2');
		const saving = page.waitForRequest(
			(request) => request.method() === 'PUT' && request.url().endsWith(`/receipts/${receiptId}`)
		);
		await page.getByRole('button', { name: 'Save', exact: true }).click();
		const request = await saving;
		expect(request.postDataJSON().line_items[0].remember_category).not.toBe(true);
		await expect(page.getByRole('button', { name: 'Save', exact: true })).toHaveCount(0);

		const repeatedId = await uploadFixture(page, context, fixture, { verify: false });
		await expect(page.getByLabel('Spending category', { exact: true }).first()).toHaveValue('milk');
		const detail = await (await page.request.get(`/api/v1/receipts/${repeatedId}`)).json();
		expect(detail.line_items[0].category_source).toBe('history');
		expect(detail.line_items[0].original_description).toBe(
			fixture.expected.line_items[0].description
		);
	});

	test('verified item categories show spending shares and a selectable monthly trend', async ({
		page,
		context
	}) => {
		await uploadFixture(page, context, await loadFixture('a'), { verify: false });
		await page.getByLabel('Date', { exact: true }).fill(new Date().toISOString().slice(0, 10));
		await page.getByLabel('Spending category', { exact: true }).nth(0).selectOption('milk');
		await page.getByLabel('Spending category', { exact: true }).nth(1).selectOption('bread_bakery');
		await page.getByRole('button', { name: 'Save & verify', exact: true }).click();
		await expect(page.getByText('Receipt verified', { exact: true })).toBeVisible();
		await page.goto('/statistics');
		await expect(page.getByRole('heading', { name: 'Spending by item category' })).toBeVisible();
		await expect(page.getByText('57.14%', { exact: true }).first()).toBeVisible();
		await page.getByLabel('Category to follow').selectOption('milk');
		await expect(page.getByRole('table')).toContainText('Milk');
		await page.getByLabel('Months', { exact: true }).selectOption('3');
		await expect(page.getByRole('table').locator('tbody tr')).toHaveCount(3);
		await expect(page.getByRole('table')).toContainText('(partial)');
	});
});
