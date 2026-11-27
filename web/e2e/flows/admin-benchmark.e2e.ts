import { expect, test } from '@playwright/test';
import { loginAdmin } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';
import { uploadFixture } from '../helpers/ui';

test.describe('admin: benchmarks', () => {
	stubGuard();

	test('create a run and watch it complete in the results view', async ({ page, context }) => {
		test.setTimeout(180_000);
		await loginAdmin(context);
		const receiptId = await uploadFixture(page, context, await loadFixture('a'));

		await page.goto('/benchmarks');
		await expect(page.getByRole('checkbox', { name: 'e2e-vision' })).toBeChecked();
		await page.getByPlaceholder('Comma-separated receipt UUIDs').fill(receiptId);
		await page.getByRole('button', { name: 'Create benchmark' }).first().click();
		// The confirmation dialog is rendered inline (not role="dialog"),
		// so click the second "Create benchmark" button directly.
		await page.getByRole('button', { name: 'Create benchmark' }).nth(1).click();
		await expect(page).toHaveURL(/\/benchmarks\/results$/);

		// The run is picked up by the benchmark worker and finishes fast
		// (the stub answers instantly; ground truth = the verified receipt).
		await expect(page.getByText('Completed', { exact: true })).toBeVisible();
		await expect(page.getByText('1 / 1')).toBeVisible({ timeout: 60_000 });
		await expect(page.getByText('e2e-vision')).toBeVisible();
		await page.getByRole('button', { name: 'Load individual results' }).click();
		const individual = page.locator('summary').filter({ hasText: receiptId });
		await expect(individual).toContainText('Extraction 100.0%');
		await individual.click();
		await expect(page.getByText(/Strict:/)).toBeVisible();
		await page.getByRole('link', { name: 'Reuse this receipt selection' }).click();
		await expect(page).toHaveURL(/\/benchmarks$/);
		await expect(page.getByPlaceholder('Comma-separated receipt UUIDs')).toHaveValue(receiptId);
	});

	test('receipt used in a benchmark run cannot be deleted', async ({ page, context }) => {
		test.setTimeout(150_000);
		await loginAdmin(context);
		const receiptId = await uploadFixture(page, context, await loadFixture('b'));

		const run = await context.request.post('/api/v1/benchmarks', {
			data: { model_ids: ['e2e-vision'], receipt_ids: [receiptId] }
		});
		expect(run.status()).toBe(202);

		// The benchmark FK (benchmark_tasks.receipt_id -> receipts) is committed
		// together with the run (create_run inserts run + tasks + summaries in one
		// connection), so the delete is blocked deterministically. Verify at the API
		// level rather than depending on the receipt appearing in the list, which is
		// slow and order-dependent under full-suite load.
		expect((await context.request.delete(`/api/v1/receipts/${receiptId}`)).status()).toBe(409);
		// The receipt must still exist (the delete was blocked).
		expect((await context.request.get(`/api/v1/receipts/${receiptId}`)).status()).toBe(200);

		// The UI still shows the receipt (it was not deleted).
		await page.goto(`/receipts/${receiptId}`);
		await expect(page.locator('#re-merchant')).toBeVisible();
	});
});
