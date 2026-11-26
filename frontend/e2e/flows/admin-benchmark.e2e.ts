import { expect, test } from '@playwright/test';
import { getSettings, qualifyGold } from '../helpers/api';
import { loginAdmin } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';
import { uploadFixture } from '../helpers/ui';

test.describe('admin: benchmarks', () => {
	stubGuard();

	test('create a run and watch it complete in the results view', async ({ page, context }) => {
		test.setTimeout(180_000);
		await loginAdmin(context);
		const defaults = (await getSettings(context)).llm;
		const receiptId = await uploadFixture(page, context, await loadFixture('a'));

		await page.getByRole('link', { name: 'Use this receipt in a benchmark' }).click();
		await expect(page).toHaveURL(new RegExp('\\?receipt=' + receiptId));
		await page.getByRole('checkbox', { name: 'e2e-vision' }).check();
		await page.getByRole('checkbox', { name: 'e2e-alternate-vision' }).check();
		await page.getByLabel('Temperature').fill('0');
		await page.getByRole('button', { name: 'Refresh models' }).click();
		await expect(page.getByRole('checkbox', { name: 'e2e-alternate-vision' })).toBeChecked();
		await page.getByRole('button', { name: 'Preview human references' }).click();
		await page.locator('summary').filter({ hasText: receiptId }).click();
		await expect(page.locator('pre')).toContainText('Aldi Test');
		await page
			.getByRole('checkbox', {
				name: 'I checked these current receipts and originals and explicitly qualify them as benchmark references.'
			})
			.check();
		await page.getByRole('button', { name: 'Create benchmark' }).first().click();
		await page.getByRole('button', { name: 'Confirm and start', exact: true }).click();
		await expect(page).toHaveURL(/\/benchmarks\/results\?run=[0-9a-f-]+$/);

		// The run is picked up by the benchmark worker and finishes fast
		// (the stub answers instantly; ground truth = the verified receipt).
		await expect(page.getByText('Completed', { exact: true })).toBeVisible();
		await expect(page.getByText('2 / 2')).toBeVisible({ timeout: 60_000 });
		await expect(page.getByRole('cell', { name: /e2e-vision/ })).toBeVisible();
		await expect(page.getByRole('cell', { name: /e2e-alternate-vision/ })).toBeVisible();
		const runId = new URL(page.url()).searchParams.get('run');
		const run = await context.request.get('/api/v1/benchmarks/' + runId);
		expect(run.status()).toBe(200);
		expect((await run.json()).run.experiment.options.temperature).toBe(0);
		expect((await getSettings(context)).llm).toEqual(defaults);
		await page.getByRole('button', { name: 'Load individual results' }).click();
		const individual = page
			.locator('summary')
			.filter({ hasText: receiptId })
			.filter({ hasText: 'e2e-vision' });
		await expect(individual).toContainText('Extraction 100.0%');
		await individual.click();
		const details = page.locator('details').filter({ has: individual });
		await expect(details.getByText('Fully correct receipt', { exact: true })).toBeVisible();
		await page.getByRole('link', { name: 'Reuse this receipt selection' }).click();
		await expect(page).toHaveURL(/\/benchmarks$/);
		await expect(page.getByPlaceholder('Comma-separated receipt UUIDs')).toHaveValue(receiptId);
	});

	test('receipt used in a benchmark run cannot be deleted', async ({ page, context }) => {
		test.setTimeout(150_000);
		await loginAdmin(context);
		const receiptId = await uploadFixture(page, context, await loadFixture('b'));

		await qualifyGold(context, receiptId);
		const run = await context.request.post('/api/v1/benchmarks', {
			data: { model_ids: ['e2e-vision'], receipt_ids: [receiptId] }
		});
		expect(run.status()).toBe(202);
		const { id: runId } = await run.json();

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
		await expect
			.poll(
				async () => {
					const status = await context.request.get('/api/v1/benchmarks/' + runId);
					expect(status.status()).toBe(200);
					return (await status.json()).terminal;
				},
				{ timeout: 60_000 }
			)
			.toBe(1);
	});
});
