import { expect, test } from '@playwright/test';
import { loginAdmin } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { setStubMode, stubGuard } from '../helpers/stub';
import { uploadFixture } from '../helpers/ui';

test.describe('admin: benchmark results, comparison and failures', () => {
	stubGuard();

	test('rerun, compare only comparable runs, charts and receipt links', async ({
		page,
		context
	}) => {
		test.setTimeout(180_000);
		await loginAdmin(context);
		const receiptA = await uploadFixture(page, context, await loadFixture('a'));
		const receiptB = await uploadFixture(page, context, await loadFixture('b'));

		const create = async (receipt: string, label: string) => {
			const r = await context.request.post('/api/v1/benchmarks', {
				data: { model_ids: ['e2e-vision'], receipt_ids: [receipt], label }
			});
			expect(r.status()).toBe(202);
			return ((await r.json()) as { id: string }).id;
		};
		const terminal = async (id: string) =>
			(
				(await (await context.request.get(`/api/v1/benchmarks/${id}`)).json()) as {
					terminal: number;
				}
			).terminal;
		const baselineId = await create(receiptA, 'baseline');
		await expect.poll(() => terminal(baselineId), { timeout: 60_000 }).toBe(1);
		const otherId = await create(receiptB, 'other receipts');
		await expect.poll(() => terminal(otherId), { timeout: 60_000 }).toBe(1);

		// The newest run is selected; a run on other receipts is not selectable and says why.
		await page.goto('/benchmarks/results');
		await expect(page.getByTestId('run-picker')).toContainText('other receipts');
		const blocked = page.locator('label[data-blocked="true"]').filter({ hasText: 'baseline' });
		await expect(blocked.getByRole('checkbox')).toBeDisabled();
		await expect(blocked).toHaveAttribute('title', /Different receipts or expected values/);

		// Select the baseline run and rerun it: its rerun is comparable with it.
		await page.getByTestId('run-picker').getByRole('combobox').selectOption(baselineId);
		await page.getByPlaceholder('e.g. after model update, ngl 20').fill('second try');
		await page.getByRole('button', { name: 'Rerun with the same inputs' }).click();
		await expect(page.getByRole('status')).toContainText('Queued rerun');
		await expect(page.getByText('1 / 1')).toBeVisible({ timeout: 60_000 });

		const original = page.locator('label').filter({ hasText: 'baseline' });
		await expect(original.getByRole('checkbox')).toBeEnabled();
		await expect(
			page.locator('label').filter({ hasText: 'other receipts' }).getByRole('checkbox')
		).toBeDisabled();
		await original.getByRole('checkbox').check();

		await expect(page.getByTestId('comparability')).toContainText('Directly comparable');
		await expect(page).toHaveURL(/run=.*&with=/);
		await expect(page.getByTestId('summary-table')).toContainText('baseline');
		await expect(page.getByTestId('summary-table')).toContainText('second try');
		for (const id of [
			'chart-summary-bars',
			'chart-score-dots',
			'chart-dumbbell',
			'chart-latency',
			'receipt-table',
			'run-details'
		]) {
			await expect(page.getByTestId(id)).toBeVisible();
		}

		// In-depth details sit below the charts.
		await page.getByRole('button', { name: 'Load individual results' }).click();
		await expect(
			page.getByTestId('run-details').locator('summary').filter({ hasText: receiptA })
		).toContainText('Extraction 100.0%');

		// Receipt ids are links; hovering one shows a small thumbnail of the photo.
		const idLink = page
			.getByTestId('receipt-table')
			.getByRole('link', { name: receiptA.slice(0, 8) });
		await idLink.hover();
		await expect(page.getByTestId('receipt-preview')).toBeVisible();
		await page.mouse.move(0, 0);
		await expect(page.getByTestId('receipt-preview')).toHaveCount(0);
		await expect(
			page
				.getByTestId('chart-score-dots')
				.getByRole('link', { name: receiptA.slice(0, 8), exact: true })
		).toHaveAttribute('href', new RegExp(`/receipts/${receiptA}$`));

		// Hovering a dot pins its values; clicking it opens the receipt.
		const dot = page
			.getByTestId('chart-score-dots')
			.getByRole('link', { name: /^Receipt .*: / })
			.first();
		await dot.focus();
		await expect(page.getByTestId('chart-pinned')).toContainText(receiptA.slice(0, 8));
		await dot.click();
		await expect(page).toHaveURL(new RegExp(`/receipts/${receiptA}$`));
	});

	test('a model that cannot read the receipt is documented with its reason', async ({
		page,
		context
	}) => {
		test.setTimeout(120_000);
		await loginAdmin(context);
		const receipt = await uploadFixture(page, context, await loadFixture('a'));
		await setStubMode('unreadable');
		try {
			const run = await context.request.post('/api/v1/benchmarks', {
				data: { model_ids: ['e2e-vision'], receipt_ids: [receipt] }
			});
			expect(run.status()).toBe(202);
			await page.goto('/benchmarks/results');
			const failures = page.getByTestId('failure-table');
			await expect(failures).toBeVisible({ timeout: 60_000 });
			await expect(failures).toContainText('Unreadable');
			await expect(failures).toContainText(receipt.slice(0, 8));
			await expect(failures.getByRole('link', { name: receipt.slice(0, 8) })).toBeVisible();
			await expect(page.getByTestId('receipt-table')).toContainText('unreadable');
		} finally {
			await setStubMode('ok');
		}
	});
});
