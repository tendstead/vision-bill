import { expect, test } from '@playwright/test';
import { loginAdmin } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';
import { uploadFixture } from '../helpers/ui';

test.describe('admin: compare benchmark runs', () => {
	stubGuard();

	test('rerun, compare and follow a receipt link', async ({ page, context }) => {
		test.setTimeout(180_000);
		await loginAdmin(context);
		const receiptId = await uploadFixture(page, context, await loadFixture('a'));

		const first = await context.request.post('/api/v1/benchmarks', {
			data: { model_ids: ['e2e-vision'], receipt_ids: [receiptId], label: 'baseline' }
		});
		expect(first.status()).toBe(202);
		const firstId = ((await first.json()) as { id: string }).id;
		const terminal = async (id: string) =>
			(
				(await (await context.request.get(`/api/v1/benchmarks/${id}`)).json()) as {
					terminal: number;
				}
			).terminal;
		await expect.poll(() => terminal(firstId), { timeout: 60_000 }).toBe(1);

		// Rerun from the results page.
		await page.goto('/benchmarks/results');
		await page.getByPlaceholder('e.g. after model update, ngl 20').fill('second try');
		await page.getByRole('button', { name: 'Rerun with the same inputs' }).click();
		await expect(page.getByRole('status')).toContainText('Queued rerun');
		const runs = (await (await context.request.get('/api/v1/benchmarks')).json()) as {
			id: string;
			label: string | null;
			parent_run_id: string | null;
		}[];
		const second = runs.find((r) => r.label === 'second try');
		expect(second?.parent_run_id).toBe(firstId);
		await expect.poll(() => terminal(second!.id), { timeout: 60_000 }).toBe(1);

		// The default selection is the newest run and its parent.
		await page.goto('/benchmarks/compare');
		await expect(page.getByTestId('comparability')).toContainText('Directly comparable');
		await expect(page).toHaveURL(new RegExp(`runs=${firstId},${second!.id}`));
		await expect(page.getByTestId('summary-table')).toContainText('baseline');
		await expect(page.getByTestId('summary-table')).toContainText('second try');
		// The table links every receipt too.
		await expect(
			page.getByTestId('receipt-table').getByRole('link', { name: receiptId.slice(0, 8) })
		).toBeVisible();

		await page
			.getByTestId('receipt-table')
			.getByRole('link', { name: receiptId.slice(0, 8) })
			.click();
		await expect(page).toHaveURL(new RegExp(`/receipts/${receiptId}$`));
	});

	test('a single run cannot be compared', async ({ page, context }) => {
		await loginAdmin(context);
		await page.goto('/benchmarks/compare?runs=99999999-9999-9999-9999-999999999999');
		await expect(
			page.getByText('Select at least two runs').or(page.getByText('At least two'))
		).toBeVisible();
	});
});
