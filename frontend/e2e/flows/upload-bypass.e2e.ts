import { expect, test } from '@playwright/test';
import { loginAdmin } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';
import { uploadFixture, waitForSnackbar } from '../helpers/ui';

test.describe('bypass review', () => {
	stubGuard();

	test('skip review saves directly as verified', async ({ page, context }) => {
		test.setTimeout(120_000);
		const fixture = await loadFixture('a');
		await uploadFixture(page, context, fixture, { bypass: true });
		await expect(page.getByText('Verified', { exact: true })).toBeVisible();
		await expect(page.getByRole('button', { name: 'Verify' })).toHaveCount(0);

		// Counted in the verified-only dashboard stats.
		await page.goto('/');
		await expect(page.getByText('€3.86')).toBeVisible();
	});
});

test.describe('human review of a verified receipt', () => {
	stubGuard();

	test('records human review separately from Gold qualification', async ({ page, context }) => {
		test.setTimeout(120_000);
		await loginAdmin(context);
		const id = await uploadFixture(page, context, await loadFixture('b'), { bypass: true });
		const receiptPath = `/api/v1/receipts/${id}`;
		const referencePath = `/api/v1/benchmarks/references/${id}`;
		const before = await (await context.request.get(receiptPath)).json();
		expect(before.receipt.verification_source).toBe('automatic_bypass');
		expect((await context.request.get(referencePath)).status()).toBe(422);

		await page.getByRole('button', { name: 'Review as human', exact: true }).click();
		const dialog = page.getByRole('dialog', { name: 'Review receipt as human?' });
		await expect(dialog).toBeVisible();
		const response = page.waitForResponse(
			(response) =>
				response.url().endsWith(receiptPath + '/verify') && response.request().method() === 'POST'
		);
		await dialog.getByRole('button', { name: 'Review as human', exact: true }).click();
		const result = await response;
		expect(result.status()).toBe(200);
		expect(result.request().postDataJSON()).toEqual({
			expected_operation_revision: before.receipt.operation_revision,
			expected_review_revision: before.receipt.review_revision,
			acknowledgements: []
		});
		await expect(page.getByRole('button', { name: 'Review as human', exact: true })).toHaveCount(0);
		const after = await (await context.request.get(receiptPath)).json();
		expect(after.receipt.verification_source).toBe('human');
		expect(after.receipt.verified).toBe(true);
		expect(after.receipt.review_revision).toBe(before.receipt.review_revision);
		expect(after.receipt.total).toBe(before.receipt.total);
		expect(after.line_items).toEqual(before.line_items);
		const preview = await context.request.get(referencePath);
		expect(preview.status()).toBe(200);
		const document = await preview.json();
		const qualify = await context.request.post(referencePath, {
			data: {
				expected_review_revision: document.review_revision,
				expected_document_sha256: document.document_sha256
			}
		});
		expect(qualify.status()).toBe(200);
		const gold = await qualify.json();
		expect(gold.qualified_by).toBe(before.receipt.user_id);

		await page.getByLabel('Category', { exact: true }).selectOption('restaurant');
		await page.getByRole('button', { name: 'Save', exact: true }).click();
		await waitForSnackbar(page, 'Changes saved');
		const updated = await (await context.request.get(referencePath)).json();
		expect(updated.review_revision).toBe(document.review_revision);
		expect(updated.document_sha256).not.toBe(document.document_sha256);
		expect(updated.expected.category).toBe('restaurant');
		const stale = await context.request.post(referencePath, {
			data: {
				expected_review_revision: document.review_revision,
				expected_document_sha256: document.document_sha256
			}
		});
		expect(stale.status()).toBe(409);
		const fresh = await context.request.post(referencePath, {
			data: {
				expected_review_revision: updated.review_revision,
				expected_document_sha256: updated.document_sha256
			}
		});
		expect(fresh.status()).toBe(200);
		expect((await fresh.json()).id).not.toBe(gold.id);
	});
});
