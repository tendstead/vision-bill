import { expect, test } from '@playwright/test';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard, setStubScenario } from '../helpers/stub';
import { ensureSession, confirmDialog } from '../helpers/ui';
import { awaitSubmission, uploadViaApi } from '../helpers/api';
import type { ReceiptWithDetails } from '../../src/lib/types';

// Uses server-selected OCR/direct extraction and deterministic CPU stub evidence.
test.describe('draft review and versioned receipt editing', () => {
	stubGuard();
	test('finalizing from a draft preserves absent optional observations', async ({
		page,
		context
	}) => {
		test.setTimeout(120_000);
		await ensureSession(context);
		await setStubScenario('partial');
		const fixture = await loadFixture('a');
		const uploaded = await uploadViaApi(context, fixture.name, fixture.bytes, fixture.mimeType);
		expect(uploaded.status).toBe(202);
		const id = uploaded.body.id as string;
		const job = await awaitSubmission(context, id);
		expect(job.status).toBe('needs_review');
		await page.goto(`/upload?job=${id}&draft=1`);
		await expect(page.getByLabel('Merchant', { exact: true })).toHaveValue(
			fixture.expected.merchant_name
		);
		const before = await context.request.get(`/api/v1/images/${id}/draft`);
		const draft = await before.json();
		await page.getByLabel('Merchant', { exact: true }).fill('Human correction');
		const patchPromise = page.waitForRequest(
			(r) => r.method() === 'PATCH' && r.url().endsWith(`/images/${id}/draft`)
		);
		await page.getByRole('button', { name: 'Save draft' }).click();
		const patch = await patchPromise;
		expect(patch.postDataJSON()).toEqual({
			expected_revision: draft.revision,
			changes: { merchant_name: 'Human correction' }
		});
		await expect(page.getByRole('button', { name: 'Save draft' })).toBeDisabled();
		const partial = (await (await context.request.get(`/api/v1/images/${id}/draft`)).json())
			.document;
		expect(partial.discount_total).toBeNull();
		expect(partial.tax_total).toBeNull();
		expect(partial.total).toBe(draft.document.total);
		await page.getByRole('button', { name: 'Finalize draft' }).click();
		await expect(page.getByRole('alert')).toContainText('Required missing fields');
		const taxTotal = page.getByLabel('Tax total', { exact: true });
		await page.getByRole('button', { name: 'Tax total', exact: true }).click();
		await expect(taxTotal).toBeFocused();
		await taxTotal.fill(fixture.expected.tax_total);
		const addTags = page.getByRole('button', { name: 'Add tags', exact: true });
		const missingTags = await addTags.count();
		for (let i = 0; i < missingTags; i++) await addTags.first().click();
		await expect(addTags).toHaveCount(0);
		await page.getByRole('button', { name: 'Save draft' }).click();
		await expect(page.getByRole('button', { name: 'Save draft' })).toBeDisabled();
		const finalize = page.waitForResponse(
			(r) => r.request().method() === 'POST' && r.url().endsWith('/draft/finalize')
		);
		await page.getByRole('button', { name: 'Finalize draft' }).click();
		expect((await finalize).status()).toBe(200);
		await expect(page).toHaveURL(/\/receipts\/[a-f0-9-]+/);
		const receiptId = page.url().split('/').pop()!;
		const result = await context.request.get(`/api/v1/receipts/${receiptId}`);
		const detail = (await result.json()) as ReceiptWithDetails;
		expect(detail.receipt).toMatchObject({
			verified: false,
			status: 'unverified',
			merchant_name: 'Human correction',
			discount_total: null,
			tax_total: fixture.expected.tax_total,
			tip: null
		});
		expect(detail.line_items[0]).toMatchObject({
			description: fixture.expected.line_items[0].description,
			quantity: null,
			unit_price: null,
			total_price: fixture.expected.line_items[0].total_price
		});
		const finalized = await context.request.get(`/api/v1/images/${id}/draft`);
		expect((await finalized.json()).finalized_receipt_id).toBe(receiptId);
	});
	test('two stale concurrent PUTs cannot overwrite, and a financial mismatch revokes verification', async ({
		page,
		context
	}) => {
		test.setTimeout(120_000);
		await ensureSession(context);
		await setStubScenario('partial');
		const fixture = await loadFixture('a');
		const uploaded = await uploadViaApi(context, fixture.name, fixture.bytes, fixture.mimeType);
		const id = uploaded.body.id as string;
		const job = await awaitSubmission(context, id);
		expect(job.status).toBe('needs_review');
		const draft = await (await context.request.get(`/api/v1/images/${id}/draft`)).json();
		const document = {
			...draft.document,
			tax_total: fixture.expected.tax_total,
			line_items: draft.document.line_items.map((line: Record<string, unknown>) => ({
				...line,
				tags: line.tags ?? []
			}))
		};
		const patched = await context.request.patch(`/api/v1/images/${id}/draft`, {
			data: {
				expected_revision: draft.revision,
				changes: {
					...document,
					amount_policy: {
						tax_basis: 'net',
						discount_treatment: 'included_in_lines',
						tip_treatment: 'none',
						subtotal_origin: 'printed',
						tax_information: 'present'
					}
				}
			}
		});
		expect(patched.status()).toBe(200);
		const final = await context.request.post(`/api/v1/images/${id}/draft/finalize`, {
			data: { expected_revision: (await patched.json()).revision, request_key: crypto.randomUUID() }
		});
		expect(final.status()).toBe(200);
		const receiptId = (await final.json()).id;
		await page.goto(`/receipts/${receiptId}`);
		const detail = (await (
			await context.request.get(`/api/v1/receipts/${receiptId}`)
		).json()) as ReceiptWithDetails;
		expect(detail.receipt.findings.filter((f) => f.severity !== 'info')).toEqual([]);
		await page.getByRole('button', { name: 'Verify', exact: true }).click();
		await confirmDialog(page, 'Verify receipt?', 'Verify');
		await expect(page.getByText('Verified', { exact: true })).toBeVisible();
		const verified = (await (
			await context.request.get(`/api/v1/receipts/${receiptId}`)
		).json()) as ReceiptWithDetails;
		const write = {
			...verified.receipt,
			time: verified.receipt.time?.slice(0, 5) ?? null,
			line_items: verified.line_items,
			taxes: verified.taxes,
			collection_ids: verified.collection_ids,
			expected_operation_revision: verified.receipt.operation_revision,
			merchant_name: 'Concurrent edit',
			total: '999.00'
		};
		const results = await Promise.all([
			context.request.put(`/api/v1/receipts/${receiptId}`, { data: write }),
			context.request.put(`/api/v1/receipts/${receiptId}`, { data: write })
		]);
		expect(results.map((r) => r.status()).sort()).toEqual([200, 409]);
		const current = (await (
			await context.request.get(`/api/v1/receipts/${receiptId}`)
		).json()) as ReceiptWithDetails;
		expect(current.receipt.total).toBe('999.00');
		expect(
			current.receipt.findings.filter((f) => f.severity === 'review').map((f) => f.code)
		).toEqual(['amount_discrepancy']);
		expect(current.receipt.verified).toBe(false);
		expect(current.line_items.map((line) => line.id)).toEqual(
			verified.line_items.map((line) => line.id)
		);
		expect(current.line_items.map((line) => line.position)).toEqual(
			verified.line_items.map((_, index) => index)
		);
		await page.reload();
		await expect(page.getByRole('button', { name: 'Verify', exact: true })).toBeVisible();
	});
});
