import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/svelte';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { api, ApiError } from '$lib/api/client';
import { queryClient } from '$lib/query/client';
import { queryKeys } from '$lib/query/keys';
import type { ReceiptRow, ReceiptWithDetails, ReviewPreview } from '$lib/types';
import Page from './+page.svelte';
import { page } from '$app/state';
import { tick } from 'svelte';
import { advanceSessionGeneration, session } from '$lib/session-lifetime';
vi.mock('$app/state', async () => {
	const { SvelteURL } = await import('svelte/reactivity');
	return {
		page: { url: new SvelteURL('http://localhost/receipts/00000000-0000-4000-8000-000000000001') }
	};
});
vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
const row: ReceiptRow = {
	id: '00000000-0000-4000-8000-000000000001',
	confidence: 90,
	merchant_name: 'Shop',
	merchant_address: null,
	receipt_number: null,
	date: '2026-01-01',
	time: null,
	currency: 'EUR',
	category: 'grocery',
	subtotal: '2',
	discount_total: '0',
	tax_total: '0',
	tip: null,
	total: '2',
	payment_method: 'cash',
	created_at: null,
	status: 'unverified',
	image_id: null,
	thumbnail_path: null,
	verified: false,
	user_id: null,
	operation_revision: 7,
	review_revision: 3,
	amount_policy: {
		tax_basis: 'gross',
		discount_treatment: 'included_in_lines',
		tip_treatment: 'none',
		subtotal_origin: 'printed',
		tax_information: 'explicit_zero'
	},
	findings: [],
	field_sources: {},
	overlap_decisions: [],
	acknowledgements: [],
	checks_version: '1',
	verification_source: null
};
const owner = { id: 'owner', username: 'alice', is_admin: false, can_see_all: false };
const verifiedByBypass: ReceiptRow = {
	...row,
	status: 'verified',
	verified: true,
	verification_source: 'automatic_bypass'
};
const previewResponse: ReviewPreview = { operation_revision: 7, review_revision: 3, findings: [] };
const detail = (receipt = row, canRereview = false): ReceiptWithDetails => ({
	receipt,
	can_rereview: canRereview,
	line_items: [],
	taxes: [],
	collection_ids: [],
	image_path: null,
	classification_enrichment: { state: 'not_recorded', reason_code: null, missing_fields: [] }
});
beforeEach(() => {
	session.set(owner);
	queryClient.clear();
	page.url.pathname = '/receipts/00000000-0000-4000-8000-000000000001';
	vi.spyOn(api, 'getReceipt').mockResolvedValue(detail());
	vi.spyOn(api, 'listTags').mockResolvedValue([]);
	vi.spyOn(api, 'listCollections').mockResolvedValue([]);
	vi.spyOn(api, 'listSpendingCategories').mockResolvedValue([]);
});
afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
	session.set(null);
	queryClient.clear();
	vi.restoreAllMocks();
});
it('keeps dirty form and expected operation revision on a PUT conflict', async () => {
	const update = vi.spyOn(api, 'updateReceipt').mockRejectedValue(new ApiError(409, 'stale'));
	render(Page);
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'My edit' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	await waitFor(() => expect(update).toHaveBeenCalledOnce());
	expect((screen.getByLabelText('Merchant') as HTMLInputElement).value).toBe('My edit');
	await waitFor(() =>
		expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(false)
	);
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(update.mock.calls[1][1].expected_operation_revision).toBe(7);
});
it('does not verify new material findings using pre-save comments', async () => {
	const findings = [
		{
			id: 'fresh',
			code: 'total_mismatch',
			severity: 'review' as const,
			field_path: 'total',
			source_refs: [],
			comparison: {}
		}
	];
	const saved = {
		...row,
		merchant_name: 'Changed',
		operation_revision: 8,
		review_revision: 4,
		findings
	};
	vi.spyOn(api, 'updateReceipt').mockResolvedValue(saved);
	vi.spyOn(api, 'getReceipt').mockResolvedValueOnce(detail()).mockResolvedValue(detail(saved));
	vi.spyOn(api, 'previewHumanReview').mockResolvedValue(saved);
	const verify = vi.spyOn(api, 'verifyReceipt');
	render(Page);
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Changed' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save & verify' }));
	await screen.findByLabelText(
		'Review comment: Additional diagnostic: total mismatch. Check the original or processing status.'
	);
	expect(verify).not.toHaveBeenCalled();
	expect(screen.queryByRole('button', { name: 'Save' })).toBeNull();
});
it('uses successful PUT versions for verification and keeps save successful if verify fails', async () => {
	const saved = { ...row, merchant_name: 'Saved', operation_revision: 8, review_revision: 4 };
	vi.spyOn(api, 'updateReceipt').mockResolvedValue(saved);
	vi.spyOn(api, 'getReceipt').mockResolvedValueOnce(detail()).mockResolvedValue(detail(saved));
	vi.spyOn(api, 'previewHumanReview').mockResolvedValue(saved);
	const verify = vi
		.spyOn(api, 'verifyReceipt')
		.mockRejectedValue(new ApiError(422, 'missing original'));
	render(Page);
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Saved' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save & verify' }));
	await waitFor(() =>
		expect(verify).toHaveBeenCalledWith(row.id, {
			expected_operation_revision: 8,
			expected_review_revision: 4,
			acknowledgements: []
		})
	);
	await waitFor(() => expect(screen.queryByRole('button', { name: 'Save' })).toBeNull());
	expect((screen.getByLabelText('Merchant') as HTMLInputElement).value).toBe('Saved');
});

it('has no acknowledgement control for missing originals and disables verification', async () => {
	const missing = detail({
		...row,
		findings: [
			{
				id: 'missing',
				code: 'missing_original',
				severity: 'integrity',
				field_path: 'originals',
				source_refs: [],
				comparison: {}
			}
		]
	});
	vi.mocked(api.getReceipt).mockResolvedValue(missing);
	vi.spyOn(api, 'previewHumanReview').mockResolvedValue({
		...previewResponse,
		findings: missing.receipt.findings
	});
	render(Page);
	await screen.findByText(/Verification blocked: The original photo is missing/);
	expect(screen.queryByRole('textbox', { name: /Review comment/ })).toBeNull();
	await fireEvent.click(screen.getByRole('button', { name: 'Verify' }));
	const dialog = within(await screen.findByRole('dialog'));
	expect(dialog.queryByRole('textbox')).toBeNull();
	expect((dialog.getByRole('button', { name: 'Verify' }) as HTMLButtonElement).disabled).toBe(true);
});
it('displays server revocation after editing a verified receipt', async () => {
	vi.mocked(api.getReceipt)
		.mockResolvedValueOnce(detail({ ...row, status: 'verified', verified: true }, true))
		.mockResolvedValue(
			detail({ ...row, merchant_name: 'Edited', operation_revision: 8, review_revision: 4 })
		);
	vi.spyOn(api, 'updateReceipt').mockResolvedValue({
		...row,
		merchant_name: 'Edited',
		operation_revision: 8,
		review_revision: 4
	});
	render(Page);
	await screen.findByText('Verified');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Edited' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	await screen.findByRole('button', { name: 'Verify' });
	expect(screen.queryByText('Verified')).toBeNull();
});

function deferred<T>() {
	let resolve!: (value: T) => void;
	let reject!: (error: unknown) => void;
	const promise = new Promise<T>((ok, no) => {
		resolve = ok;
		reject = no;
	});
	return { promise, resolve, reject };
}
const second = { ...row, id: '00000000-0000-4000-8000-000000000002', merchant_name: 'Second' };
async function navigateSecond() {
	page.url.pathname = `/receipts/${second.id}`;
	await tick();
	await screen.findByDisplayValue('Second');
}
it('resets busy state and ignores a late PUT across receipt navigation', async () => {
	const pending = deferred<ReceiptRow>();
	vi.spyOn(api, 'updateReceipt').mockReturnValue(pending.promise);
	vi.mocked(api.getReceipt).mockImplementation(async (id) =>
		detail(id === second.id ? second : row)
	);
	render(Page);
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'A edit' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	await navigateSecond();
	expect((screen.getByRole('button', { name: 'Verify' }) as HTMLButtonElement).disabled).toBe(
		false
	);
	pending.reject(new ApiError(409, 'A conflict'));
	await tick();
	expect(screen.queryByRole('alert')).toBeNull();
});
it('resets a pending verify and its dialog across receipt navigation', async () => {
	const pending = deferred<ReceiptRow>();
	vi.spyOn(api, 'previewHumanReview').mockResolvedValue(previewResponse);
	vi.spyOn(api, 'verifyReceipt').mockReturnValue(pending.promise);
	vi.mocked(api.getReceipt).mockImplementation(async (id) =>
		detail(id === second.id ? second : row)
	);
	render(Page);
	await screen.findByLabelText('Merchant');
	await fireEvent.click(screen.getByRole('button', { name: 'Verify' }));
	await fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Verify' }));
	await waitFor(() => expect(api.verifyReceipt).toHaveBeenCalledOnce());
	await navigateSecond();
	expect(screen.queryByRole('dialog')).toBeNull();
	expect((screen.getByRole('button', { name: 'Verify' }) as HTMLButtonElement).disabled).toBe(
		false
	);
	pending.reject(new ApiError(409, 'A conflict'));
	await tick();
	expect(screen.queryByRole('alert')).toBeNull();
});
it('fences discard-reload errors to their originating receipt view', async () => {
	vi.spyOn(api, 'updateReceipt').mockRejectedValue(new ApiError(409, 'stale'));
	const pending = deferred<ReceiptWithDetails>();
	let reads = 0;
	vi.mocked(api.getReceipt).mockImplementation((id) =>
		id === second.id
			? Promise.resolve(detail(second))
			: ++reads === 1
				? Promise.resolve(detail())
				: pending.promise
	);
	render(Page);
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'A edit' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	await screen.findByRole('alert');
	await fireEvent.click(
		screen.getByRole('button', { name: 'Discard edits and load current version' })
	);
	await navigateSecond();
	pending.reject(new ApiError(0, 'A reload failed'));
	await tick();
	expect(screen.queryByRole('alert')).toBeNull();
	expect((screen.getByRole('button', { name: 'Verify' }) as HTMLButtonElement).disabled).toBe(
		false
	);
});
const item = {
	id: 'line',
	receipt_id: row.id,
	description: 'Item',
	original_description: null,
	quantity: 1,
	unit_price: '2',
	total_price: '2',
	spending_category: 'unknown',
	category_source: 'unknown' as const,
	tags: [] as string[]
};
it('waits for deferred tag creation before saving, then reloads the submitted tag', async () => {
	vi.mocked(api.listTags).mockResolvedValue(['food']);
	const pending = deferred<Awaited<ReturnType<typeof api.createTag>>>();
	vi.spyOn(api, 'createTag').mockReturnValue(pending.promise);
	const update = vi
		.spyOn(api, 'updateReceipt')
		.mockResolvedValue({ ...row, merchant_name: 'Changed', operation_revision: 8 });
	vi.mocked(api.getReceipt)
		.mockResolvedValueOnce({ ...detail(), line_items: [item] })
		.mockResolvedValue({
			...detail({ ...row, merchant_name: 'Changed', operation_revision: 8 }),
			line_items: [{ ...item, tags: ['new tag'] }]
		});
	render(Page);
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Changed' } });
	await fireEvent.click(screen.getByText('Optional tags'));
	await fireEvent.click(screen.getByRole('button', { name: 'Tags' }));
	await fireEvent.input(screen.getByPlaceholderText('Search tags…'), {
		target: { value: 'new tag' }
	});
	await fireEvent.click(screen.getByRole('option', { name: 'Create “new tag”' }));
	expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(true);
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(update).not.toHaveBeenCalled();
	pending.resolve({ name: 'new tag', created: true });
	await waitFor(() =>
		expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(false)
	);
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	await waitFor(() => expect(update).toHaveBeenCalledOnce());
	expect(update.mock.calls[0][1].line_items[0].tags).toEqual(['new tag']);
	await waitFor(() => expect(screen.queryByRole('button', { name: 'Save' })).toBeNull());
	await fireEvent.click(screen.getByText('Optional tags'));
	expect(screen.getByText('new tag')).toBeTruthy();
});
it('waits for deferred collection creation before saving its membership', async () => {
	const pending = deferred<import('$lib/types').Collection>();
	vi.spyOn(api, 'createCollection').mockReturnValue(pending.promise);
	const update = vi
		.spyOn(api, 'updateReceipt')
		.mockResolvedValue({ ...row, merchant_name: 'Changed', operation_revision: 8 });
	vi.mocked(api.getReceipt)
		.mockResolvedValueOnce(detail())
		.mockResolvedValue({
			...detail({ ...row, merchant_name: 'Changed', operation_revision: 8 }),
			collection_ids: ['created']
		});
	render(Page);
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Changed' } });
	const picker = screen.getByPlaceholderText('Search collections…');
	await fireEvent.click(picker);
	await fireEvent.input(picker, { target: { value: 'New' } });
	await fireEvent.keyDown(picker, { key: 'Enter' });
	expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(true);
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(update).not.toHaveBeenCalled();
	pending.resolve({
		id: 'created',
		name: 'New',
		color: null,
		start_date: null,
		end_date: null,
		active: true,
		created_at: null
	});
	await waitFor(() =>
		expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(false)
	);
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	await waitFor(() => expect(update).toHaveBeenCalledOnce());
	expect(update.mock.calls[0][1].collection_ids).toEqual(['created']);
});
it('does not attach a late created collection to the next receipt', async () => {
	const pending = deferred<import('$lib/types').Collection>();
	vi.spyOn(api, 'createCollection').mockReturnValue(pending.promise);
	const update = vi
		.spyOn(api, 'updateReceipt')
		.mockResolvedValue({ ...second, merchant_name: 'B edit', operation_revision: 8 });
	vi.mocked(api.getReceipt).mockImplementation(async (id) =>
		detail(id === second.id ? second : row)
	);
	render(Page);
	await screen.findByLabelText('Merchant');
	const picker = screen.getByPlaceholderText('Search collections…');
	await fireEvent.click(picker);
	await fireEvent.input(picker, { target: { value: 'For A' } });
	await fireEvent.keyDown(picker, { key: 'Enter' });
	await navigateSecond();
	pending.resolve({
		id: 'for-a',
		name: 'For A',
		color: null,
		start_date: null,
		end_date: null,
		active: true,
		created_at: null
	});
	await tick();
	expect((screen.getByRole('button', { name: 'Verify' }) as HTMLButtonElement).disabled).toBe(
		false
	);
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'B edit' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	await waitFor(() => expect(update).toHaveBeenCalledOnce());
	expect(update.mock.calls[0]).toEqual([
		second.id,
		expect.objectContaining({ collection_ids: [], expected_operation_revision: 7 })
	]);
});

it.each([
	['unavailable', 'recovered_without_classification'],
	['skipped', 'classification_budget_exhausted']
] as const)(
	'shows missing optional labels for %s / %s without blocking review',
	async (state, reason_code) => {
		vi.mocked(api.getReceipt).mockResolvedValue({
			...detail(),
			classification_enrichment: {
				state,
				reason_code,
				missing_fields: ['category']
			}
		});
		render(Page);
		await screen.findByText(/some optional labels are unavailable/i);
		await waitFor(() =>
			expect((screen.getByRole('button', { name: 'Verify' }) as HTMLButtonElement).disabled).toBe(
				false
			)
		);
	}
);

it.each(['legacy', 'automatic_bypass', 'human'])(
	'reviews a %s receipt checked under version 2 when the server allows re-review',
	async (verificationSource) => {
		const verified: ReceiptRow = {
			...row,
			status: 'verified',
			verified: true,
			checks_version: '2',
			verification_source: verificationSource
		};
		const human = {
			...verified,
			operation_revision: 8,
			review_revision: 4,
			checks_version: '3',
			verification_source: 'human'
		};
		const findings = [
			{
				id: 'current-finding',
				code: 'total_mismatch',
				severity: 'review' as const,
				field_path: 'total',
				source_refs: [],
				comparison: {}
			}
		];
		vi.mocked(api.getReceipt)
			.mockResolvedValueOnce(detail(verified, true))
			.mockResolvedValue(detail(human));
		const preview = vi
			.fn()
			.mockResolvedValue(Response.json({ operation_revision: 7, review_revision: 3, findings }));
		vi.stubGlobal('fetch', preview);
		const verify = vi.spyOn(api, 'verifyReceipt').mockResolvedValue(human);
		const save = vi.spyOn(api, 'updateReceipt');
		try {
			render(Page);
			await screen.findByLabelText('Merchant');
			expect(screen.queryByRole('button', { name: 'Save & verify' })).toBeNull();
			await fireEvent.click(screen.getByRole('button', { name: 'Review as human' }));
			const dialog = await screen.findByRole('dialog', { name: 'Review receipt as human?' });
			const confirm = within(dialog).getByRole('button', {
				name: 'Review as human'
			}) as HTMLButtonElement;
			expect(confirm.disabled).toBe(true);
			await fireEvent.input(await within(dialog).findByRole('textbox'), {
				target: { value: 'Checked the original' }
			});
			await fireEvent.click(confirm);
			await waitFor(() =>
				expect(verify).toHaveBeenCalledWith(row.id, {
					expected_operation_revision: 7,
					expected_review_revision: 3,
					acknowledgements: [{ finding_id: 'current-finding', comment: 'Checked the original' }]
				})
			);
			await waitFor(() =>
				expect(screen.queryByRole('button', { name: 'Review as human' })).toBeNull()
			);
			expect(preview).toHaveBeenCalledOnce();
			const [url, init] = preview.mock.calls[0];
			expect(url).toBe(`/api/v1/receipts/${row.id}/verify`);
			expect(new Headers(init.headers).get('X-Vision-Bill-Expected-User')).toBe(owner.id);
			expect(save).not.toHaveBeenCalled();
		} finally {
			vi.unstubAllGlobals();
		}
	}
);

it('abandons the verify continuation if the session changes while reloading the save', async () => {
	const saved = { ...row, merchant_name: 'Saved', operation_revision: 8, review_revision: 4 };
	const pending = deferred<ReceiptWithDetails>();
	vi.spyOn(api, 'updateReceipt').mockResolvedValue(saved);
	vi.mocked(api.getReceipt).mockResolvedValueOnce(detail()).mockReturnValue(pending.promise);
	const verify = vi.spyOn(api, 'verifyReceipt').mockResolvedValue(saved);
	render(Page);
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Saved' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save & verify' }));
	await waitFor(() => expect(api.getReceipt).toHaveBeenCalledTimes(2));
	advanceSessionGeneration();
	queryClient.clear();
	pending.resolve(detail(saved));
	await pending.promise;
	await tick();
	await new Promise((resolve) => setTimeout(resolve, 0));
	expect(verify).not.toHaveBeenCalled();
	expect(queryClient.getQueryData(['receipts', row.id])).toBeUndefined();
});

it('stops an accepted human-review preview after the originating account changes', async () => {
	vi.mocked(api.getReceipt).mockResolvedValue(detail(verifiedByBypass, true));
	const body = deferred<ReviewPreview>();
	const json = vi.fn(() => body.promise);
	vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json }));
	render(Page);
	await fireEvent.click(await screen.findByRole('button', { name: 'Review as human' }));
	await waitFor(() => expect(json).toHaveBeenCalledOnce());

	body.resolve(previewResponse);
	// The client validates A when it resumes parsing. Switch before its fulfilled
	// promise resumes the still-mounted receipt view.
	queueMicrotask(() => {
		advanceSessionGeneration();
		session.set({ ...owner, id: 'successor', username: 'bob' });
	});
	await new Promise((resolve) => setTimeout(resolve, 0));
	expect(screen.queryByRole('dialog', { name: 'Review receipt as human?' })).toBeNull();
});

it('drops a human-review preview from a different receipt view', async () => {
	vi.mocked(api.getReceipt).mockImplementation(async (id) =>
		id === second.id ? detail(second) : detail(verifiedByBypass, true)
	);
	const pending = deferred<Response>();
	const fetchMock = vi.fn().mockReturnValue(pending.promise);
	vi.stubGlobal('fetch', fetchMock);
	render(Page);
	await fireEvent.click(await screen.findByRole('button', { name: 'Review as human' }));
	await waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());

	await navigateSecond();
	pending.resolve(Response.json(previewResponse));
	await new Promise((resolve) => setTimeout(resolve, 0));
	expect(screen.queryByRole('dialog', { name: 'Review receipt as human?' })).toBeNull();
	expect(screen.queryByRole('alert')).toBeNull();
});

it.each([row, verifiedByBypass])(
	'binds confirmation to the rendered $status receipt and closes it on revision change',
	async (cached) => {
		const next = {
			...cached,
			operation_revision: 8,
			review_revision: 4,
			merchant_name: 'A revised receipt',
			currency: 'CHF',
			total: '10'
		};
		const refreshed = deferred<ReceiptWithDetails>();
		queryClient.setQueryData(queryKeys.receipt(row.id), detail(cached, cached.verified), {
			updatedAt: Date.now() - 60_000
		});
		vi.mocked(api.getReceipt).mockReturnValue(refreshed.promise);
		vi.spyOn(api, 'previewHumanReview')
			.mockResolvedValueOnce(previewResponse)
			.mockResolvedValue({ operation_revision: 8, review_revision: 4, findings: [] });
		const verify = vi.spyOn(api, 'verifyReceipt').mockResolvedValue(next);
		const label = cached.verified ? /review as human/i : 'Verify';
		render(Page);
		expect(await screen.findByDisplayValue('Shop')).not.toBeNull();
		await fireEvent.click(screen.getByRole('button', { name: label }));
		expect(await screen.findByRole('dialog')).not.toBeNull();
		refreshed.resolve(detail(next, next.verified));
		await screen.findByDisplayValue('A revised receipt');
		expect(screen.queryByRole('dialog')).toBeNull();
		expect(verify).not.toHaveBeenCalled();
		await fireEvent.click(screen.getByRole('button', { name: label }));
		await fireEvent.click(
			within(await screen.findByRole('dialog')).getByRole('button', { name: label })
		);
		await waitFor(() =>
			expect(verify).toHaveBeenCalledWith(row.id, {
				expected_operation_revision: 8,
				expected_review_revision: 4,
				acknowledgements: []
			})
		);
	}
);

it('preserves dirty values but blocks saving or verifying a stale editor until explicitly reloaded', async () => {
	const next = { ...row, operation_revision: 8, review_revision: 4, merchant_name: 'New source' };
	const update = vi.spyOn(api, 'updateReceipt');
	render(Page);
	const merchant = await screen.findByLabelText('Merchant');
	await fireEvent.input(merchant, { target: { value: 'Unsaved edit' } });
	queryClient.setQueryData(queryKeys.receipt(row.id), detail(next));
	const discard = await screen.findByRole('button', {
		name: 'Discard edits and load current version'
	});
	expect((merchant as HTMLInputElement).value).toBe('Unsaved edit');
	expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(true);
	expect(
		(screen.getByRole('button', { name: 'Save & verify' }) as HTMLButtonElement).disabled
	).toBe(true);
	expect((screen.getByRole('button', { name: 'Verify' }) as HTMLButtonElement).disabled).toBe(true);
	expect(update).not.toHaveBeenCalled();
	vi.mocked(api.getReceipt).mockResolvedValue(detail(next));
	await fireEvent.click(discard);
	await screen.findByDisplayValue('New source');
	expect((screen.getByRole('button', { name: 'Verify' }) as HTMLButtonElement).disabled).toBe(
		false
	);
});

it.each([row, verifiedByBypass])(
	'never publishes new $status revisions with cached children after a failed refresh and remount',
	async (previous) => {
		const original = { ...detail(previous, previous.verified), line_items: [item] };
		const saved = { ...previous, operation_revision: 8, review_revision: 4 };
		const latest = {
			...detail(saved, saved.verified),
			line_items: [{ ...item, description: 'Corrected item' }]
		};
		const next = deferred<ReceiptWithDetails>();
		vi.mocked(api.getReceipt)
			.mockResolvedValueOnce(original)
			.mockRejectedValueOnce(new ApiError(503, 'Detail refresh unavailable'))
			.mockReturnValue(next.promise);
		vi.spyOn(api, 'updateReceipt').mockResolvedValue(saved);
		const preview = vi.spyOn(api, 'previewHumanReview').mockResolvedValue({
			operation_revision: 8,
			review_revision: 4,
			findings: []
		});
		const verify = vi.spyOn(api, 'verifyReceipt').mockResolvedValue(saved);
		const first = render(Page);
		const description = await screen.findByDisplayValue(item.description);
		await fireEvent.input(description, { target: { value: 'Corrected item' } });
		await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
		await screen.findByRole('alert');
		expect(queryClient.getQueryData(queryKeys.receipt(row.id))).toEqual(original);

		first.unmount();
		render(Page);
		await screen.findByDisplayValue(item.description);
		await waitFor(() => expect(api.getReceipt).toHaveBeenCalledTimes(3));
		expect(verify).not.toHaveBeenCalled();
		expect(preview).not.toHaveBeenCalled();
		next.resolve(latest);
		await screen.findByDisplayValue('Corrected item');
		const label = previous.verified ? 'Review as human' : 'Verify';
		await fireEvent.click(screen.getByRole('button', { name: label }));
		await fireEvent.click(
			within(await screen.findByRole('dialog')).getByRole('button', { name: label })
		);
		await waitFor(() =>
			expect(verify).toHaveBeenCalledWith(row.id, {
				expected_operation_revision: 8,
				expected_review_revision: 4,
				acknowledgements: []
			})
		);
	}
);

it('uses a full-width data workspace without a receipt tab for image-less records', async () => {
	render(Page);
	await screen.findByLabelText('Merchant');
	expect(screen.queryByRole('button', { name: 'Receipt' })).toBeNull();
	expect(document.querySelector('.workspace.without-photo')).toBeTruthy();
	expect(document.querySelector('.workspace > aside')).toBeNull();
});

it('opens unknown evidence requests from receipt source decisions', async () => {
	vi.mocked(api.getReceipt).mockResolvedValue(
		detail({ ...row, image_id: 'image', field_sources: { total: [] } })
	);
	vi.spyOn(api, 'getImage').mockResolvedValue({
		additional_images: [],
		source_views: [
			{
				view_id: 'first',
				photo_index: 0,
				coordinate_space: 'working',
				width: 100,
				height: 100,
				url: '/first.png'
			}
		]
	} as never);
	render(Page);
	await screen.findByLabelText('Merchant');
	const sourceDecisions = screen.getByText('Source decisions').closest('details')!;
	sourceDecisions.open = true;
	await fireEvent.click(within(sourceDecisions).getByRole('button', { name: 'Show source' }));
	expect(screen.getByRole('button', { name: 'Receipt' }).getAttribute('aria-pressed')).toBe('true');
	expect(screen.queryByRole('img', { name: 'Source view' })).toBeNull();
	expect(screen.getByText('Source mapping unknown')).toBeTruthy();
});

it.each([
	{ action: 'Verify', version: null },
	{ action: 'Save & verify', version: null },
	{ action: 'Verify', version: '2' },
	{ action: 'Save & verify', version: '2' }
])(
	'previews current findings for checks version $version on $action',
	async ({ action, version }) => {
		const migrated = {
			...row,
			checks_version: version,
			findings: version
				? [
						{
							id: 'obsolete',
							code: 'unknown_subtotal_origin',
							severity: 'review' as const,
							field_path: 'amount_policy.subtotal_origin',
							source_refs: [],
							comparison: {}
						}
					]
				: []
		};
		const displayed = { ...migrated, operation_revision: action === 'Verify' ? 7 : 8 };
		vi.mocked(api.getReceipt)
			.mockResolvedValueOnce(detail(migrated))
			.mockResolvedValue(detail(displayed));
		vi.spyOn(api, 'updateReceipt').mockResolvedValue(displayed);
		vi.spyOn(api, 'previewHumanReview').mockResolvedValue({
			...previewResponse,
			operation_revision: displayed.operation_revision,
			findings: [
				{
					id: 'live',
					code: 'unknown_tax_basis',
					severity: 'review',
					field_path: 'amount_policy.tax_basis',
					source_refs: [],
					comparison: {}
				}
			]
		});
		const verify = vi.spyOn(api, 'verifyReceipt').mockResolvedValue({ ...row, verified: true });
		render(Page);
		await screen.findByLabelText('Merchant');
		if (action === 'Save & verify') {
			await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Edited' } });
		}
		await fireEvent.click(screen.getByRole('button', { name: action }));
		const dialog = await screen.findByRole('dialog');
		const confirm = within(dialog).getByRole('button', { name: 'Verify' }) as HTMLButtonElement;
		expect(confirm.disabled).toBe(true);
		await fireEvent.input(within(dialog).getByRole('textbox'), {
			target: { value: 'Checked receipt' }
		});
		await fireEvent.click(confirm);
		await waitFor(() =>
			expect(verify).toHaveBeenCalledWith(row.id, {
				expected_operation_revision: displayed.operation_revision,
				expected_review_revision: 3,
				acknowledgements: [{ finding_id: 'live', comment: 'Checked receipt' }]
			})
		);
	}
);
