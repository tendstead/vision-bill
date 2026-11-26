import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import ExtractionDraftEditor from './ExtractionDraftEditor.svelte';
import { setLocale } from '$lib/i18n';
import { api, ApiError } from '$lib/api/client';
import { queryClient } from '$lib/query/client';
import type { DraftView } from '$lib/types';
const draft = {
	image_id: 'job',
	revision: 7,
	field_sources: {},
	overlap_decisions: [],
	document: {
		merchant_name: null,
		subtotal: null,
		total: null,
		currency: null,
		line_items: [
			{ description: null, quantity: null, unit_price: null, total_price: null, tags: null }
		]
	},
	amount_policy: {
		tax_basis: 'unknown' as const,
		discount_treatment: 'unknown' as const,
		tip_treatment: 'unknown' as const,
		subtotal_origin: 'unknown' as const,
		tax_information: 'missing' as const
	},
	findings: [],
	source_views: [],
	finalized_receipt_id: null
};
const completeDraft: DraftView = {
	...draft,
	document: {
		confidence: 90,
		merchant_name: 'Shop',
		date: '2026-01-01',
		currency: 'EUR',
		category: 'grocery',
		line_items: [{ description: 'Bread', quantity: null, unit_price: null, total_price: '3' }],
		taxes: [],
		subtotal: '3',
		discount_total: null,
		tax_total: '0',
		tip: null,
		total: '3',
		payment_method: 'cash'
	}
};
beforeEach(() => {
	queryClient.clear();
	vi.spyOn(api, 'listTags').mockResolvedValue([]);
	vi.spyOn(api, 'listSpendingCategories').mockResolvedValue([]);
	vi.stubGlobal('matchMedia', () => ({ matches: false }));
	vi.spyOn(api, 'getImage').mockResolvedValue({ additional_images: [] } as never);
	vi.spyOn(api, 'getDraft').mockResolvedValue(draft);
});
afterEach(() => {
	queryClient.clear();
	cleanup();
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
});
it('partial saves preserve nulls, missing fields and actual revision', async () => {
	const patch = vi.spyOn(api, 'patchDraft').mockResolvedValue({
		...draft,
		revision: 8,
		document: { ...draft.document, merchant_name: 'Shop' }
	});
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Shop' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	expect(patch).toHaveBeenCalledWith('job', {
		expected_revision: 7,
		changes: { merchant_name: 'Shop' }
	});
	expect((screen.getByLabelText('Currency') as HTMLInputElement).value).toBe('');
	expect((screen.getByLabelText('Qty') as HTMLInputElement).value).toBe('');
});
it('conflicts preserve dirty input and original expected revision', async () => {
	const patch = vi.spyOn(api, 'patchDraft').mockRejectedValue(new ApiError(409, 'Draft conflict'));
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'My edit' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	await screen.findByRole('alert');
	expect((screen.getByLabelText('Merchant') as HTMLInputElement).value).toBe('My edit');
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	expect(patch.mock.calls[1][1].expected_revision).toBe(7);
});

it('preserves visible precise time through a merchant-only partial save', async () => {
	const loaded = { ...draft, document: { ...draft.document, time: '14:30:00.000001' } };
	vi.mocked(api.getDraft).mockResolvedValue(loaded);
	const patch = vi.spyOn(api, 'patchDraft').mockResolvedValue({
		...loaded,
		revision: 8,
		document: { ...loaded.document, merchant_name: 'Shop' }
	});
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByLabelText('Time');
	expect((screen.getByLabelText('Time') as HTMLInputElement).value).toBe('14:30:00.000001');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Shop' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	expect(patch).toHaveBeenCalledWith('job', {
		expected_revision: 7,
		changes: { merchant_name: 'Shop' }
	});
	await waitFor(() =>
		expect((screen.getByRole('button', { name: 'Save draft' }) as HTMLButtonElement).disabled).toBe(
			true
		)
	);
	expect((screen.getByLabelText('Time') as HTMLInputElement).value).toBe('14:30:00.000001');
	expect((screen.getByLabelText('Subtotal') as HTMLInputElement).value).toBe('');
});

it('sends precise draft time, adopts canonical response and retains rejected edits', async () => {
	const patch = vi
		.spyOn(api, 'patchDraft')
		.mockResolvedValueOnce({
			...draft,
			revision: 8,
			document: { ...draft.document, time: '14:30:15.100000' }
		})
		.mockRejectedValueOnce(new ApiError(422, 'Invalid local time'));
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByLabelText('Time');
	await fireEvent.input(screen.getByLabelText('Time'), { target: { value: '14:30:15.1' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	expect(patch).toHaveBeenCalledWith('job', {
		expected_revision: 7,
		changes: { time: '14:30:15.1' }
	});
	await screen.findByDisplayValue('14:30:15.100000');
	await fireEvent.input(screen.getByLabelText('Time'), { target: { value: '14:30:15.1234567' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	await screen.findByRole('alert');
	expect((screen.getByLabelText('Time') as HTMLInputElement).value).toBe('14:30:15.1234567');
	expect(patch.mock.calls[1][1]).toEqual({
		expected_revision: 8,
		changes: { time: '14:30:15.1234567' }
	});
});
it('shows required missing fields without attempting incomplete finalization', async () => {
	const finalize = vi.spyOn(api, 'finalizeDraft');
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByLabelText('Merchant');
	await fireEvent.click(screen.getByRole('button', { name: 'Finalize draft' }));
	expect(screen.getByRole('alert').textContent).toContain('Required missing fields');
	await fireEvent.click(screen.getByRole('button', { name: 'Merchant' }));
	expect(document.activeElement).toBe(screen.getByLabelText('Merchant'));
	expect(finalize).not.toHaveBeenCalled();
});

it('finalizes a saved draft with absent discounts, tips, quantities and unit prices', async () => {
	vi.mocked(api.getDraft).mockResolvedValue(completeDraft);
	const finalize = vi.spyOn(api, 'finalizeDraft').mockResolvedValue({ id: 'receipt' } as never);
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByDisplayValue('Bread');
	await fireEvent.click(screen.getByRole('button', { name: 'Finalize draft' }));
	expect(finalize).toHaveBeenCalledWith('job', {
		expected_revision: 7,
		request_key: expect.any(String)
	});
});

it('saves partial corrections with an out-of-range tax observation', async () => {
	const loaded = {
		...completeDraft,
		document: { ...completeDraft.document, taxes: [{ name: 'VAT', rate: 7, amount: '0' }] }
	};
	vi.mocked(api.getDraft).mockResolvedValue(loaded);
	const patch = vi.spyOn(api, 'patchDraft').mockResolvedValue({
		...loaded,
		revision: 8,
		document: { ...loaded.document, merchant_name: 'Shop corrected' }
	});
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Shop corrected' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	expect(patch).toHaveBeenCalledWith('job', {
		expected_revision: 7,
		changes: { merchant_name: 'Shop corrected' }
	});
	expect((screen.getByLabelText('Tax rate (fraction)') as HTMLInputElement).value).toBe('7');
});

it.each([7, '1.00000000000000001'])(
	'prompts for a fractional tax rate before finalizing %s',
	async (rate) => {
		vi.mocked(api.getDraft).mockResolvedValue({
			...completeDraft,
			document: { ...completeDraft.document, taxes: [{ name: 'VAT', rate, amount: '0' }] }
		});
		const finalize = vi.spyOn(api, 'finalizeDraft');
		render(ExtractionDraftEditor, { id: 'job' });
		await screen.findByLabelText('Tax rate (fraction)');
		await fireEvent.click(screen.getByRole('button', { name: 'Finalize draft' }));
		expect(screen.getByRole('alert').textContent).toContain('between 0 and 1');
		expect(finalize).not.toHaveBeenCalled();
	}
);

it('finalizes the current saved revision with one stable key across ambiguous responses', async () => {
	const complete = {
		...draft,
		document: {
			confidence: 90,
			merchant_name: 'Shop',
			date: '2026-01-01',
			currency: 'EUR',
			category: 'grocery' as const,
			line_items: [],
			taxes: [],
			subtotal: '0',
			discount_total: '0',
			tax_total: '0',
			total: '0',
			payment_method: 'cash' as const
		}
	};
	vi.mocked(api.getDraft).mockResolvedValue(complete);
	const finalize = vi
		.spyOn(api, 'finalizeDraft')
		.mockRejectedValueOnce(new ApiError(0, 'Network error'))
		.mockResolvedValue({
			...complete.document,
			id: 'receipt',
			status: 'unverified',
			verified: false
		} as never);
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByLabelText('Merchant');
	await fireEvent.click(screen.getByRole('button', { name: 'Finalize draft' }));
	await screen.findByRole('alert');
	await fireEvent.click(screen.getByRole('button', { name: 'Finalize draft' }));
	expect(finalize).toHaveBeenCalledTimes(2);
	expect(finalize.mock.calls[0][1]).toEqual(finalize.mock.calls[1][1]);
	expect(finalize.mock.calls[1][1]).toMatchObject({
		expected_revision: 7,
		request_key: expect.any(String)
	});
});
it('ignores a late save after switching to another job', async () => {
	let done!: (value: typeof draft) => void;
	vi.spyOn(api, 'patchDraft').mockImplementation(
		() =>
			new Promise((resolve) => {
				done = resolve as typeof done;
			})
	);
	const view = render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByLabelText('Merchant');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'old edit' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	vi.mocked(api.getDraft).mockResolvedValue({
		...draft,
		image_id: 'other',
		document: { merchant_name: 'Other job' }
	});
	await view.rerender({ id: 'other' });
	await screen.findByDisplayValue('Other job');
	done({ ...draft, revision: 8 });
	expect((screen.getByLabelText('Merchant') as HTMLInputElement).value).toBe('Other job');
});

it('uses the owner tag vocabulary when reusing tag controls in a nullable draft', async () => {
	const tags = vi.mocked(api.listTags).mockResolvedValue(['food']);
	vi.mocked(api.getDraft).mockResolvedValue({
		...draft,
		document: { ...draft.document, line_items: [{ description: 'Item', tags: ['food'] }] }
	});
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByLabelText('Merchant');
	expect(tags).toHaveBeenCalledOnce();
	expect(screen.queryByText('Suggested')).toBeNull();
});
it('drops ambiguous tax sources when a preceding draft tax is deleted', async () => {
	const source = {
		part_id: 'part',
		photo_index: 0,
		source_sha256: 'hash',
		coordinate_space: 'unknown' as const
	};
	vi.mocked(api.getDraft).mockResolvedValue({
		...draft,
		field_sources: { 'taxes.0.amount': [source], 'taxes.1.amount': [source], total: [source] },
		document: {
			...draft.document,
			line_items: [],
			taxes: [
				{ name: 'A', amount: '1' },
				{ name: 'B', amount: '2' }
			]
		}
	});
	const patch = vi.spyOn(api, 'patchDraft').mockResolvedValue({ ...draft, revision: 8 });
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByLabelText('Merchant');
	await fireEvent.click(screen.getAllByRole('button', { name: 'Remove' })[0]);
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	expect(patch.mock.calls[0][1].changes.taxes?.[0].name).toBe('B');
	expect(patch.mock.calls[0][1].changes.field_sources).toEqual({ total: [source] });
});

it.each([
	['preceding deletion', 1, 2],
	['out of range', 0, 9]
] as const)(
	'keeps safe draft display and evidence round-trip after %s',
	async (kind, first_index, second_index) => {
		vi.mocked(api.getDraft).mockResolvedValue({
			...draft,
			overlap_decisions: [{ first_index, second_index, verified: true }],
			document: {
				...draft.document,
				line_items: [{ description: 'A' }, { description: 'B' }, { description: 'C' }]
			}
		});
		const patch = vi.spyOn(api, 'patchDraft').mockResolvedValue({ ...draft, revision: 8 });
		render(ExtractionDraftEditor, { id: 'job' });
		await screen.findByLabelText('Merchant');
		if (kind === 'preceding deletion')
			await fireEvent.click(screen.getAllByRole('button', { name: 'Remove' })[0]);
		else await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Changed' } });
		expect(
			(
				screen.getByRole('checkbox', {
					name: 'I reviewed this overlap; keep the current rows'
				}) as HTMLInputElement
			).disabled
		).toBe(true);
		expect(screen.getByText('Row association unknown; review the original evidence.')).toBeTruthy();
		await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
		expect(patch.mock.calls[0][1].changes.overlap_decisions).toEqual(
			kind === 'preceding deletion' ? [{ verified: false, association_unknown: true }] : undefined
		);
	}
);

it.each([1, 2])(
	'adopts producer removal evidence verbatim without making a draft dirty (removed %s)',
	async (removed_index) => {
		const decisions = [
			{ first_index: removed_index - 1, second_index: removed_index, verified: true, removed_index }
		];
		const loaded = {
			...draft,
			overlap_decisions: decisions,
			document: { ...draft.document, line_items: [{ description: 'A' }, { description: 'B' }] }
		};
		vi.mocked(api.getDraft).mockResolvedValue(loaded);
		const patch = vi.spyOn(api, 'patchDraft').mockResolvedValue({
			...loaded,
			revision: 8,
			document: { ...loaded.document, merchant_name: 'Changed' }
		});
		render(ExtractionDraftEditor, { id: 'job' });
		await screen.findByLabelText('Merchant');
		expect((screen.getByRole('button', { name: 'Save draft' }) as HTMLButtonElement).disabled).toBe(
			true
		);
		expect(patch).not.toHaveBeenCalled();
		expect(screen.queryByText(/^Check rows/)).toBeNull();
		expect(
			(
				screen.getByRole('checkbox', {
					name: 'I reviewed this overlap; keep the current rows'
				}) as HTMLInputElement
			).disabled
		).toBe(true);
		await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Changed' } });
		await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
		expect(patch).toHaveBeenCalledWith('job', {
			expected_revision: 7,
			changes: { merchant_name: 'Changed' }
		});
		await waitFor(() =>
			expect(
				(screen.getByRole('button', { name: 'Save draft' }) as HTMLButtonElement).disabled
			).toBe(true)
		);
	}
);

it('blocks save, finalize and removal during pending tag creation', async () => {
	vi.mocked(api.listTags).mockResolvedValue(['food']);
	let finish!: (value: { name: string; created: boolean }) => void;
	vi.spyOn(api, 'createTag').mockImplementation(
		() =>
			new Promise((resolve) => {
				finish = resolve;
			})
	);
	vi.mocked(api.getDraft).mockResolvedValue({
		...draft,
		document: {
			...draft.document,
			line_items: [
				{ description: 'First', tags: [] },
				{ description: 'Second', tags: [] }
			]
		}
	});
	const patch = vi.spyOn(api, 'patchDraft').mockResolvedValue(draft);
	const finalize = vi.spyOn(api, 'finalizeDraft');
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByDisplayValue('First');
	await fireEvent.click(screen.getAllByRole('button', { name: 'Tags' })[0]);
	await fireEvent.input(screen.getByPlaceholderText('Search tags…'), { target: { value: 'new' } });
	await fireEvent.click(screen.getByRole('option', { name: 'Create “new”' }));
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Changed' } });
	for (const name of ['Save draft', 'Finalize draft', 'Remove'])
		for (const button of screen.getAllByRole('button', { name }))
			expect((button as HTMLButtonElement).disabled).toBe(true);
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	await fireEvent.click(screen.getByRole('button', { name: 'Finalize draft' }));
	expect(patch).not.toHaveBeenCalled();
	expect(finalize).not.toHaveBeenCalled();
	finish({ name: 'new', created: true });
	await waitFor(() =>
		expect((screen.getByRole('button', { name: 'Save draft' }) as HTMLButtonElement).disabled).toBe(
			false
		)
	);
	await fireEvent.click(screen.getAllByRole('button', { name: 'Remove' })[0]);
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	expect(patch.mock.calls[0][1].changes.line_items).toEqual([{ description: 'Second', tags: [] }]);
});

it('ignores pending tag completion after draft replacement', async () => {
	vi.mocked(api.listTags).mockResolvedValue(['food']);
	let finish!: (value: { name: string; created: boolean }) => void;
	vi.spyOn(api, 'createTag').mockImplementation(
		() =>
			new Promise((resolve) => {
				finish = resolve;
			})
	);
	vi.mocked(api.getDraft).mockResolvedValue({
		...draft,
		document: { ...draft.document, line_items: [{ description: 'First', tags: [] }] }
	});
	const view = render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByDisplayValue('First');
	await fireEvent.click(screen.getByRole('button', { name: 'Tags' }));
	await fireEvent.input(screen.getByPlaceholderText('Search tags…'), { target: { value: 'new' } });
	await fireEvent.click(screen.getByRole('option', { name: 'Create “new”' }));
	vi.mocked(api.getDraft).mockResolvedValue({
		...draft,
		image_id: 'other',
		document: { ...draft.document, line_items: [{ description: 'Second', tags: [] }] }
	});
	await view.rerender({ id: 'other' });
	await screen.findByDisplayValue('Second');
	finish({ name: 'new', created: true });
	await new Promise((resolve) => setTimeout(resolve, 0));
	expect(screen.queryByRole('button', { name: 'Remove “new”' })).toBeNull();
	expect((screen.getByRole('button', { name: 'Save draft' }) as HTMLButtonElement).disabled).toBe(
		true
	);
});

it('reopens the same source after returning to Data and Original', async () => {
	const source = { photo_index: 0, view_id: 'working', coordinate_space: 'working', bbox: null };
	vi.mocked(api.getDraft).mockResolvedValue({
		...completeDraft,
		field_sources: { total: [source] },
		source_views: [
			{
				view_id: 'working',
				photo_index: 0,
				coordinate_space: 'working',
				width: 100,
				height: 200,
				url: '/working.png'
			}
		]
	} as unknown as DraftView);
	render(ExtractionDraftEditor, { id: 'job' });
	const evidence = await screen.findByRole('button', { name: 'Show source' });
	await fireEvent.click(evidence);
	expect(screen.getByRole('button', { name: 'Receipt' }).getAttribute('aria-pressed')).toBe('true');
	await fireEvent.click(screen.getByRole('button', { name: 'Original photos' }));
	await fireEvent.click(screen.getByRole('button', { name: 'Data' }));
	await fireEvent.click(evidence);
	expect(screen.getByRole('button', { name: 'Receipt' }).getAttribute('aria-pressed')).toBe('true');
	expect(screen.getByRole('button', { name: 'Working image' }).getAttribute('aria-pressed')).toBe(
		'true'
	);
});

it('summarizes missing tax names and amounts as navigable fields', async () => {
	vi.mocked(api.getDraft).mockResolvedValue({
		...completeDraft,
		document: { ...completeDraft.document, taxes: [{ name: null, amount: null, rate: null }] }
	});
	const finalize = vi.spyOn(api, 'finalizeDraft');
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByDisplayValue('Bread');
	await fireEvent.click(screen.getByRole('button', { name: 'Finalize draft' }));
	await fireEvent.click(screen.getByRole('button', { name: '1 · Amount' }));
	expect(document.activeElement).toBe(screen.getByLabelText('Amount'));
	expect(screen.getByRole('button', { name: '1 · Name' })).toBeTruthy();
	expect(finalize).not.toHaveBeenCalled();
});
it('labels a missing taxes array as Taxes', async () => {
	vi.mocked(api.getDraft).mockResolvedValue({
		...completeDraft,
		document: { ...completeDraft.document, taxes: null }
	});
	render(ExtractionDraftEditor, { id: 'job' });
	await screen.findByDisplayValue('Bread');
	await fireEvent.click(screen.getByRole('button', { name: 'Finalize draft' }));
	await fireEvent.click(screen.getByRole('button', { name: 'Taxes' }));
	expect(document.activeElement?.id).toBe('draft-taxes');
});

it('opens the receipt pane for an explicitly unknown field source', async () => {
	vi.mocked(api.getDraft).mockResolvedValue({
		...completeDraft,
		field_sources: { total: [] },
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
	} as unknown as DraftView);
	render(ExtractionDraftEditor, { id: 'job' });
	await fireEvent.click(await screen.findByRole('button', { name: 'Show source' }));
	expect(screen.getByRole('button', { name: 'Receipt' }).getAttribute('aria-pressed')).toBe('true');
	expect(screen.queryByRole('img', { name: 'Source view' })).toBeNull();
	expect(screen.getByText('Source mapping unknown')).toBeTruthy();
});

it('saves and reloads an explicitly printed net subtotal independently of gross line prices', async () => {
	const loaded: DraftView = {
		...completeDraft,
		amount_policy: { ...completeDraft.amount_policy, tax_basis: 'gross' },
		document: { ...completeDraft.document, taxes: [{ name: 'VAT', rate: '19.0', amount: '1.90' }] }
	};
	const saved: DraftView = {
		...loaded,
		revision: loaded.revision + 1,
		amount_policy: { ...loaded.amount_policy, subtotal_origin: 'printed_net' }
	};
	vi.mocked(api.getDraft).mockResolvedValue(loaded);
	const patch = vi.spyOn(api, 'patchDraft').mockResolvedValue(saved);
	render(ExtractionDraftEditor, { id: 'job' });
	const select = await screen.findByRole('combobox', { name: /Subtotal origin/ });
	await fireEvent.change(select, { target: { value: 'printed_net' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	expect(patch).toHaveBeenCalledWith('job', {
		expected_revision: loaded.revision,
		changes: { amount_policy: saved.amount_policy }
	});
	expect((select as HTMLSelectElement).value).toBe('printed_net');
	const rate = screen.getByLabelText('Tax rate (fraction)') as HTMLInputElement;
	expect(rate.value).toBe('19.0');
	expect(rate.getAttribute('aria-describedby')).toBe(screen.getByText(/Enter 0.19 for 19%/).id);
});

it('saves selected cash and numeric confidence before finalizing the newly checked revision', async () => {
	const loaded: DraftView = {
		...completeDraft,
		document: { ...completeDraft.document, payment_method: null },
		findings: [
			{
				id: 'missing',
				code: 'missing_mandatory_field',
				field_path: 'payment_method',
				severity: 'integrity',
				comparison: {},
				source_refs: []
			}
		]
	};
	vi.mocked(api.getDraft).mockResolvedValue(loaded);
	const saved: DraftView = {
		...completeDraft,
		revision: 8,
		document: { ...completeDraft.document, confidence: 100 }
	};
	const patch = vi.spyOn(api, 'patchDraft').mockResolvedValue(saved);
	const finalize = vi.spyOn(api, 'finalizeDraft').mockResolvedValue({ id: 'receipt' } as never);
	render(ExtractionDraftEditor, { id: 'job' });
	const select = await screen.findByLabelText('Payment method', { exact: true });
	expect((select as HTMLSelectElement).labels?.[0].textContent).toBe('Payment method');
	await fireEvent.click(screen.getByRole('button', { name: 'Finalize draft' }));
	expect(screen.getByRole('button', { name: 'Payment method' })).toBeTruthy();
	try {
		setLocale('de');
		expect(await screen.findByLabelText('Zahlungsart', { exact: true })).toBe(select);
		expect(await screen.findByRole('button', { name: 'Zum Feld Zahlungsart' })).toBeTruthy();
	} finally {
		setLocale('en');
	}
	await fireEvent.click(await screen.findByRole('button', { name: 'Go to field Payment method' }));
	expect(document.activeElement).toBe(select);
	await fireEvent.change(select, { target: { value: 'cash' } });
	await fireEvent.input(screen.getByLabelText('Confidence (%)'), { target: { value: '100' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Finalize draft' }));
	expect(finalize).not.toHaveBeenCalled();
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	expect(patch).toHaveBeenCalledWith('job', {
		expected_revision: 7,
		changes: { payment_method: 'cash', confidence: 100 }
	});
	await waitFor(() =>
		expect(screen.queryByRole('button', { name: 'Go to field Payment method' })).toBeNull()
	);
	await fireEvent.click(screen.getByRole('button', { name: 'Finalize draft' }));
	expect(finalize).toHaveBeenCalledWith('job', {
		expected_revision: 8,
		request_key: expect.any(String)
	});
});

it('shows the editable field from a server validation error and preserves the rejected changes', async () => {
	vi.mocked(api.getDraft).mockResolvedValue(completeDraft);
	vi.spyOn(api, 'patchDraft').mockRejectedValue(
		new ApiError(422, 'details', undefined, { code: 'draft_invalid_fields', fields: ['time'] })
	);
	render(ExtractionDraftEditor, { id: 'job' });
	const time = await screen.findByLabelText('Time');
	await fireEvent.input(time, { target: { value: 'missing' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	expect((await screen.findByRole('alert')).textContent).toContain('Invalid fields');
	await fireEvent.click(screen.getByRole('button', { name: 'Time' }));
	expect(document.activeElement).toBe(time);
	expect((time as HTMLInputElement).value).toBe('missing');
});

it('submits canonical receipt and spending categories while retaining a historical option until changed', async () => {
	const loaded: DraftView = {
		...completeDraft,
		document: {
			...completeDraft.document,
			line_items: [{ description: 'Item', spending_category: 'historical' }]
		}
	};
	vi.mocked(api.getDraft).mockResolvedValue(loaded);
	vi.mocked(api.listSpendingCategories).mockResolvedValue([{ code: 'milk' }]);
	const patch = vi.spyOn(api, 'patchDraft').mockResolvedValue(loaded);
	render(ExtractionDraftEditor, { id: 'job' });
	const spending = await screen.findByRole('combobox', { name: 'Spending category' });
	expect((spending as HTMLSelectElement).value).toBe('historical');
	await fireEvent.change(spending, { target: { value: 'milk' } });
	await fireEvent.change(screen.getByRole('combobox', { name: 'Category' }), {
		target: { value: 'restaurant' }
	});
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	expect(patch).toHaveBeenCalledWith('job', {
		expected_revision: 7,
		changes: {
			category: 'restaurant',
			line_items: [{ description: 'Item', spending_category: 'milk' }]
		}
	});
});

it('keeps findings visible when revisiting a finalized draft without offering field actions', async () => {
	vi.mocked(api.getDraft).mockResolvedValue({
		...completeDraft,
		finalized_receipt_id: 'receipt',
		findings: [
			{
				id: 'missing',
				code: 'missing_mandatory_field',
				field_path: 'payment_method',
				severity: 'integrity',
				comparison: {},
				source_refs: []
			}
		]
	});
	render(ExtractionDraftEditor, { id: 'job' });
	expect(await screen.findByRole('article')).toBeTruthy();
	expect(screen.queryByRole('button', { name: 'Go to field Payment method' })).toBeNull();
	expect(screen.queryByLabelText('Payment method')).toBeNull();
});

it('focuses a nested server validation field and omits unavailable row actions', async () => {
	vi.mocked(api.getDraft).mockResolvedValue(completeDraft);
	vi.spyOn(api, 'patchDraft').mockRejectedValue(
		new ApiError(422, 'details', undefined, {
			code: 'draft_invalid_fields',
			fields: ['line_items[0].total_price', 'line_items.0.tags', 'line_items.9.description']
		})
	);
	render(ExtractionDraftEditor, { id: 'job' });
	const merchant = await screen.findByLabelText('Merchant');
	await fireEvent.input(merchant, { target: { value: 'Edited' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	await screen.findByRole('alert');
	const jump = screen.getByRole('button', { name: '1 · Total' });
	await fireEvent.click(jump);
	expect(document.activeElement?.id).toBe('draft-line_items.0.total_price');
	expect(document.activeElement?.getAttribute('aria-invalid')).toBe('true');
	expect(screen.queryByRole('button', { name: /^10 · / })).toBeNull();
	await fireEvent.click(screen.getByRole('button', { name: '1 · Tags' }));
	expect(document.activeElement?.id).toBe('draft-line_items.0.tags');
});

it('patches only the canonical merchant name when the printed merchant stays unchanged', async () => {
	vi.mocked(api.getDraft).mockResolvedValue(completeDraft);
	const patch = vi.spyOn(api, 'patchDraft').mockResolvedValue({
		...completeDraft,
		revision: 8,
		document: { ...completeDraft.document, merchant_display_name: 'Shop Group' }
	});
	render(ExtractionDraftEditor, { id: 'job' });
	const display = await screen.findByLabelText('Merchant display name');
	expect((display as HTMLInputElement).maxLength).toBe(200);
	await fireEvent.input(display, { target: { value: 'Shop Group' } });
	expect((screen.getByLabelText('Merchant') as HTMLInputElement).value).toBe('Shop');
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	expect(patch).toHaveBeenCalledWith('job', {
		expected_revision: 7,
		changes: { merchant_display_name: 'Shop Group' }
	});
});
