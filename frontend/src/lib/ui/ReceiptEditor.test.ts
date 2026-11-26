import { api } from '$lib/api/client';
import { setLocale } from '$lib/i18n';
import { session } from '$lib/session-lifetime';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LineItemRow, ReceiptRow, ReceiptWrite } from '$lib/types';
import { queryClient } from '$lib/query/client';
import ReceiptEditor from './ReceiptEditor.svelte';

const receipt: ReceiptRow = {
	operation_revision: 1,
	review_revision: 1,
	amount_policy: {
		tax_basis: 'unknown',
		discount_treatment: 'unknown',
		tip_treatment: 'unknown',
		subtotal_origin: 'unknown',
		tax_information: 'missing'
	},
	findings: [],
	field_sources: {},
	overlap_decisions: [],
	acknowledgements: [],
	verification_source: null,
	checks_version: null,
	id: 'receipt-1',
	confidence: 90,
	merchant_name: 'Shop A',
	merchant_address: null,
	receipt_number: null,
	date: '2026-09-10',
	time: null,
	currency: 'EUR',
	category: 'grocery',
	subtotal: '2.00',
	discount_total: '0',
	tax_total: '0',
	tip: null,
	total: '2.00',
	payment_method: 'cash',
	created_at: null,
	status: 'unverified',
	image_id: null,
	thumbnail_path: null,
	verified: false,
	user_id: null
};

function line(
	category = 'unknown',
	source: LineItemRow['category_source'] = 'unknown'
): LineItemRow {
	return {
		id: 'line-1',
		receipt_id: receipt.id,
		description: 'MUESLI',
		original_description: null,
		quantity: 1,
		unit_price: '2.00',
		total_price: '2.00',
		spending_category: category,
		category_source: source,
		tags: []
	};
}

function response(body: unknown): Response {
	return new Response(JSON.stringify(body), {
		status: 200,
		headers: { 'Content-Type': 'application/json' }
	});
}

function deferred<T>() {
	let resolve!: (value: T) => void;
	const promise = new Promise<T>((done) => {
		resolve = done;
	});
	return { promise, resolve };
}

let suggestions: ReturnType<typeof deferred<Response>>[];

beforeEach(() => {
	queryClient.clear();
	suggestions = [];
	vi.stubGlobal(
		'fetch',
		vi.fn((input: RequestInfo | URL) => {
			const url = String(input);
			if (url.includes('/category-suggestion')) {
				const pending = deferred<Response>();
				suggestions.push(pending);
				return pending.promise;
			}
			if (url.endsWith('/tags/categories'))
				return Promise.resolve(
					response([{ code: 'unknown' }, { code: 'milk' }, { code: 'cheese' }])
				);
			if (url.endsWith('/tags') || url.endsWith('/collections'))
				return Promise.resolve(response([]));
			return Promise.resolve(response({}));
		})
	);
});

beforeEach(() =>
	session.set({ id: 'viewer', username: 'alice', is_admin: false, can_see_all: false })
);

afterEach(() => {
	cleanup();
	queryClient.clear();
	vi.unstubAllGlobals();
});

describe('receipt spending category editor', () => {
	it('ignores an older lookup when the description changes and uses the latest history result', async () => {
		render(ReceiptEditor, { receipt, lineItems: [line()], taxes: [], onSave: vi.fn() });
		const description = screen.getByLabelText('Description');
		await fireEvent.blur(description);
		await waitFor(() => expect(suggestions).toHaveLength(1));
		await fireEvent.input(description, { target: { value: 'MILCH' } });
		await fireEvent.blur(description);
		await waitFor(() => expect(suggestions).toHaveLength(2));
		suggestions[1].resolve(response({ category: 'milk', source: 'history' }));
		await waitFor(() =>
			expect((screen.getByLabelText('Spending category') as HTMLSelectElement).value).toBe('milk')
		);
		suggestions[0].resolve(response({ category: 'cheese', source: 'history' }));
		await waitFor(() =>
			expect((screen.getByLabelText('Spending category') as HTMLSelectElement).value).toBe('milk')
		);
	});

	it('keeps an explicit choice when history replies late and clears pending memory on merchant edit', async () => {
		const onSave = vi.fn<(write: ReceiptWrite) => void>();
		render(ReceiptEditor, { receipt, lineItems: [line()], taxes: [], onSave });
		const description = screen.getByLabelText('Description');
		await fireEvent.blur(description);
		await waitFor(() => expect(suggestions).toHaveLength(1));
		await waitFor(() => expect(screen.getByRole('option', { name: 'Milk' })).not.toBeNull());
		await fireEvent.change(screen.getByLabelText('Spending category'), {
			target: { value: 'milk' }
		});
		suggestions[0].resolve(response({ category: 'cheese', source: 'history' }));
		await waitFor(() =>
			expect(screen.getByText(/Previous purchase suggests Cheese/)).not.toBeNull()
		);
		await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Shop B' } });
		expect((screen.getByLabelText('Spending category') as HTMLSelectElement).value).toBe('milk');
		await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
		expect(onSave).toHaveBeenCalledOnce();
		expect(onSave.mock.calls[0][0].line_items[0]).toMatchObject({
			id: 'line-1',
			spending_category: 'milk'
		});
		expect(onSave.mock.calls[0][0].line_items[0]).not.toHaveProperty('remember_category');
		expect(onSave.mock.calls[0][0].line_items[0]).not.toHaveProperty('original_description');
	});

	it('sends memory only for an explicit category action, not a generic save', async () => {
		const onSave = vi.fn<(write: ReceiptWrite) => void>();
		render(ReceiptEditor, { receipt, lineItems: [line('milk', 'llm')], taxes: [], onSave });
		await fireEvent.input(screen.getByLabelText('Qty'), { target: { value: '2' } });
		await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
		expect(onSave.mock.calls[0][0].line_items[0]).not.toHaveProperty('remember_category');
		await fireEvent.click(screen.getByRole('button', { name: 'Remember for this item' }));
		await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
		expect(onSave.mock.calls[1][0].line_items[0]).toMatchObject({
			id: 'line-1',
			spending_category: 'milk',
			remember_category: true
		});
	});

	it('resets remembered choices and item IDs after a successful detail refresh', async () => {
		const onSave = vi.fn<(write: ReceiptWrite) => void>();
		const view = render(ReceiptEditor, {
			receipt,
			lineItems: [line('milk', 'llm')],
			taxes: [],
			onSave,
			revision: 0,
			savedRevision: 0
		});
		await fireEvent.click(screen.getByRole('button', { name: 'Remember for this item' }));
		await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
		expect(onSave.mock.calls[0][0].line_items[0].remember_category).toBe(true);
		await view.rerender({
			receipt,
			lineItems: [{ ...line('milk', 'user'), id: 'server-id-2' }],
			taxes: [],
			onSave,
			revision: 1,
			savedRevision: 1
		});
		await waitFor(() => expect(screen.queryByRole('button', { name: 'Save' })).toBeNull());
		await fireEvent.input(screen.getByLabelText('Qty'), { target: { value: '2' } });
		await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
		expect(onSave.mock.calls[1][0].line_items[0]).toMatchObject({
			id: 'server-id-2',
			spending_category: 'milk'
		});
		expect(onSave.mock.calls[1][0].line_items[0]).not.toHaveProperty('remember_category');
	});
});

it('captures actual operation revision and explicit policy/source decisions without rebasing dirty edits', async () => {
	const onSave = vi.fn();
	const view = render(ReceiptEditor, {
		receipt: { ...receipt, operation_revision: 9 },
		lineItems: [line()],
		taxes: [],
		onSave
	});
	await fireEvent.input(document.querySelector('#re-total')!, {
		target: { value: '2.00000000000001' }
	});
	await view.rerender({
		receipt: { ...receipt, operation_revision: 10 },
		lineItems: [line()],
		taxes: [],
		onSave
	});
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(onSave.mock.calls[0][0]).toMatchObject({
		expected_operation_revision: 9,
		total: '2.00000000000001',
		amount_policy: receipt.amount_policy,
		field_sources: {},
		overlap_decisions: []
	});
	expect(screen.queryByText(/Expected total/)).toBeNull();
});
it('drops shifted numeric source paths after deletion rather than assigning them to another item', async () => {
	const onSave = vi.fn();
	const source = {
		part_id: 'part',
		photo_index: 0,
		source_sha256: 'hash',
		coordinate_space: 'unknown' as const
	};
	render(ReceiptEditor, {
		receipt: {
			...receipt,
			field_sources: {
				'line_items.0.total_price': [source],
				'line_items.1.total_price': [source],
				total: [source]
			}
		},
		lineItems: [line(), { ...line(), id: 'second' }],
		taxes: [],
		onSave
	});
	await fireEvent.click(screen.getAllByRole('button', { name: 'Remove' })[0]);
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(onSave.mock.calls[0][0].line_items[0].id).toBe('second');
	expect(Object.keys(onSave.mock.calls[0][0].field_sources)).toEqual(['total']);
});

it('submits an explicit reviewed overlap decision and invalidates it when its row order changes', async () => {
	const onSave = vi.fn();
	const overlap = { first_index: 0, second_index: 1, verified: null };
	render(ReceiptEditor, {
		receipt: { ...receipt, overlap_decisions: [overlap] },
		lineItems: [line(), { ...line(), id: 'second' }, { ...line(), id: 'third' }],
		taxes: [],
		onSave
	});
	await fireEvent.click(
		screen.getByRole('checkbox', { name: 'I reviewed this overlap; keep the current rows' })
	);
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(onSave.mock.calls[0][0].overlap_decisions).toEqual([{ ...overlap, verified: true }]);
	await fireEvent.click(screen.getAllByRole('button', { name: 'Move item down' })[1]);
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(onSave.mock.calls[1][0].line_items.map((line: { id: string }) => line.id)).toEqual([
		'line-1',
		'third',
		'second'
	]);
	expect(onSave.mock.calls[1][0].overlap_decisions[0]).toEqual({
		verified: false,
		association_unknown: true
	});
	expect(
		(
			screen.getByRole('checkbox', {
				name: 'I reviewed this overlap; keep the current rows'
			}) as HTMLInputElement
		).disabled
	).toBe(true);
	expect(screen.queryByText('Check rows 1 and 2 against the photos.')).toBeNull();
});

it('navigates from a field source decision to its exact source references', async () => {
	const source = {
		part_id: 'part',
		photo_index: 0,
		source_sha256: 'hash',
		coordinate_space: 'working' as const,
		view_id: 'view',
		bbox: [
			[1, 1],
			[2, 1],
			[2, 2]
		]
	};
	const onSource = vi.fn();
	render(ReceiptEditor, {
		receipt: { ...receipt, field_sources: { total: [source] } },
		lineItems: [line()],
		taxes: [],
		onSource
	});
	await fireEvent.click(screen.getByText('Source decisions'));
	await fireEvent.click(screen.getByRole('button', { name: 'Show source' }));
	expect(onSource).toHaveBeenCalledWith([source]);
});
it('drops ambiguous tax sources when a preceding tax is deleted', async () => {
	const onSave = vi.fn();
	const source = {
		part_id: 'part',
		photo_index: 0,
		source_sha256: 'hash',
		coordinate_space: 'unknown' as const
	};
	render(ReceiptEditor, {
		receipt: {
			...receipt,
			field_sources: { 'taxes.0.amount': [source], 'taxes.1.amount': [source], total: [source] }
		},
		lineItems: [],
		taxes: [
			{ id: 'tax-a', receipt_id: receipt.id, name: 'A', rate: null, amount: '1' },
			{ id: 'tax-b', receipt_id: receipt.id, name: 'B', rate: null, amount: '2' }
		],
		onSave
	});
	await fireEvent.click(screen.getAllByRole('button', { name: 'Remove' })[0]);
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(onSave.mock.calls[0][0].taxes[0].name).toBe('B');
	expect(onSave.mock.calls[0][0].field_sources).toEqual({ total: [source] });
});

it.each([
	['preceding deletion', 1, 2],
	['out of range', 0, 9]
] as const)(
	'keeps safe display and evidence round-trip after %s',
	async (kind, first_index, second_index) => {
		const onSave = vi.fn();
		render(ReceiptEditor, {
			receipt: { ...receipt, overlap_decisions: [{ first_index, second_index, verified: true }] },
			lineItems: [line(), { ...line(), id: 'b' }, { ...line(), id: 'c' }],
			taxes: [],
			onSave
		});
		if (kind === 'preceding deletion')
			await fireEvent.click(screen.getAllByRole('button', { name: 'Remove' })[0]);
		else await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Changed' } });
		await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
		expect(onSave.mock.calls[0][0].overlap_decisions).toEqual(
			kind === 'preceding deletion'
				? [{ verified: false, association_unknown: true }]
				: [{ first_index, second_index, verified: true }]
		);
		expect(
			(
				screen.getByRole('checkbox', {
					name: 'I reviewed this overlap; keep the current rows'
				}) as HTMLInputElement
			).disabled
		).toBe(true);
		expect(screen.getByText('Row association unknown; review the original evidence.')).toBeTruthy();
	}
);

it.each([1, 2])(
	'preserves a producer-verified removed-row pair on classification-only save (removed %s)',
	async (removed_index) => {
		const decisions = [
			{ first_index: removed_index - 1, second_index: removed_index, verified: true, removed_index }
		];
		const onSave = vi.fn();
		render(ReceiptEditor, {
			receipt: { ...receipt, overlap_decisions: decisions },
			lineItems: [
				{ ...line(), description: 'A' },
				{ ...line(), id: 'b', description: 'B' }
			],
			taxes: [],
			onSave
		});
		expect(screen.queryByRole('button', { name: 'Save' })).toBeNull();
		expect(screen.queryByText(/^Check rows/)).toBeNull();
		expect(
			(
				screen.getByRole('checkbox', {
					name: 'I reviewed this overlap; keep the current rows'
				}) as HTMLInputElement
			).disabled
		).toBe(true);
		await fireEvent.change(screen.getByLabelText('Category'), { target: { value: 'electronics' } });
		await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
		expect(onSave.mock.calls[0][0]).toMatchObject({
			category: 'electronics',
			expected_operation_revision: receipt.operation_revision,
			amount_policy: receipt.amount_policy,
			field_sources: receipt.field_sources,
			overlap_decisions: decisions
		});
	}
);

it('normalizes a read DTO time with seconds to the minute write contract', async () => {
	const onSave = vi.fn();
	render(ReceiptEditor, {
		receipt: { ...receipt, time: '14:30:00' },
		lineItems: [line()],
		taxes: [],
		onSave
	});
	await fireEvent.input(screen.getByLabelText('Merchant'), {
		target: { value: 'Time round trip' }
	});
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(onSave).toHaveBeenCalledOnce();
	expect(onSave.mock.calls[0][0].time).toBe('14:30');
});

it('preserves legacy nonzero seconds for explicit validation rather than truncating them', async () => {
	const onSave = vi.fn();
	render(ReceiptEditor, {
		receipt: { ...receipt, time: '14:30:15' },
		lineItems: [line()],
		taxes: [],
		onSave
	});
	await fireEvent.input(screen.getByLabelText('Merchant'), {
		target: { value: 'Preserve seconds' }
	});
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(onSave.mock.calls[0][0].time).toBe('14:30:15');
});

describe.each(['Save', 'Save & verify'])('%s time precision', (action) => {
	it.each([
		['14:30:00.000000', '14:30', 'time'],
		['14:30:15', '14:30:15', 'text'],
		['14:30:15.123000', '14:30:15.123000', 'text'],
		['14:30:00.000001', '14:30:00.000001', 'text']
	])('displays and emits %s across reload', async (stored, displayed, type) => {
		const onSave = vi.fn();
		const onSaveAndVerify = vi.fn();
		const view = render(ReceiptEditor, {
			receipt: { ...receipt, time: stored },
			lineItems: [line()],
			taxes: [],
			onSave,
			onSaveAndVerify
		});
		expect((screen.getByLabelText('Time') as HTMLInputElement).value).toBe(displayed);
		expect(screen.getByLabelText('Time').getAttribute('type')).toBe(type);
		await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Changed' } });
		await fireEvent.click(screen.getByRole('button', { name: action }));
		const callback = action === 'Save' ? onSave : onSaveAndVerify;
		expect(callback.mock.calls[0][0].time).toBe(displayed);
		await view.rerender({ receipt: { ...receipt, time: stored, operation_revision: 2 } });
		expect((screen.getByLabelText('Time') as HTMLInputElement).value).toBe(displayed);
	});
});

it.each(['re-currency', 're-tax_total'])(
	'requires an explicit completed receipt value in %s before either save action',
	async (inputId) => {
		const onSave = vi.fn();
		const onSaveAndVerify = vi.fn();
		const { container } = render(ReceiptEditor, {
			receipt: { ...receipt, discount_total: '2', tax_total: '1' },
			lineItems: [line()],
			taxes: [],
			onSave,
			onSaveAndVerify
		});
		await fireEvent.input(container.querySelector(`#${inputId}`)!, { target: { value: '' } });
		await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
		await fireEvent.click(screen.getByRole('button', { name: 'Save & verify' }));
		expect(onSave).not.toHaveBeenCalled();
		expect(onSaveAndVerify).not.toHaveBeenCalled();
		await fireEvent.input(container.querySelector(`#${inputId}`)!, {
			target: { value: inputId === 're-currency' ? 'chf' : '0' }
		});
		await fireEvent.click(screen.getByRole('button', { name: 'Save & verify' }));
		expect(onSaveAndVerify).toHaveBeenCalledOnce();
		expect(onSaveAndVerify.mock.calls[0][0]).toMatchObject({
			currency: inputId === 're-currency' ? 'CHF' : 'EUR',
			discount_total: '2',
			tax_total: inputId === 're-tax_total' ? '0' : '1',
			tip: null
		});
	}
);

it('shows stored null observations as blank and preserves them on save', async () => {
	const onSave = vi.fn();
	render(ReceiptEditor, {
		receipt: { ...receipt, discount_total: null },
		lineItems: [{ ...line(), quantity: null, unit_price: null }],
		taxes: [],
		onSave
	});
	for (const label of ['Discount', 'Tip', 'Qty', 'Unit price'])
		expect((screen.getByLabelText(label) as HTMLInputElement).value).toBe('');
	await fireEvent.input(screen.getByLabelText('Merchant'), { target: { value: 'Changed' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(onSave).toHaveBeenCalledWith(
		expect.objectContaining({
			discount_total: null,
			tip: null,
			line_items: [
				expect.objectContaining({ quantity: null, unit_price: null, total_price: '2.00' })
			]
		})
	);
});

it('allows clearing optional observations when saving and verifying', async () => {
	const onSaveAndVerify = vi.fn();
	render(ReceiptEditor, {
		receipt: { ...receipt, discount_total: '0', tip: '0' },
		lineItems: [line()],
		taxes: [],
		onSaveAndVerify
	});
	for (const label of ['Discount', 'Tip', 'Qty', 'Unit price'])
		await fireEvent.input(screen.getByLabelText(label), { target: { value: '' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save & verify' }));
	expect(onSaveAndVerify).toHaveBeenCalledWith(
		expect.objectContaining({
			discount_total: null,
			tip: null,
			line_items: [expect.objectContaining({ quantity: null, unit_price: null })]
		})
	);
});

it('adds a line without assuming quantity or unit price', async () => {
	const onSave = vi.fn();
	const { container } = render(ReceiptEditor, { receipt, lineItems: [], taxes: [], onSave });
	await fireEvent.click(screen.getByRole('button', { name: 'Add item' }));
	await fireEvent.input(screen.getByLabelText('Description'), { target: { value: 'Bread' } });
	await fireEvent.input(container.querySelector('#li-0-total')!, { target: { value: '2' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(onSave).toHaveBeenCalledWith(
		expect.objectContaining({
			line_items: [
				expect.objectContaining({
					description: 'Bread',
					quantity: null,
					unit_price: null,
					total_price: '2'
				})
			]
		})
	);
});

it.each([
	['Discount', '-1'],
	['Qty', '0'],
	['Unit price', 'invalid'],
	['Description', ' ']
])('rejects an invalid %s before saving', async (label, value) => {
	const onSave = vi.fn();
	render(ReceiptEditor, { receipt, lineItems: [line()], taxes: [], onSave });
	await fireEvent.input(screen.getByLabelText(label), { target: { value } });
	expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(true);
});

it('submits the exact entered quantity and rate', async () => {
	const onSave = vi.fn();
	const { container } = render(ReceiptEditor, {
		receipt,
		lineItems: [line()],
		taxes: [{ id: 'vat', receipt_id: receipt.id, name: 'VAT', rate: null, amount: '0' }],
		onSave
	});
	await fireEvent.input(screen.getByLabelText('Qty'), { target: { value: '1.00000000000000001' } });
	await fireEvent.input(container.querySelector('#tax-0-rate')!, {
		target: { value: '0.20000000000000001' }
	});
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(onSave.mock.calls[0][0]).toMatchObject({
		line_items: [expect.objectContaining({ quantity: '1.00000000000000001' })],
		taxes: [expect.objectContaining({ rate: '0.20000000000000001' })]
	});
});

it('shows a legacy printed subtotal and raw rate until explicitly corrected', async () => {
	const onSave = vi.fn();
	render(ReceiptEditor, {
		receipt: {
			...receipt,
			amount_policy: { ...receipt.amount_policy, tax_basis: 'gross', subtotal_origin: 'printed' }
		},
		lineItems: [line()],
		taxes: [{ id: 'vat', receipt_id: receipt.id, name: 'VAT', rate: '19.0', amount: '0' }],
		onSave
	});
	const rate = screen.getByLabelText('Tax rate (fraction)') as HTMLInputElement;
	const subtotal = screen.getByRole('combobox', { name: /Subtotal origin/ }) as HTMLSelectElement;
	expect(rate.value).toBe('19.0');
	expect(rate.getAttribute('aria-describedby')).toBe(screen.getByText(/Enter 0.19 for 19%/).id);
	expect(subtotal.value).toBe('printed');
	await fireEvent.change(subtotal, { target: { value: 'printed_net' } });
	expect(rate.value).toBe('19.0');
	await fireEvent.input(rate, { target: { value: '0.190' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(onSave).toHaveBeenCalled();
	expect(onSave.mock.calls[0][0]).toMatchObject({
		amount_policy: { tax_basis: 'gross', subtotal_origin: 'printed_net' },
		taxes: [{ name: 'VAT', amount: '0', rate: '0.190' }]
	});
});

it('selects a suggested display name independently of the printed name and switches locale with populated categories and tags', async () => {
	const lookup = vi
		.spyOn(api, 'listMerchants')
		.mockResolvedValue({ merchants: ['REWE'], suggestion: 'REWE' });
	const onSave = vi.fn();
	const item = { ...line('milk', 'llm'), tags: ['apples', 'Hausmarke'] };
	render(ReceiptEditor, {
		receipt: { ...receipt, merchant_name: 'REWE CITY 123' },
		lineItems: [item],
		taxes: [],
		onSave
	});
	try {
		await fireEvent.click(await screen.findByRole('button', { name: 'Use “REWE”' }));
		expect(lookup).toHaveBeenCalledWith(
			expect.objectContaining({ merchant_name: 'REWE CITY 123', receipt_id: receipt.id }),
			expect.objectContaining({ userId: 'viewer' })
		);
		setLocale('de');
		const display = await screen.findByLabelText('Anzeigename des Händlers');
		expect((display as HTMLInputElement).value).toBe('REWE');
		expect((screen.getByLabelText('Händler') as HTMLInputElement).value).toBe('REWE CITY 123');
		expect(screen.getByRole('option', { name: 'Milch', selected: true })).toBeTruthy();
		expect(screen.getByRole('option', { name: 'Bar', selected: true })).toBeTruthy();
		expect(screen.getByText('Äpfel').closest('details')?.open).toBe(true);
		expect(screen.getByText('Hausmarke')).toBeTruthy();
		await fireEvent.input(display, { target: { value: 'REWE Nord' } });
		await fireEvent.click(screen.getByRole('button', { name: 'Speichern' }));
		expect(onSave.mock.calls[0][0]).toMatchObject({
			merchant_name: 'REWE CITY 123',
			merchant_display_name: 'REWE Nord'
		});
	} finally {
		setLocale('en');
		lookup.mockRestore();
	}
});

it('discards a merchant suggestion that arrives after the rendered account changes', async () => {
	const pending = deferred<{ suggestion: string | null; merchants: string[] }>();
	const lookup = vi.spyOn(api, 'listMerchants').mockReturnValue(pending.promise);
	try {
		render(ReceiptEditor, { receipt, lineItems: [], taxes: [] });
		await waitFor(() => expect(lookup).toHaveBeenCalled());
		session.set({ id: 'another-user', username: 'bob', is_admin: false, can_see_all: false });
		pending.resolve({ suggestion: 'Private Shop', merchants: ['Private Shop'] });
		await pending.promise;
		expect(screen.queryByRole('button', { name: 'Use “Private Shop”' })).toBeNull();
		expect(document.querySelector('datalist option[value="Private Shop"]')).toBeNull();
	} finally {
		lookup.mockRestore();
	}
});

it('looks up category history for this receipt and branch and forgets a stale display name after address changes', async () => {
	const onSave = vi.fn();
	const pending = deferred<{ category: string; source: 'history' }>();
	const lookup = vi.spyOn(api, 'getCategorySuggestion').mockReturnValue(pending.promise);
	try {
		render(ReceiptEditor, {
			receipt: { ...receipt, merchant_display_name: 'Kiosk Nord', merchant_address: 'Berlin' },
			lineItems: [line()],
			taxes: [],
			onSave
		});
		await fireEvent.blur(screen.getByLabelText('Description'));
		expect(lookup).toHaveBeenCalledWith({
			receipt_id: receipt.id,
			merchant_name: 'Shop A',
			merchant_display_name: 'Kiosk Nord',
			merchant_address: 'Berlin',
			description: 'MUESLI'
		});
		await fireEvent.input(screen.getByLabelText('Address'), { target: { value: 'Bonn' } });
		expect((screen.getByLabelText('Merchant display name') as HTMLInputElement).value).toBe('');
		pending.resolve({ category: 'milk', source: 'history' });
		await pending.promise;
		await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
		expect(onSave.mock.calls[0][0]).not.toHaveProperty('merchant_display_name');
		expect(onSave.mock.calls[0][0].line_items[0].spending_category).toBe('unknown');
	} finally {
		lookup.mockRestore();
	}
});
