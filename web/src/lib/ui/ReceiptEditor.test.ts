import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LineItemRow, ReceiptRow, ReceiptWrite } from '$lib/types';
import { queryClient } from '$lib/query/client';
import ReceiptEditor from './ReceiptEditor.svelte';

const receipt: ReceiptRow = {
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
