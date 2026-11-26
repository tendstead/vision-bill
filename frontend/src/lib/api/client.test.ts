import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { api } from './client';
import { fetchAllReceipts } from './receipts';
import { queryKeys } from '../query/keys';
import { advanceSessionGeneration, captureSessionActor, session } from '../session-lifetime';

const USER_ID = '00000000-0000-4000-8000-000000000001';
beforeEach(() =>
	session.set({ id: USER_ID, username: 'tester', is_admin: false, can_see_all: false })
);

const OK_BODY = {
	query: 'Schinken',
	purchases: [
		{
			receipt_id: '6f1e0c1a-0000-4000-8000-000000000001',
			description: 'Schinken natur',
			merchant_name: 'REWE',
			date: '2026-08-30',
			time: '18:42',
			quantity: 1.0,
			unit_price: '4.99',
			currency: 'EUR'
		}
	],
	latest_price: '4.99',
	cheapest_price: '4.29',
	average_price: '4.64',
	currency: 'EUR'
};

function mockOkFetch(): void {
	vi.stubGlobal(
		'fetch',
		vi.fn().mockResolvedValue({
			ok: true,
			status: 200,
			statusText: 'OK',
			json: async () => OK_BODY
		})
	);
}

afterEach(() => {
	vi.unstubAllGlobals();
	session.set(null);
});

describe('api.searchProducts', () => {
	it('GETs /search with the url-encoded query and returns the parsed body', async () => {
		mockOkFetch();

		const result = await api.searchProducts('Schinken');

		const fetchMock = vi.mocked(fetch);
		expect(fetchMock).toHaveBeenCalledOnce();
		const [url, init] = fetchMock.mock.calls[0];
		expect(String(url)).toContain('/search?query=Schinken');
		expect(init).toMatchObject({ credentials: 'same-origin', cache: 'no-store' });
		expect(new Headers(init?.headers).get('X-Vision-Bill-Expected-User')).toBe(USER_ID);
		expect(result).toEqual(OK_BODY);
	});

	it('url-encodes special characters in the term', async () => {
		mockOkFetch();

		await api.searchProducts('a b&c');

		const [url] = vi.mocked(fetch).mock.calls[0];
		expect(String(url)).toContain('/search?query=a%20b%26c');
	});
});

describe('queryKeys.search', () => {
	it('builds a per-term key', () => {
		expect(queryKeys.search('Schinken')).toEqual(['search', 'Schinken']);
		expect(queryKeys.search('a')).not.toEqual(queryKeys.search('b'));
	});
});

const COLLECTION_BODY = {
	id: '6f1e0c1a-0000-4000-8000-000000000001',
	name: 'Berlin',
	color: '#3B82F6',
	start_date: null,
	end_date: null,
	active: true,
	created_at: null
};

function mockCollectionFetch(): void {
	vi.stubGlobal(
		'fetch',
		vi.fn().mockResolvedValue({
			ok: true,
			status: 200,
			statusText: 'OK',
			json: async () => COLLECTION_BODY
		})
	);
}

describe('api collection capture', () => {
	it('activateCollection POSTs to /collections/{id}/activate', async () => {
		mockCollectionFetch();

		const result = await api.activateCollection('coll-1');

		const [url, init] = vi.mocked(fetch).mock.calls[0];
		expect(String(url)).toContain('/collections/coll-1/activate');
		expect(init).toMatchObject({ credentials: 'same-origin', method: 'POST' });
		expect(new Headers(init?.headers).get('X-Vision-Bill-Expected-User')).toBe(USER_ID);
		expect(result).toEqual(COLLECTION_BODY);
	});

	it('deactivateCollection POSTs to /collections/{id}/deactivate', async () => {
		mockCollectionFetch();

		await api.deactivateCollection('coll-1');

		const [url, init] = vi.mocked(fetch).mock.calls[0];
		expect(String(url)).toContain('/collections/coll-1/deactivate');
		expect(init).toMatchObject({ credentials: 'same-origin', method: 'POST' });
		expect(new Headers(init?.headers).get('X-Vision-Bill-Expected-User')).toBe(USER_ID);
	});
});

const STATS_BODY = {
	verified_receipt_count: 0,
	currencies: [],
	merchants: [],
	categories: [],
	payment_methods: [],
	weekdays: [],
	weekly_spending: []
};

function mockStatsFetch(): void {
	vi.stubGlobal(
		'fetch',
		vi.fn().mockResolvedValue({
			ok: true,
			status: 200,
			statusText: 'OK',
			json: async () => STATS_BODY
		})
	);
}

describe('api.getStatistics', () => {
	it('GETs /statistics?weeks=12 with no collection filter', async () => {
		mockStatsFetch();

		await api.getStatistics(12);

		const [url, init] = vi.mocked(fetch).mock.calls[0];
		expect(String(url)).toContain('/statistics?weeks=12');
		expect(String(url)).not.toContain('collection_id');
		expect(init).toMatchObject({ credentials: 'same-origin', cache: 'no-store' });
		expect(new Headers(init?.headers).get('X-Vision-Bill-Expected-User')).toBe(USER_ID);
	});

	it('appends collection_id when a collection is selected', async () => {
		mockStatsFetch();

		await api.getStatistics(12, 'coll-1');

		const [url] = vi.mocked(fetch).mock.calls[0];
		expect(String(url)).toContain('/statistics?weeks=12&collection_id=coll-1');
	});
});

describe('api image URL helpers', () => {
	it('builds the image thumbnail URL', () => {
		expect(api.imageThumbUrl('abc-123')).toBe(
			`/api/v1/images/abc-123/thumb?vb_expected_user=${USER_ID}`
		);
		expect(api.imageFileUrl('abc-123', 1)).toBe(
			`/api/v1/images/abc-123/file?part=1&vb_expected_user=${USER_ID}`
		);
		expect(api.sourceViewUrl('/api/v1/images/abc-123/views/working')).toBe(
			`/api/v1/images/abc-123/views/working?vb_expected_user=${USER_ID}`
		);
	});
});

describe('durable submissions', () => {
	it('keeps ordered multipart files and explicit option presence with an idempotency key', async () => {
		vi.stubGlobal(
			'fetch',
			vi.fn().mockResolvedValue({
				ok: true,
				status: 202,
				json: async () => ({ id: 'job', terminal: false, status: 'pending' })
			})
		);
		const files = [new File(['second'], 'b.png'), new File(['first'], 'a.png')];
		expect(await api.uploadImages(files, {}, 'stable-key')).toMatchObject({
			id: 'job',
			terminal: false
		});
		const [url, init] = vi.mocked(fetch).mock.calls[0];
		expect(url).toBe('/api/v1/images');
		expect(new Headers(init?.headers).get('Idempotency-Key')).toBe('stable-key');
		expect(new Headers(init?.headers).get('X-Vision-Bill-Expected-User')).toBe(USER_ID);
		expect((init?.body as FormData).getAll('receipt')).toEqual(files);
		await api.uploadImages(files, { bypass_review: false, model_id: 'text model' }, 'stable-key');
		expect(vi.mocked(fetch).mock.calls[1][0]).toBe(
			'/api/v1/images?model_id=text%20model&bypass_review=false'
		);
	});
	it('sends generation and request key for retries and preserves conflict details', async () => {
		vi.stubGlobal(
			'fetch',
			vi.fn().mockResolvedValue({
				ok: false,
				status: 409,
				json: async () => ({ detail: { reason: 'stale generation', generation: 2 } })
			})
		);
		await expect(
			api.retrySubmission('job', { expected_generation: 1, request_key: 'retry-key' })
		).rejects.toMatchObject({
			status: 409,
			detail: '{"reason":"stale generation","generation":2}'
		});
		expect(JSON.parse(vi.mocked(fetch).mock.calls[0][1]!.body as string)).toEqual({
			expected_generation: 1,
			request_key: 'retry-key'
		});
	});
});

it('preserves explicitly empty model selection instead of changing option presence', async () => {
	mockOkFetch();
	await api.uploadImages([new File(['image'], 'a.png')], { model_id: '' }, 'key');
	expect(vi.mocked(fetch).mock.calls[0][0]).toBe('/api/v1/images?model_id=');
});

describe('versioned drafts and review', () => {
	it('sends exact changes, revisions, finalize key and acknowledgements', async () => {
		mockOkFetch();
		await api.getDraft('job');
		await api.patchDraft('job', {
			expected_revision: 7,
			changes: { total: null, subtotal: '0.0000000001' }
		});
		await api.finalizeDraft('job', { expected_revision: 8, request_key: 'stable' });
		await api.verifyReceipt('receipt', {
			expected_operation_revision: 4,
			expected_review_revision: 3,
			acknowledgements: [{ finding_id: 'f', comment: 'Checked original' }]
		});
		const calls = vi.mocked(fetch).mock.calls;
		expect(calls[0][0]).toBe('/api/v1/images/job/draft');
		expect(calls[1][1]?.method).toBe('PATCH');
		expect(JSON.parse(calls[1][1]?.body as string)).toEqual({
			expected_revision: 7,
			changes: { total: null, subtotal: '0.0000000001' }
		});
		expect(calls[2][0]).toBe('/api/v1/images/job/draft/finalize');
		expect(JSON.parse(calls[2][1]?.body as string)).toEqual({
			expected_revision: 8,
			request_key: 'stable'
		});
		expect(JSON.parse(calls[3][1]?.body as string).acknowledgements).toEqual([
			{ finding_id: 'f', comment: 'Checked original' }
		]);
	});
});

it('never fetches a protected endpoint or produces an unbound media URL without an actor', async () => {
	mockOkFetch();
	const stale = captureSessionActor();
	session.set(null);
	await expect(api.getImage('private')).rejects.toMatchObject({ name: 'AbortError' });
	expect(fetch).not.toHaveBeenCalled();
	expect(api.imageThumbUrl('private')).toBeUndefined();
	expect(api.imageFileUrl('private', 0, stale)).toBeUndefined();
});

it('does not start another receipt page after the originating session changes', async () => {
	const list = vi
		.spyOn(api, 'listReceipts')
		.mockImplementationOnce(async () => {
			queueMicrotask(advanceSessionGeneration);
			return Array.from({ length: 500 }, () => ({ id: 'old' }) as never);
		})
		.mockResolvedValueOnce([]);
	try {
		await expect(fetchAllReceipts()).rejects.toMatchObject({ name: 'AbortError' });
		expect(list).toHaveBeenCalledOnce();
	} finally {
		list.mockRestore();
	}
});

it('exposes draft validation codes and field paths to the editor', async () => {
	vi.stubGlobal(
		'fetch',
		vi
			.fn()
			.mockResolvedValue(
				new Response(
					JSON.stringify({ detail: { code: 'draft_invalid_fields', fields: ['payment_method'] } }),
					{ status: 422 }
				)
			)
	);
	await expect(
		api.patchDraft('job', { expected_revision: 7, changes: { confidence: 100 } })
	).rejects.toMatchObject({
		status: 422,
		problem: { code: 'draft_invalid_fields', fields: ['payment_method'] }
	});
});

it('sends merchant evidence in the suggestion request body and binds it to the active account', async () => {
	mockOkFetch();
	const evidence = {
		receipt_id: 'receipt-id',
		merchant_name: 'Local Grocer',
		merchant_address: 'Sample Street 3'
	};
	await api.listMerchants(evidence);
	const [url, init] = vi.mocked(fetch).mock.calls[0];
	expect(url).toBe('/api/v1/merchants/suggestions');
	expect(init?.method).toBe('POST');
	expect(JSON.parse(init?.body as string)).toEqual(evidence);
	expect(new Headers(init?.headers).get('X-Vision-Bill-Expected-User')).toBe(USER_ID);
});
