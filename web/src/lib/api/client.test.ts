import { afterEach, describe, expect, it, vi } from 'vitest';

import { api } from './client';
import { queryKeys } from '../query/keys';

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
});

describe('api.searchProducts', () => {
	it('GETs /search with the url-encoded query and returns the parsed body', async () => {
		mockOkFetch();

		const result = await api.searchProducts('Schinken');

		const fetchMock = vi.mocked(fetch);
		expect(fetchMock).toHaveBeenCalledOnce();
		const [url, init] = fetchMock.mock.calls[0];
		expect(String(url)).toContain('/search?query=Schinken');
		expect(init).toEqual({ credentials: 'same-origin' });
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
		expect(init).toEqual({ credentials: 'same-origin', method: 'POST' });
		expect(result).toEqual(COLLECTION_BODY);
	});

	it('deactivateCollection POSTs to /collections/{id}/deactivate', async () => {
		mockCollectionFetch();

		await api.deactivateCollection('coll-1');

		const [url, init] = vi.mocked(fetch).mock.calls[0];
		expect(String(url)).toContain('/collections/coll-1/deactivate');
		expect(init).toEqual({ credentials: 'same-origin', method: 'POST' });
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
		expect(init).toEqual({ credentials: 'same-origin' });
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
		expect(api.imageThumbUrl('abc-123')).toBe('/api/v1/images/abc-123/thumb');
	});
});
