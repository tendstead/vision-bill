import { describe, expect, it } from 'vitest';
import { filterCollections, normalizeQuery } from './collection-picker-logic';
import type { CollectionSummary } from './types';

const c = (id: string, name: string): CollectionSummary => ({
	id,
	name,
	color: null,
	start_date: null,
	end_date: null,
	active: false,
	created_at: null,
	receipt_count: 0,
	totals: []
});

describe('collection picker logic', () => {
	it('normalizeQuery trims, lowercases and collapses whitespace', () => {
		expect(normalizeQuery('  Berlin   Trip ')).toBe('berlin trip');
	});
	it('filterCollections is a case-insensitive substring match', () => {
		const list = [c('1', 'Alice Wedding'), c('2', 'Berlin')];
		expect(filterCollections(list, 'berlin').map((x) => x.id)).toEqual(['2']);
		// Empty query keeps every option (already in name order here).
		expect(filterCollections(list, '').map((x) => x.id)).toEqual(['1', '2']);
	});
	it('filterCollections sorts alphabetically with selected options pinned first', () => {
		const list = [c('1', 'Berlin'), c('2', 'Alice Wedding')];
		expect(filterCollections(list, '').map((x) => x.id)).toEqual(['2', '1']);
		// Selected 'Berlin' (1) is pinned ahead of the alphabetical 'Alice Wedding' (2).
		expect(filterCollections(list, '', ['1']).map((x) => x.id)).toEqual(['1', '2']);
	});
	it('filterCollections keeps selected options even when not matching', () => {
		const list = [c('1', 'Berlin'), c('2', 'Christmas')];
		// selected 'Christmas' stays visible while searching 'berlin'
		const rows = filterCollections(list, 'berlin', ['2']);
		expect(rows.map((x) => x.id)).toContain('1');
		expect(rows.map((x) => x.id)).toContain('2');
	});
});
