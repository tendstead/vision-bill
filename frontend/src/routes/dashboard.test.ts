import { cleanup, render, screen } from '@testing-library/svelte';
import { afterEach, expect, it, vi } from 'vitest';
import { api } from '$lib/api/client';
import { queryClient } from '$lib/query/client';
import Page from './+page.svelte';

vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));

afterEach(() => {
	cleanup();
	queryClient.clear();
	vi.restoreAllMocks();
});

it('links observed unverified receipts and exposes the weekly values as a table', async () => {
	vi.spyOn(api, 'listReceipts').mockResolvedValue([
		{
			id: 'review',
			merchant_name: 'Market',
			date: '2026-11-19',
			total: '12.34',
			currency: 'EUR',
			status: 'unverified'
		}
	] as never);
	vi.spyOn(api, 'listImages').mockResolvedValue([
		{ id: 'draft', original_filename: 'Open draft', draft_available: true, receipt_id: null }
	] as never);
	vi.spyOn(api, 'getActiveCollection').mockResolvedValue(null);
	vi.spyOn(api, 'listCollections').mockResolvedValue([]);
	vi.spyOn(api, 'getStatistics').mockResolvedValue({
		verified_receipt_count: 1,
		currencies: [{ currency: 'EUR', total: '3.86' }],
		weekly_spending: [],
		merchants: []
	} as never);
	render(Page);
	const review = await screen.findByRole('link', { name: /Market/ });
	expect(review.getAttribute('href')).toBe('/receipts/review');
	expect(screen.getByRole('link', { name: /Open draft/ }).getAttribute('href')).toBe(
		'/upload?job=draft&draft=1'
	);
	const table = await screen.findByRole('table', { name: 'Weekly spending · EUR' });
	expect(table.querySelectorAll('tbody tr')).toHaveLength(12);
	expect(screen.getByRole('columnheader', { name: 'Total' })).toBeTruthy();
});

it.each(['pending', 'failed'])(
	'keeps receipts visible while draft loading is %s',
	async (state) => {
		vi.spyOn(api, 'listReceipts').mockResolvedValue([
			{
				id: 'ready',
				merchant_name: 'Available Market',
				date: '2026-11-19',
				total: '1.23',
				currency: 'EUR'
			}
		] as never);
		vi.spyOn(api, 'listImages').mockImplementation(() =>
			state === 'pending' ? new Promise(() => {}) : Promise.reject(new Error('Unavailable'))
		);
		vi.spyOn(api, 'getActiveCollection').mockResolvedValue(null);
		vi.spyOn(api, 'listCollections').mockResolvedValue([]);
		vi.spyOn(api, 'getStatistics').mockResolvedValue({ verified_receipt_count: 0 } as never);
		render(Page);
		expect(
			(await screen.findByRole('link', { name: /Available Market/ })).getAttribute('href')
		).toBe('/receipts/ready');
	}
);

it.each(['pending', 'failed'])(
	'keeps drafts visible while receipt loading is %s',
	async (state) => {
		vi.spyOn(api, 'listReceipts').mockImplementation(() =>
			state === 'pending' ? new Promise(() => {}) : Promise.reject(new Error('Unavailable'))
		);
		vi.spyOn(api, 'listImages').mockResolvedValue([
			{
				id: 'ready-draft',
				original_filename: 'Available draft',
				draft_available: true,
				receipt_id: null
			}
		] as never);
		vi.spyOn(api, 'getActiveCollection').mockResolvedValue(null);
		vi.spyOn(api, 'listCollections').mockResolvedValue([]);
		vi.spyOn(api, 'getStatistics').mockResolvedValue({ verified_receipt_count: 0 } as never);
		render(Page);
		expect(
			(await screen.findByRole('link', { name: /Available draft/ })).getAttribute('href')
		).toBe('/upload?job=ready-draft&draft=1');
	}
);
