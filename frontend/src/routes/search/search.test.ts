import { cleanup, fireEvent, render, screen, within } from '@testing-library/svelte';
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

it('displays purchases without invented quantities, unit prices or price summaries', async () => {
	vi.spyOn(api, 'searchProducts').mockResolvedValue({
		query: 'bread',
		purchases: [
			{
				receipt_id: 'receipt',
				description: 'Wholegrain bread',
				merchant_name: 'Bakery',
				date: '2026-01-01',
				time: null,
				quantity: null,
				unit_price: null,
				currency: 'EUR'
			}
		],
		latest_price: null,
		cheapest_price: null,
		average_price: null,
		currency: null
	});
	render(Page);
	await fireEvent.input(screen.getByRole('searchbox'), { target: { value: 'bread' } });
	const purchase = await screen.findByRole('listitem');
	expect(within(purchase).getAllByText('—')).toHaveLength(2);
	expect(screen.getByRole('status').textContent).toContain('1 purchase');
	expect(screen.queryByText('Latest')).toBeNull();
});
