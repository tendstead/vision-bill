import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it } from 'vitest';
import type { ItemSpendingStatistics } from '$lib/types';
import ItemSpendingCharts from './ItemSpendingCharts.svelte';

const stats: ItemSpendingStatistics = {
	months: 3,
	date_from: '2026-07-01',
	date_to: '2026-09-27',
	categories: [
		{ category: 'milk', currency: 'EUR', total: '9.00', share: '90.00' },
		{ category: 'unknown', currency: 'EUR', total: '1.00', share: '10.00' },
		{ category: 'milk', currency: 'USD', total: '-2.00', share: null }
	],
	monthly: [
		{ month: '2026-07-01', category: 'milk', currency: 'EUR', total: '0.00', share: null },
		{ month: '2026-08-01', category: 'milk', currency: 'EUR', total: '4.00', share: '100.00' },
		{ month: '2026-09-01', category: 'milk', currency: 'EUR', total: '5.00', share: '83.33' },
		{ month: '2026-07-01', category: 'unknown', currency: 'EUR', total: '0.00', share: null },
		{ month: '2026-08-01', category: 'unknown', currency: 'EUR', total: '0.00', share: '0.00' },
		{ month: '2026-09-01', category: 'unknown', currency: 'EUR', total: '1.00', share: '16.67' },
		{ month: '2026-07-01', category: 'milk', currency: 'USD', total: '0.00', share: null },
		{ month: '2026-08-01', category: 'milk', currency: 'USD', total: '0.00', share: null },
		{ month: '2026-09-01', category: 'milk', currency: 'USD', total: '-2.00', share: null }
	],
	reconciliation: [
		{
			currency: 'EUR',
			receipt_total: '12.00',
			item_total: '11.00',
			deposit_total: '1.00',
			unallocated_total: '1.00'
		},
		{
			currency: 'USD',
			receipt_total: '-2.00',
			item_total: '-2.00',
			deposit_total: '0.00',
			unallocated_total: '0.00'
		}
	]
};

afterEach(cleanup);

describe('item spending report', () => {
	it('shows zero months, unknown spending, partial current month, and selected category history', async () => {
		render(ItemSpendingCharts, { stats });
		expect(
			screen.getAllByText('Share of item spending (excluding deposits; includes unknown)')
		).toHaveLength(2);
		expect(screen.getByText('10.00%')).not.toBeNull();
		expect(screen.getByRole('table').querySelectorAll('tbody tr')).toHaveLength(3);
		expect(screen.getByText('(partial)')).not.toBeNull();
		await fireEvent.change(screen.getByLabelText('Category to follow'), {
			target: { value: 'unknown' }
		});
		expect(screen.getByRole('table').querySelector('caption')?.textContent).toContain('Unknown');
		expect(screen.getByRole('table').querySelectorAll('tbody tr')).toHaveLength(3);
	});

	it('keeps currencies separate and marks nonpositive shares as unavailable', async () => {
		render(ItemSpendingCharts, { stats });
		await fireEvent.change(screen.getByLabelText('Currency'), { target: { value: 'USD' } });
		expect(screen.getByText('No positive spending total')).not.toBeNull();
		expect(screen.getByRole('table').textContent).toContain('-$2.00');
		expect(screen.getByRole('table').querySelectorAll('tbody tr')).toHaveLength(3);
	});
});
