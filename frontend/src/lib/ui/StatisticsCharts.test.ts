import { cleanup, render, screen } from '@testing-library/svelte';
import { afterEach, expect, it, vi } from 'vitest';
import type { ReceiptStatistics } from '$lib/types';
import StatisticsCharts from './StatisticsCharts.svelte';
afterEach(() => {
	cleanup();
	vi.useRealTimers();
});
it('provides dated weekly values and distinguishes all-time aggregates from the weekly period', () => {
	vi.useFakeTimers();
	vi.setSystemTime(new Date('2026-11-19T12:00:00Z'));
	const stats: ReceiptStatistics = {
		verified_receipt_count: 1,
		currencies: [
			{
				currency: 'EUR',
				receipt_count: 1,
				total: '12.34',
				average: '12.34',
				median: '12.34',
				minimum: '12.34',
				maximum: '12.34',
				subtotal: '12.34',
				discounts: '0',
				taxes: '0',
				tips: '0'
			}
		],
		merchants: [],
		categories: [],
		payment_methods: [],
		weekdays: [],
		weekly_spending: [
			{
				week_start: '2026-11-16',
				currency: 'EUR',
				receipt_count: 1,
				total: '12.34',
				average: '12.34'
			}
		]
	};
	render(StatisticsCharts, { stats });
	expect(screen.getByText(/Receipt statistics: all verified/)).not.toBeNull();
	const details = screen.getByText('Show chart data').closest('details')!;
	details.open = true;
	const table = screen.getByRole('table', { name: 'Weekly spending · EUR' });
	expect(table.querySelectorAll('tbody tr')).toHaveLength(12);
	expect(table.querySelector('tbody tr:last-child')?.textContent).toContain('2026-11-16');
	expect(table.querySelector('tbody tr:last-child')?.textContent).toContain('€12.34');
});
