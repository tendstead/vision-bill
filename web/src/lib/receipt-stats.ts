import type { Category, ReceiptRow, WeeklyStatistics } from '$lib/types';

export interface CurrencyTotal {
	currency: string;
	total: number;
}

export interface CategoryTotal extends CurrencyTotal {
	category: Category;
	count: number;
}

export interface WeeklyPoint {
	week_start: string;
	total: number;
	receipt_count: number;
}

function amount(value: string): number {
	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : 0;
}

export function verifiedReceipts(receipts: ReceiptRow[]): ReceiptRow[] {
	return receipts.filter((receipt) => receipt.status === 'verified');
}

export function totalsByCurrency(receipts: ReceiptRow[]): CurrencyTotal[] {
	const totals = new Map<string, number>();
	for (const receipt of receipts) {
		totals.set(receipt.currency, (totals.get(receipt.currency) ?? 0) + amount(receipt.total));
	}
	return [...totals.entries()]
		.map(([currency, total]) => ({ currency, total }))
		.sort((a, b) => b.total - a.total);
}

export function totalsByCategory(receipts: ReceiptRow[]): CategoryTotal[] {
	const totals = new Map<string, CategoryTotal>();
	for (const receipt of receipts) {
		const key = `${receipt.category}:${receipt.currency}`;
		const current = totals.get(key) ?? {
			category: receipt.category,
			currency: receipt.currency,
			total: 0,
			count: 0
		};
		current.total += amount(receipt.total);
		current.count += 1;
		totals.set(key, current);
	}
	return [...totals.values()].sort((a, b) => b.total - a.total);
}

export function currentMonthReceipts(receipts: ReceiptRow[], today = new Date()): ReceiptRow[] {
	const prefix = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}`;
	return receipts.filter((receipt) => receipt.date.startsWith(prefix));
}

export function weeklySeries(
	rows: WeeklyStatistics[],
	currency: string,
	today = new Date(),
	points = 12
): WeeklyPoint[] {
	const start = new Date(Date.UTC(today.getFullYear(), today.getMonth(), today.getDate()));
	start.setUTCDate(start.getUTCDate() - ((start.getUTCDay() + 6) % 7));
	const byWeek = new Map(
		rows.filter((row) => row.currency === currency).map((row) => [row.week_start, row] as const)
	);

	return Array.from({ length: points }, (_, index) => {
		const week = new Date(start);
		week.setUTCDate(week.getUTCDate() - (points - index - 1) * 7);
		const weekStart = week.toISOString().slice(0, 10);
		const row = byWeek.get(weekStart);
		return {
			week_start: weekStart,
			total: row ? amount(row.total) : 0,
			receipt_count: row?.receipt_count ?? 0
		};
	});
}
