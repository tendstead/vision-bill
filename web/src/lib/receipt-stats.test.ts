import { describe, expect, it } from 'vitest';
import {
	currentMonthReceipts,
	totalsByCategory,
	totalsByCurrency,
	verifiedReceipts
} from './receipt-stats';
import type { ReceiptRow } from './types';

function receipt(overrides: Partial<ReceiptRow>): ReceiptRow {
	return {
		id: '00000000-0000-4000-8000-000000000001',
		confidence: 100,
		merchant_name: 'Shop',
		merchant_address: null,
		receipt_number: null,
		date: '2026-08-01',
		time: null,
		currency: 'EUR',
		category: 'grocery',
		subtotal: '0',
		discount_total: '0',
		tax_total: '0',
		tip: null,
		total: '10',
		payment_method: 'unknown',
		created_at: null,
		status: 'verified',
		image_id: null,
		thumbnail_path: null,
		verified: true,
		user_id: null,
		...overrides
	};
}

describe('receipt statistics', () => {
	it('uses only verified receipts and keeps currencies separate', () => {
		const receipts = [
			receipt({ id: '00000000-0000-4000-8000-000000000001', total: '10', currency: 'EUR' }),
			receipt({
				id: '00000000-0000-4000-8000-000000000002',
				total: '5.50',
				currency: 'EUR',
				category: 'restaurant'
			}),
			receipt({ id: '00000000-0000-4000-8000-000000000003', total: '20', currency: 'USD' }),
			receipt({
				id: '00000000-0000-4000-8000-000000000004',
				status: 'unverified',
				verified: false,
				total: '999'
			})
		];

		const verified = verifiedReceipts(receipts);
		expect(verified).toHaveLength(3);
		expect(totalsByCurrency(verified)).toEqual([
			{ currency: 'USD', total: 20 },
			{ currency: 'EUR', total: 15.5 }
		]);
		expect(totalsByCategory(verified)).toEqual([
			{ category: 'grocery', currency: 'USD', total: 20, count: 1 },
			{ category: 'grocery', currency: 'EUR', total: 10, count: 1 },
			{ category: 'restaurant', currency: 'EUR', total: 5.5, count: 1 }
		]);
	});

	it('selects receipts from the current month', () => {
		const receipts = [
			receipt({ date: '2026-08-31' }),
			receipt({ id: '00000000-0000-4000-8000-000000000002', date: '2026-07-31' })
		];
		expect(currentMonthReceipts(receipts, new Date('2026-08-31T12:00:00'))).toEqual([receipts[0]]);
	});
});
