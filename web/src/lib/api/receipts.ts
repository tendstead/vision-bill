import { api } from '$lib/api/client';
import type { ReceiptListFilters, ReceiptRow } from '$lib/types';

const PAGE_SIZE = 500;

/** Fetch every receipt matching the supplied filters, following the API's offset pagination. */
export async function fetchAllReceipts(
	filters: Omit<ReceiptListFilters, 'limit' | 'offset'> = {}
): Promise<ReceiptRow[]> {
	const receipts: ReceiptRow[] = [];
	let offset = 0;

	while (true) {
		const page = await api.listReceipts({ ...filters, limit: PAGE_SIZE, offset });
		receipts.push(...page);
		if (page.length < PAGE_SIZE) {
			return receipts;
		}
		offset += PAGE_SIZE;
	}
}
