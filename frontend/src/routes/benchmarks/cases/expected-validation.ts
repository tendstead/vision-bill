import type { ExtractionDraft } from '$lib/types';
import { isSignedDecimal, isPlainDecimal, validReviewDecimal } from '$lib/ui/money';

export function expectedIssue(expected: ExtractionDraft, confirm: boolean): string | null {
	const hasObservation = Object.entries(expected).some(
		([key, value]) =>
			key !== 'category' &&
			key !== 'confidence' &&
			value != null &&
			(Array.isArray(value) || String(value).trim() !== '')
	);
	if (confirm && !hasObservation) return 'benchmarkCase.missing';
	for (const row of expected.line_items ?? []) {
		if (row.tags?.some((tag) => Array.from(tag.trim()).length > 100))
			return 'benchmarkCase.longTag';
		if (
			confirm &&
			!row.description?.trim() &&
			row.quantity == null &&
			row.unit_price == null &&
			row.total_price == null
		)
			return 'benchmarkCase.emptyRow';
		if (row.quantity != null && !validReviewDecimal(row.quantity, 'positive'))
			return 'editor.invalidQuantity';
		if (
			[row.total_price, row.unit_price].some((amount) => amount != null && !isSignedDecimal(amount))
		)
			return 'editor.invalidNumber';
	}
	for (const row of expected.taxes ?? []) {
		if (confirm && !row.name?.trim() && row.rate == null && row.amount == null)
			return 'benchmarkCase.emptyRow';
		if (row.rate != null && !validReviewDecimal(row.rate, confirm ? 'rate' : 'any'))
			return 'editor.invalidRate';
		if (row.amount != null && !isPlainDecimal(row.amount)) return 'editor.invalidAmount';
	}
	return null;
}
