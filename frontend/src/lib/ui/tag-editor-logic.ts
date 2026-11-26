/**
 * Pure dropdown logic for the TagEditor chip-input combobox.
 * Kept free of Svelte so it can be unit-tested directly.
 */

import { productTagLabel, productTagValue } from '$lib/i18n/product-tag-label';

export type TagRow =
	| { kind: 'tag'; tag: string; selected: boolean; optionIndex: number }
	| { kind: 'create'; tag: string; optionIndex: number }
	| { kind: 'nomatch'; query: string }
	| { kind: 'empty' };

/** Trim, collapse whitespace, lowercase — the tag normalization used everywhere. */
export function normalizeTag(input: string): string {
	return input.trim().split(/\s+/).join(' ').toLowerCase();
}

/** Only search aliases participate here. Stored tags always keep their own identity. */
const germanSearchTerms = new Map([
	['carrots', ['Möhre', 'Möhren']],
	['rolls', ['Semmel', 'Semmeln', 'Schrippe', 'Schrippen']],
	['cream', ['Rahm']],
	['sauces', ['Remoulade']]
]);

function searchText(text: string, locale?: string | null): string {
	const normalized = normalizeTag(text);
	return locale === 'de'
		? normalized
				.replaceAll('ä', 'ae')
				.replaceAll('ö', 'oe')
				.replaceAll('ü', 'ue')
				.replaceAll('ß', 'ss')
		: normalized;
}

/** Exact labels come first, followed by other matches in the displayed language's order. */
export function filterTags(options: string[], query: string, locale?: string | null): string[] {
	const q = searchText(query, locale);
	const collator = new Intl.Collator(locale === 'de' ? 'de' : 'en');
	const matches = options.map((tag) => {
		const label = productTagLabel(tag, locale);
		const aliases = locale === 'de' ? (germanSearchTerms.get(tag.toLowerCase()) ?? []) : [];
		const terms = [label, tag, ...aliases].map((term) => searchText(term, locale));
		return {
			tag,
			label,
			exact: terms[0] === q || terms[1] === q,
			matches: terms.some((term) => term.includes(q))
		};
	});
	return matches
		.filter((entry) => entry.matches)
		.sort((a, b) => Number(b.exact) - Number(a.exact) || collator.compare(a.label, b.label))
		.map((entry) => entry.tag);
}

/**
 * Build the full dropdown content for the given query and current selection.
 *
 * - empty vocabulary       → [{ kind: 'empty' }]
 * - otherwise              → the filtered tags (selected flags + optionIndex)
 * - non-empty query with no matches → a nomatch hint row plus a trailing
 *                                     create row (the only case where you can
 *                                     create a new tag)
 */
export function buildTagRows(
	options: string[],
	query: string,
	selected: string[],
	locale?: string | null
): TagRow[] {
	if (options.length === 0 && normalizeTag(query) === '') {
		return [{ kind: 'empty' }];
	}

	const selectedSet = new Set(selected.map((tag) => tag.toLowerCase()));
	const filtered = filterTags(options, query, locale);
	const rows: TagRow[] = filtered.map((tag, index) => ({
		kind: 'tag',
		tag,
		selected: selectedSet.has(tag.toLowerCase()),
		optionIndex: index
	}));

	const q = normalizeTag(query);
	if (q !== '' && filtered.length === 0) {
		rows.push({ kind: 'nomatch', query: q });
		rows.push({ kind: 'create', tag: productTagValue(q, locale), optionIndex: 0 });
	}

	return rows;
}

/**
 * Move the active option index with wrap-around. `from === -1` (nothing active)
 * starts at the first row for dir 1 and the last row for dir −1. Returns −1 when
 * `count === 0`.
 */
export function moveActiveIndex(count: number, from: number, dir: 1 | -1): number {
	if (count === 0) {
		return -1;
	}
	if (from < 0) {
		return dir === 1 ? 0 : count - 1;
	}
	return (from + dir + count) % count;
}
