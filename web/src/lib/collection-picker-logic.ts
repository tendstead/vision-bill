import type { CollectionSummary } from './types';

/** Trim, lowercase and collapse runs of whitespace so searches are forgiving. */
export function normalizeQuery(value: string): string {
	return value.trim().toLowerCase().replace(/\s+/g, ' ');
}

/**
 * Filter by (case-insensitive, whitespace-collapsed) substring; selected
 * options always remain visible. Selected rows sort first, then by name.
 */
export function filterCollections(
	list: CollectionSummary[],
	query: string,
	selected: string[] = []
): CollectionSummary[] {
	const q = normalizeQuery(query);
	const selectedSet = new Set(selected);
	return list
		.filter((c) => selectedSet.has(c.id) || normalizeQuery(c.name).includes(q))
		.sort(
			(a, b) =>
				Number(selectedSet.has(b.id)) - Number(selectedSet.has(a.id)) ||
				a.name.localeCompare(b.name)
		);
}
