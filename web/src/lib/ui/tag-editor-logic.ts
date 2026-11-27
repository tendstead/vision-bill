/**
 * Pure dropdown logic for the TagEditor chip-input combobox.
 * Kept free of Svelte so it can be unit-tested directly.
 */

export type TagRow =
	| { kind: 'tag'; tag: string; selected: boolean; optionIndex: number }
	| { kind: 'create'; tag: string; optionIndex: number }
	| { kind: 'nomatch'; query: string }
	| { kind: 'empty' };

/** Trim, collapse whitespace, lowercase — the tag normalization used everywhere. */
export function normalizeTag(input: string): string {
	return input.trim().split(/\s+/).join(' ').toLowerCase();
}

/** Case-insensitive substring filter; an empty query returns all options in order. */
export function filterTags(options: string[], query: string): string[] {
	const q = normalizeTag(query);
	if (q === '') {
		return [...options];
	}
	return options.filter((option) => option.toLowerCase().includes(q));
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
export function buildTagRows(options: string[], query: string, selected: string[]): TagRow[] {
	if (options.length === 0) {
		return [{ kind: 'empty' }];
	}

	const selectedSet = new Set(selected.map((tag) => tag.toLowerCase()));
	const filtered = filterTags(options, query);
	const rows: TagRow[] = filtered.map((tag, index) => ({
		kind: 'tag',
		tag,
		selected: selectedSet.has(tag.toLowerCase()),
		optionIndex: index
	}));

	const q = normalizeTag(query);
	if (q !== '' && filtered.length === 0) {
		rows.push({ kind: 'nomatch', query: q });
		rows.push({ kind: 'create', tag: q, optionIndex: 0 });
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
