import labels from '../../../../src/vision_bill/model/product_tags.json';

const vocabulary = new Map(Object.entries(labels));

export function productTagLabel(tag: string, locale: string | null | undefined): string {
	const entry = vocabulary.get(tag.toLowerCase());
	return entry ? entry[locale === 'de' ? 'de' : 'en'] : tag;
}

/** Resolve an explicitly entered bilingual vocabulary label. */
export function productTagValue(label: string, locale: string | null | undefined): string {
	const normalized = label.trim().toLowerCase();
	for (const [code, names] of vocabulary) {
		if (code === normalized || names[locale === 'de' ? 'de' : 'en'].toLowerCase() === normalized)
			return code;
	}
	return label;
}

/** Mark custom values whose label is also in use by a built-in tag in this view. */
export function collidingProductTagLabels(tags: string[], locale: string | null | undefined) {
	const builtIn = new Set<string>();
	const custom = new Set<string>();
	for (const tag of tags) {
		const normalized = productTagLabel(tag, locale).trim().toLowerCase();
		(vocabulary.has(tag.toLowerCase()) ? builtIn : custom).add(normalized);
	}
	return new Set([...custom].filter((label) => builtIn.has(label)));
}

export function hasCustomTagCollision(tag: string, collisions: Set<string>): boolean {
	return !vocabulary.has(tag.toLowerCase()) && collisions.has(tag.trim().toLowerCase());
}

/** Keep colliding lines distinct and reversible in the editable benchmark reference. */
export function referenceTagText(
	tag: string,
	locale: string | null | undefined,
	collisions: Set<string>
) {
	const label = productTagLabel(tag, locale);
	return collisions.has(label.trim().toLowerCase()) ? tag : label;
}
