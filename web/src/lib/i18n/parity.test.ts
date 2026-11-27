import { describe, expect, it } from 'vitest';
import en from './en.json';
import de from './de.json';

function flattenKeys(value: unknown, prefix = ''): string[] {
	if (value === null || typeof value !== 'object' || Array.isArray(value)) {
		return [prefix];
	}
	const entries = Object.entries(value as Record<string, unknown>);
	if (entries.length === 0) {
		return [prefix];
	}
	return entries.flatMap(([key, child]) => flattenKeys(child, prefix ? `${prefix}.${key}` : key));
}

describe('i18n key parity', () => {
	it('has identical key sets in en and de', () => {
		const enKeys = flattenKeys(en).sort();
		const deKeys = flattenKeys(de).sort();
		expect(deKeys).toEqual(enKeys);
	});

	it('only uses string leaf values', () => {
		for (const messages of [en, de]) {
			function walk(value: unknown): void {
				if (value === null || typeof value === 'object') {
					for (const child of Object.values(value as Record<string, unknown>)) {
						walk(child);
					}
					return;
				}
				expect(typeof value).toBe('string');
			}
			walk(messages);
		}
	});
});
