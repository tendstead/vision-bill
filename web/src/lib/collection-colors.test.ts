import { describe, expect, it } from 'vitest';
import { PRESET_COLORS, colorToStyle } from './collection-colors';

describe('collection colors', () => {
	it('has 6-12 presets, all valid hex', () => {
		expect(PRESET_COLORS.length).toBeGreaterThanOrEqual(6);
		expect(PRESET_COLORS.length).toBeLessThanOrEqual(12);
		for (const c of PRESET_COLORS) expect(/^#(?:[0-9a-f]{3}|[0-9a-f]{6})$/i.test(c)).toBe(true);
	});
	it('colorToStyle returns a CSS declaration, defaulting to neutral', () => {
		expect(colorToStyle('#123456')).toBe('background: #123456;');
		expect(colorToStyle(null)).toBe('background: var(--color-neutral, #64748B);');
	});
});
