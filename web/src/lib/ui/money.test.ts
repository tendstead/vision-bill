import { describe, expect, it } from 'vitest';
import { isPlainDecimal, isSignedDecimal } from './money';

describe('isPlainDecimal', () => {
	it('accepts plain non-negative decimals', () => {
		expect(isPlainDecimal('5')).toBe(true);
		expect(isPlainDecimal('5.20')).toBe(true);
	});

	it('rejects negative values and malformed input', () => {
		expect(isPlainDecimal('-5')).toBe(false);
		expect(isPlainDecimal('-0.75')).toBe(false);
		expect(isPlainDecimal('abc')).toBe(false);
		expect(isPlainDecimal('5.')).toBe(false);
		expect(isPlainDecimal('.20')).toBe(false);
		expect(isPlainDecimal('')).toBe(false);
		expect(isPlainDecimal(null)).toBe(false);
	});
});

describe('isSignedDecimal', () => {
	it('accepts plain decimals with an optional leading minus', () => {
		expect(isSignedDecimal('5')).toBe(true);
		expect(isSignedDecimal('5.20')).toBe(true);
		expect(isSignedDecimal('-5')).toBe(true);
		expect(isSignedDecimal('-0.75')).toBe(true);
		expect(isSignedDecimal('-2.50')).toBe(true);
	});

	it('rejects malformed input', () => {
		expect(isSignedDecimal('abc')).toBe(false);
		expect(isSignedDecimal('5.')).toBe(false);
		expect(isSignedDecimal('.20')).toBe(false);
		expect(isSignedDecimal('')).toBe(false);
		expect(isSignedDecimal(null)).toBe(false);
	});
});
