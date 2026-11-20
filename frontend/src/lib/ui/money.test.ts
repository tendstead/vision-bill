import { describe, expect, it } from 'vitest';
import { decMul, isPlainDecimal, isSignedDecimal } from './money';

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

it.each([
	[1.5, '1.23', '1.845'],
	[2, '-0.75', '-1.5'],
	[0.25, '-2.50', '-0.625'],
	[0, '-0.75', '0'],
	[1e-7, '0.123', '0.0000000123'],
	[3, '10000000000000000000.0001', '30000000000000000000.0003']
] as const)('exactly multiplies %s by %s', (quantity, price, expected) => {
	expect(decMul(quantity, price)).toBe(expected);
});
