import { expect, it } from 'vitest';
import { hasOverlapRows, unknownOverlap } from './overlap-decisions';
it('does not use pre-reconciliation removal indices as current row labels', () => {
	const removed = { first_index: 0, second_index: 1, verified: true, removed_index: 1 };
	expect(hasOverlapRows(removed, 2, [removed])).toBe(false);
});
it('does not label another pair with current indices after any earlier removal', () => {
	const earlier = { first_index: 0, second_index: 1, verified: true, removed_index: 1 };
	const later = { first_index: 1, second_index: 2, verified: false, removed_index: null };
	expect(hasOverlapRows(later, 3, [earlier, later])).toBe(false);
});
it('still invalidates a pair only when an actual positional edit requests it', () => {
	const original = { first_index: 1, second_index: 2, verified: true, removed_index: 2 };
	expect(unknownOverlap(original)).toEqual({ verified: false, association_unknown: true });
	expect(original).toEqual({ first_index: 1, second_index: 2, verified: true, removed_index: 2 });
});
