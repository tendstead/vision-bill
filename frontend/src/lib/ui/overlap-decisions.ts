/** Display eligibility only: extraction indices are not validated or rewritten by form bounds. */
export function hasOverlapRows(
	decision: Record<string, unknown>,
	count: number,
	decisions: Record<string, unknown>[] = [decision]
): boolean {
	// A producer removal can shift every later extraction index, including pairs
	// whose own indices still happen to fit inside the final form's row count.
	if (
		decisions.some(
			(evidence) => evidence.removed_index !== null && evidence.removed_index !== undefined
		)
	)
		return false;
	const first = decision.first_index,
		second = decision.second_index;
	return (
		decision.association_unknown !== true &&
		typeof first === 'number' &&
		Number.isInteger(first) &&
		typeof second === 'number' &&
		Number.isInteger(second) &&
		first >= 0 &&
		second >= 0 &&
		first < count &&
		second < count &&
		first !== second
	);
}
export function unknownOverlap(decision: Record<string, unknown>): Record<string, unknown> {
	const result = { ...decision, verified: false, association_unknown: true };
	for (const key of ['first_index', 'second_index', 'removed_index'])
		delete (result as Record<string, unknown>)[key];
	return result;
}
