import { describe, expect, it } from 'vitest';
import { missingThumbSource } from './image-policy';

describe('missingThumbSource', () => {
	it('defaults to the full-size image URL (lazy load)', () => {
		expect(missingThumbSource('abc-123')).toBe('/api/v1/images/abc-123/file');
	});
});
