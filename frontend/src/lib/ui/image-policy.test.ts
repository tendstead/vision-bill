import { session } from '$lib/session-lifetime';
import { describe, expect, it } from 'vitest';
import { missingThumbSource } from './image-policy';

describe('missingThumbSource', () => {
	it('defaults to the full-size image URL (lazy load)', () => {
		session.set({ id: 'viewer', username: 'alice', is_admin: false, can_see_all: false });
		expect(missingThumbSource('abc-123')).toBe(
			'/api/v1/images/abc-123/file?vb_expected_user=viewer'
		);
	});
});
