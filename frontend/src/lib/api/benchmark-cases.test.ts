import { afterEach, expect, it, vi } from 'vitest';

afterEach(() => {
	vi.resetModules();
	vi.unstubAllGlobals();
	vi.unstubAllEnvs();
});

it('never attaches the private media account parameter to an external API origin', async () => {
	vi.stubEnv('VITE_API_BASE', 'https://external.invalid/api/v1');
	const { session, captureSessionActor } = await import('$lib/session-lifetime');
	const { caseApi } = await import('./benchmark-cases');
	session.set({ id: 'actor', username: 'admin', is_admin: true, can_see_all: false });
	expect(caseApi.photoUrl('case', 0, captureSessionActor()!)).toBeUndefined();
});

it('disables original photo URLs and requests from an earlier login of the same account', async () => {
	const { session, captureSessionActor, advanceSessionGeneration } =
		await import('$lib/session-lifetime');
	const { caseApi } = await import('./benchmark-cases');
	session.set({ id: 'actor', username: 'admin', is_admin: true, can_see_all: false });
	const origin = captureSessionActor()!;
	expect(caseApi.photoUrl('case', 0, origin)).toBe(
		'/api/v1/benchmarks/cases/case/photos/0?vb_expected_user=actor'
	);
	advanceSessionGeneration();
	const fetch = vi.fn();
	vi.stubGlobal('fetch', fetch);
	expect(caseApi.photoUrl('case', 0, origin)).toBeUndefined();
	await expect(caseApi.get('case', origin)).rejects.toMatchObject({ name: 'AbortError' });
	expect(fetch).not.toHaveBeenCalled();
});
