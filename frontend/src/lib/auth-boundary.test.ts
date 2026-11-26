import { get } from 'svelte/store';
import { QueryObserver } from '@tanstack/svelte-query';
import { afterEach, expect, it, vi } from 'vitest';
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));
vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
import { api } from './api/client';
import { initSession, session, signIn, signOut, signUp } from './auth';
import { queryClient } from './query/client';
import { queryKeys } from './query/keys';
const alice = { id: 'a', username: 'alice', is_admin: false, can_see_all: false };
const bob = { ...alice, id: 'b', username: 'bob' };
function deferred<T>() {
	let resolve!: (value: T) => void;
	const promise = new Promise<T>((r) => {
		resolve = r;
	});
	return { promise, resolve };
}
const response = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });
afterEach(() => {
	queryClient.clear();
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
});
it('disposes warm private queries before B can observe them in the same SPA', async () => {
	vi.spyOn(api, 'login').mockResolvedValueOnce(alice).mockResolvedValueOnce(bob);
	vi.spyOn(api, 'logout').mockResolvedValue();
	await signIn('alice', 'x');
	const keys = [
		queryKeys.receipts(),
		queryKeys.receipt('a'),
		queryKeys.statistics(),
		queryKeys.collectionsActive()
	];
	for (const key of keys) queryClient.setQueryData(key, { private: 'Alice receipt' });
	await signOut();
	await signIn('bob', 'x');
	for (const key of keys) {
		const observer = new QueryObserver(queryClient, {
			queryKey: key,
			queryFn: async () => ({ private: 'Bob receipt' })
		});
		expect(observer.getCurrentResult().data).toBeUndefined();
		observer.destroy();
	}
});
it('cancels a pending old query so its late result cannot overwrite B', async () => {
	vi.spyOn(api, 'login').mockResolvedValueOnce(alice).mockResolvedValueOnce(bob);
	await signIn('alice', 'x');
	const late = deferred<{ private: string }>();
	const pending = queryClient
		.fetchQuery({ queryKey: queryKeys.receipts(), queryFn: () => late.promise })
		.catch(() => undefined);
	await signIn('bob', 'x');
	queryClient.setQueryData(queryKeys.receipts(), { private: 'Bob' });
	late.resolve({ private: 'Alice' });
	await pending;
	expect(queryClient.getQueryData(queryKeys.receipts())).toEqual({ private: 'Bob' });
});
it.each(['init', 'login', 'signup', 'logout'] as const)(
	'ignores late %s auth completion after a successor sign-in',
	async (kind) => {
		const late = deferred<typeof alice>();
		vi.spyOn(api, 'me').mockReturnValue(late.promise);
		vi.spyOn(api, 'register').mockReturnValue(late.promise);
		vi.spyOn(api, 'logout').mockImplementation(() => late.promise.then(() => undefined));
		const login = vi.spyOn(api, 'login').mockResolvedValue(bob);
		if (kind === 'login') login.mockReturnValueOnce(late.promise);
		const old = (
			kind === 'init'
				? initSession()
				: kind === 'login'
					? signIn('alice', 'x')
					: kind === 'signup'
						? signUp('alice', 'x')
						: signOut()
		).catch(() => undefined);
		await Promise.resolve();
		const next = signIn('bob', 'x');
		late.resolve(alice);
		await old;
		await next;
		expect(get(session)).toEqual(bob);
	}
);
it('does not expire B for a late A protected 401 or expose a late successful A body', async () => {
	vi.spyOn(api, 'login').mockResolvedValueOnce(alice).mockResolvedValueOnce(bob);
	await signIn('alice', 'x');
	const late401 = deferred<Response>(),
		lateBody = deferred<Response>();
	vi.stubGlobal(
		'fetch',
		vi.fn().mockReturnValueOnce(late401.promise).mockReturnValueOnce(lateBody.promise)
	);
	const old401 = api.listReceipts().catch(() => undefined);
	const oldBody = api.getReceipt('a').catch(() => undefined);
	await signIn('bob', 'x');
	late401.resolve(response({}, 401));
	lateBody.resolve(response({ private: 'Alice' }));
	await old401;
	expect(await oldBody).toBeUndefined();
	expect(get(session)).toEqual(bob);
});
it('protected 401 disposes cached private data before redirect', async () => {
	vi.spyOn(api, 'login').mockResolvedValue(alice);
	await signIn('alice', 'x');
	queryClient.setQueryData(queryKeys.receipts(), ['Alice']);
	vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response({}, 401)));
	await api.listReceipts().catch(() => undefined);
	expect(get(session)).toBeNull();
	expect(queryClient.getQueryData(queryKeys.receipts())).toBeUndefined();
});

// Fake transport applies Set-Cookie at response delivery, as a browser does.
it.each(['login', 'signup', 'logout'] as const)(
	'orders the %s cookie write before successor login',
	async (kind) => {
		session.set(alice);
		let serverPrincipal: string | null = 'alice';
		const firstResponse = deferred<Response>();
		const calls: string[] = [];
		vi.stubGlobal(
			'fetch',
			vi.fn(async (url: string) => {
				calls.push(url);
				if (calls.length === 1) {
					const result = await firstResponse.promise;
					serverPrincipal = kind === 'logout' ? null : 'alice';
					return result;
				}
				serverPrincipal = 'bob';
				return response(bob);
			})
		);
		const first = (
			kind === 'login' ? signIn('alice', 'x') : kind === 'signup' ? signUp('alice', 'x') : signOut()
		).catch(() => undefined);
		await Promise.resolve();
		await Promise.resolve();
		const next = signIn('bob', 'x');
		await Promise.resolve();
		await Promise.resolve();
		expect(calls).toHaveLength(1);
		firstResponse.resolve(response(alice));
		await first;
		await next;
		expect(serverPrincipal).toBe('bob');
		expect(get(session)).toEqual(bob);
	}
);

it('does not expose an old protected error body that finishes after the principal changes', async () => {
	vi.spyOn(api, 'login').mockResolvedValueOnce(alice).mockResolvedValueOnce(bob);
	await signIn('alice', 'x');
	const body = deferred<unknown>();
	const json = vi.fn(() => body.promise);
	vi.stubGlobal(
		'fetch',
		vi.fn().mockResolvedValue({ ok: false, status: 409, statusText: 'Conflict', json })
	);
	const pending = api.getReceipt('a').catch((error) => error);
	await Promise.resolve();
	await Promise.resolve();
	expect(json).toHaveBeenCalled();
	await signIn('bob', 'x');
	body.resolve({ detail: 'Alice private conflict' });
	expect((await pending).name).toBe('AbortError');
});

it('binds protected browser requests to the view actor and leaves bootstrap discovery unbound', async () => {
	vi.spyOn(api, 'login').mockResolvedValue(alice);
	await signIn('alice', 'x');
	vi.stubGlobal(
		'fetch',
		vi.fn().mockImplementation(() => Promise.resolve(response({})))
	);
	await api.uploadImages([new File(['image'], 'receipt.png')], {}, 'stable-key');
	await api.me();
	await api.getUiConfig();
	const [upload, me, uiConfig] = vi
		.mocked(fetch)
		.mock.calls.map(([, init]) => new Headers(init?.headers));
	expect(upload.get('X-Vision-Bill-Expected-User')).toBe(alice.id);
	expect(upload.get('Idempotency-Key')).toBe('stable-key');
	expect(upload.has('Content-Type')).toBe(false);
	expect(me.has('X-Vision-Bill-Expected-User')).toBe(false);
	expect(uiConfig.has('X-Vision-Bill-Expected-User')).toBe(false);
});

it('revalidates a current actor mismatch and empties its private cache', async () => {
	vi.spyOn(api, 'login').mockResolvedValue(alice);
	const me = vi.spyOn(api, 'me').mockResolvedValue(bob);
	await signIn('alice', 'x');
	queryClient.setQueryData(queryKeys.receipts(), ['Alice']);
	vi.stubGlobal(
		'fetch',
		vi.fn().mockResolvedValue(response({ detail: 'session_actor_changed' }, 409))
	);
	await expect(api.listReceipts()).rejects.toMatchObject({ name: 'AbortError' });
	await vi.waitFor(() => expect(get(session)).toEqual(bob));
	expect(me).toHaveBeenCalledOnce();
	expect(queryClient.getQueryData(queryKeys.receipts())).toBeUndefined();
});

it('ignores a delayed actor mismatch body once a successor signs in', async () => {
	vi.spyOn(api, 'login').mockResolvedValueOnce(alice).mockResolvedValueOnce(bob);
	const me = vi.spyOn(api, 'me');
	await signIn('alice', 'x');
	const late = deferred<unknown>();
	const json = vi.fn(() => late.promise);
	vi.stubGlobal(
		'fetch',
		vi.fn().mockResolvedValue({ ok: false, status: 409, statusText: 'Conflict', json })
	);
	const old = api.listReceipts().catch((error) => error);
	await vi.waitFor(() => expect(json).toHaveBeenCalled());
	await signIn('bob', 'x');
	late.resolve({ detail: 'session_actor_changed' });
	expect((await old).name).toBe('AbortError');
	expect(me).not.toHaveBeenCalled();
	expect(get(session)).toEqual(bob);
});

it('binds logout to the rendered actor and revalidates without clearing a successor session', async () => {
	vi.spyOn(api, 'login').mockResolvedValue(alice);
	await signIn('alice', 'x');
	const sent = vi.fn((url: string) =>
		Promise.resolve(
			url.endsWith('/logout') ? response({ detail: 'session_actor_changed' }, 409) : response(bob)
		)
	);
	vi.stubGlobal('fetch', sent);
	await signOut();
	expect(
		new Headers(vi.mocked(fetch).mock.calls[0][1]?.headers).get('X-Vision-Bill-Expected-User')
	).toBe(alice.id);
	expect(sent.mock.calls.map(([url]) => url.split('/').at(-1))).toEqual(['logout', 'me']);
	expect(get(session)).toEqual(bob);
});
