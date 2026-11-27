import { get } from 'svelte/store';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

// Hoisted so the mock factories (processed before the imports below) can close
// over shared state.
const { mockApi, mockGoto, setUnauthorizedHandler, getUnauthorizedHandler } = vi.hoisted(() => {
	const mockApi = {
		me: vi.fn(),
		login: vi.fn(),
		register: vi.fn(),
		logout: vi.fn()
	};
	const mockGoto = vi.fn();
	let handler: (() => void) | undefined;
	return {
		mockApi,
		mockGoto,
		setUnauthorizedHandler: (fn: () => void) => {
			handler = fn;
		},
		getUnauthorizedHandler: () => handler
	};
});

vi.mock('$lib/api/client', () => ({
	api: mockApi,
	onUnauthorized: setUnauthorizedHandler
}));
vi.mock('$app/navigation', () => ({ goto: mockGoto }));
vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));

import { initSession, session, signIn, signOut, signUp } from './auth';

const ALICE = {
	id: '00000000-0000-4000-8000-000000000001',
	username: 'alice',
	is_admin: false,
	can_see_all: false
};

beforeEach(() => {
	vi.clearAllMocks();
	session.set('loading');
});

afterEach(() => {
	session.set('loading');
});

describe('session state', () => {
	it('initSession stores the user from a successful /auth/me', async () => {
		mockApi.me.mockResolvedValue(ALICE);

		await initSession();

		expect(mockApi.me).toHaveBeenCalledOnce();
		expect(get(session)).toEqual(ALICE);
	});

	it('initSession treats a 401 (no session) as signed out', async () => {
		mockApi.me.mockRejectedValue(new Error('401 Unauthorized'));

		await initSession();

		expect(get(session)).toBeNull();
	});

	it('initSession treats a network error as signed out', async () => {
		mockApi.me.mockRejectedValue(new Error('Network error'));

		await initSession();

		expect(get(session)).toBeNull();
	});

	it('signIn authenticates and stores the returned user', async () => {
		mockApi.login.mockResolvedValue(ALICE);

		await signIn('alice', 'secret');

		expect(mockApi.login).toHaveBeenCalledWith('alice', 'secret');
		expect(get(session)).toEqual(ALICE);
	});

	it('signUp registers and stores the returned user', async () => {
		mockApi.register.mockResolvedValue(ALICE);

		await signUp('alice', 'secret');

		expect(mockApi.register).toHaveBeenCalledWith('alice', 'secret');
		expect(get(session)).toEqual(ALICE);
	});

	it('signOut clears the session even when the endpoint fails', async () => {
		session.set(ALICE);
		mockApi.logout.mockRejectedValue(new Error('boom'));

		await signOut();

		expect(mockApi.logout).toHaveBeenCalledOnce();
		expect(get(session)).toBeNull();
	});

	it('a protected-endpoint 401 clears the session and redirects to /login', () => {
		const handler = getUnauthorizedHandler();
		expect(handler).toBeTypeOf('function');
		session.set(ALICE);

		handler!();

		expect(get(session)).toBeNull();
		expect(mockGoto).toHaveBeenCalledWith('/login');
	});

	it('does not double-clear an already-signed-out session but still redirects', () => {
		const handler = getUnauthorizedHandler();
		session.set(null);

		handler!();

		expect(get(session)).toBeNull();
		expect(mockGoto).toHaveBeenCalledWith('/login');
	});
});
