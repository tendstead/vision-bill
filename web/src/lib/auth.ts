import { get, writable, type Writable } from 'svelte/store';
import { goto } from '$app/navigation';
import { resolve } from '$app/paths';
import { api, onUnauthorized } from '$lib/api/client';
import type { User } from '$lib/types';

/**
 * Client-side session state. `loading` until the first `/auth/me` round-trip
 * settles; `null` when signed out; the `User` when signed in. The server keeps
 * the authoritative session in the HttpOnly cookie — this store is just the
 * client's view of it for rendering and guarding.
 */
export type SessionState = User | null | 'loading';

export const session: Writable<SessionState> = writable<SessionState>('loading');

/** Populate the session from the server. Call once on app start. */
export async function initSession(): Promise<void> {
	try {
		const user = await api.me();
		session.set(user);
	} catch {
		// 401 (no session) or a network error: treat as signed out.
		session.set(null);
	}
}

/** Authenticate with username/password and record the returned user. */
export async function signIn(username: string, password: string): Promise<User> {
	const user = await api.login(username, password);
	session.set(user);
	return user;
}

/** Create an account and sign in with it. */
export async function signUp(username: string, password: string): Promise<User> {
	const user = await api.register(username, password);
	session.set(user);
	return user;
}

/** Clear the session (server cookie via the endpoint + local state).

A failing logout endpoint (e.g. offline) must not block sign-out: the local
session is cleared regardless and the caller redirects to the login page. */
export async function signOut(): Promise<void> {
	try {
		await api.logout();
	} catch {
		// Ignore: clearing the local session is what actually signs the user out.
	} finally {
		session.set(null);
	}
}

// A 401 from a protected endpoint means the session is gone: forget the user
// locally and return to the login page. Registered here (not in the layout) so
// the handler is active for every request, including background refetches.
onUnauthorized(() => {
	const current = get(session);
	if (current !== null && current !== 'loading') {
		session.set(null);
	}
	void goto(resolve('/login'));
});
