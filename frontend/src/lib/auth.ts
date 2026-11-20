import { get } from 'svelte/store';
import { goto } from '$app/navigation';
import { resolve } from '$app/paths';
import { api, ApiError, onSessionActorChanged, onUnauthorized } from '$lib/api/client';
import type { User } from '$lib/types';
import { queryClient } from '$lib/query/client';
import { snackbar } from '$lib/ui/snackbar.svelte';
import {
	advanceSessionGeneration,
	captureSessionActor,
	currentSessionGeneration,
	requireSessionGeneration,
	session,
	type SessionState
} from '$lib/session-lifetime';

export { session };

function beginSession(state: SessionState = null): number {
	session.set('loading');
	const generation = advanceSessionGeneration();
	// Cancellation settles Query retryers before clearing; late query functions
	// cannot populate either the removed cache or successor observers.
	void queryClient.cancelQueries();
	queryClient.clear();
	snackbar.clear();
	sessionStorage.removeItem('benchmark-reuse-receipts');
	session.set(state);
	return generation;
}

// Browser cookies are changed by response headers before JSON is consumed.
// Serialize all auth requests (including /me) so an old Set-Cookie cannot land
// after the successor login. Generation checks alone only protect JS state.
let authTail: Promise<unknown> = Promise.resolve();
function authRequest<T>(generation: number, operation: () => Promise<T>): Promise<T> {
	const result = authTail.then(() => {
		requireSessionGeneration(generation);
		return operation();
	});
	authTail = result.catch(() => undefined);
	return result;
}

let channel: BroadcastChannel | undefined;
let pendingCookieWrites = 0;

async function changeCookie<T>(generation: number, operation: () => Promise<T>): Promise<T> {
	pendingCookieWrites++;
	try {
		return await authRequest(generation, async () => {
			const write = async () => {
				requireSessionGeneration(generation);
				try {
					return await operation();
				} finally {
					// An obsolete local response can still carry an applied Set-Cookie.
					channel?.postMessage('cookie-may-have-changed');
				}
			};
			return navigator.locks ? navigator.locks.request('vision-bill-auth-cookie', write) : write();
		});
	} finally {
		pendingCookieWrites--;
	}
}

/** Populate the session from the server and remount the private view. */
export async function initSession(): Promise<void> {
	const generation = beginSession('loading');
	try {
		const user = await authRequest(generation, () => api.me());
		requireSessionGeneration(generation);
		session.set(user);
	} catch {
		// 401 (no session) or a network error: treat as signed out.
		if (generation === currentSessionGeneration()) session.set(null);
	}
}

/** Authenticate with username/password and record the returned user. */
export async function signIn(username: string, password: string): Promise<User> {
	const generation = beginSession();
	const user = await changeCookie(generation, () => api.login(username, password));
	requireSessionGeneration(generation);
	session.set(user);
	return user;
}

/** Create an account and sign in with it. */
export async function signUp(username: string, password: string): Promise<User> {
	const generation = beginSession();
	const user = await changeCookie(generation, () => api.register(username, password));
	requireSessionGeneration(generation);
	session.set(user);
	return user;
}

/** Clear the session (server cookie via the endpoint + local state).

A failing logout endpoint (e.g. offline) must not block sign-out: the local
session is cleared regardless and the caller redirects to the login page. */
export async function signOut(): Promise<void> {
	const actor = captureSessionActor();
	const generation = beginSession();
	if (!actor) return;
	try {
		await changeCookie(generation, () => api.logout(actor.userId));
	} catch (error) {
		if (
			generation === currentSessionGeneration() &&
			error instanceof ApiError &&
			error.status === 409 &&
			error.detail === 'session_actor_changed'
		) {
			await initSession();
		}
	}
}

// A 401 from a protected endpoint means the session is gone: forget the user
// locally and return to the login page. Registered here (not in the layout) so
// the handler is active for every request, including background refetches.
onUnauthorized(() => {
	beginSession();
	void goto(resolve('/login'));
});

onSessionActorChanged(() => {
	void initSession();
});

let checkingGeneration: number | undefined;

async function revalidateSession(): Promise<void> {
	const generation = currentSessionGeneration();
	if (checkingGeneration === generation || get(session) === 'loading') return;
	checkingGeneration = generation;
	try {
		const user = await authRequest(generation, () => api.me());
		requireSessionGeneration(generation);
		const previous = get(session);
		if (
			!previous ||
			previous === 'loading' ||
			previous.id !== user.id ||
			previous.username !== user.username ||
			previous.is_admin !== user.is_admin ||
			previous.can_see_all !== user.can_see_all
		) {
			beginSession(user);
		}
	} catch (error) {
		if (
			generation === currentSessionGeneration() &&
			error instanceof ApiError &&
			error.status === 401 &&
			get(session) !== null
		) {
			beginSession();
		}
	} finally {
		if (checkingGeneration === generation) checkingGeneration = undefined;
	}
}

/** Listen for cookie changes and recheck after a sleeping tab resumes. */
export function watchSession(): () => void {
	const messages =
		typeof BroadcastChannel === 'undefined' ? undefined : new BroadcastChannel('vision-bill-auth');
	channel = messages;
	function onMessage(event: MessageEvent) {
		if (event.data !== 'cookie-may-have-changed') return;
		// Recheck after a locally queued cookie write; do not obsolete that write.
		if (pendingCookieWrites) void revalidateSession();
		else void initSession();
	}
	function onResume() {
		if (document.visibilityState === 'visible') void revalidateSession();
	}

	messages?.addEventListener('message', onMessage);
	window.addEventListener('focus', onResume);
	window.addEventListener('pageshow', onResume);
	document.addEventListener('visibilitychange', onResume);
	return () => {
		window.removeEventListener('focus', onResume);
		window.removeEventListener('pageshow', onResume);
		document.removeEventListener('visibilitychange', onResume);
		messages?.close();
		if (channel === messages) channel = undefined;
	};
}
