import { get, writable } from 'svelte/store';
import type { User } from '$lib/types';

// The cookie is authoritative. This store represents the account rendered
// by the SPA. Keeping it here lets the client capture intent without an auth
// module dependency.
export type SessionState = User | null | 'loading';
export const session = writable<SessionState>('loading');

export interface SessionActor {
	readonly generation: number;
	readonly userId: string;
}

// Separate from auth/API modules so all requests can capture their origin.
export const sessionGeneration = writable(0);
export const currentSessionGeneration = () => get(sessionGeneration);
export function advanceSessionGeneration(): number {
	const next = currentSessionGeneration() + 1;
	sessionGeneration.set(next);
	return next;
}
export function requireSessionGeneration(generation: number): void {
	if (generation !== currentSessionGeneration()) {
		throw new DOMException('Session changed', 'AbortError');
	}
}

export function captureSessionActor(): SessionActor | null {
	const user = get(session);
	if (!user || user === 'loading') return null;
	return { generation: currentSessionGeneration(), userId: user.id };
}

export function isCurrentSessionActor(actor: SessionActor | null): actor is SessionActor {
	const current = captureSessionActor();
	return !!actor && current?.generation === actor.generation && current.userId === actor.userId;
}

export function requireCurrentSessionActor(
	actor: SessionActor | null
): asserts actor is SessionActor {
	if (!isCurrentSessionActor(actor)) {
		throw new DOMException('Session changed', 'AbortError');
	}
}
