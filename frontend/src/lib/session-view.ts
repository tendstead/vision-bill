import { onDestroy } from 'svelte';
import { captureSessionActor, isCurrentSessionActor } from '$lib/session-lifetime';

/** Call during component initialization to check ownership of later view effects. */
export function sessionViewGuard(): () => boolean {
	const actor = captureSessionActor();
	let alive = true;
	onDestroy(() => {
		alive = false;
	});
	return () => alive && isCurrentSessionActor(actor);
}
