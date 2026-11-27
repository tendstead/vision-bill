import { writable } from 'svelte/store';

// crypto.randomUUID() is only available in secure contexts (HTTPS or
// localhost). On a plain-HTTP LAN deployment (e.g. http://nas:8080) it is
// undefined, so fall back to crypto.getRandomValues (available everywhere) to
// build a v4 UUID.
function uuid(): string {
	const c = globalThis.crypto;
	if (c && typeof c.randomUUID === 'function') {
		return c.randomUUID();
	}
	const bytes = c.getRandomValues(new Uint8Array(16));
	bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
	bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 10
	const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
	return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export type SnackbarKind = 'success' | 'error' | 'info';

export interface SnackbarItem {
	id: string;
	kind: SnackbarKind;
	text: string;
}

function createSnackbar() {
	const { subscribe, update } = writable<SnackbarItem[]>([]);

	return {
		subscribe,
		notify(kind: SnackbarKind, text: string, duration = 6000): void {
			const id = uuid();
			update((items) => [...items, { id, kind, text }]);
			if (duration > 0) {
				setTimeout(() => {
					update((items) => items.filter((item) => item.id !== id));
				}, duration);
			}
		},
		close(id: string): void {
			update((items) => items.filter((item) => item.id !== id));
		}
	};
}

export const snackbar = createSnackbar();
