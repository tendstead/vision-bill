import { writable } from 'svelte/store';

export type Theme = 'light' | 'black' | 'dark';
export function isTheme(value: unknown): value is Theme {
	return value === 'light' || value === 'black' || value === 'dark';
}
const initial = typeof document === 'undefined' ? 'light' : document.documentElement.dataset.theme;
export const theme = writable<Theme>(isTheme(initial) ? initial : 'light');
export function setTheme(value: Theme) {
	theme.set(value);
	document.documentElement.dataset.theme = value;
	document.documentElement.style.colorScheme = value === 'light' ? 'light' : 'dark';
	try {
		localStorage.setItem('vb-theme', value);
	} catch {
		/* Session appearance still applies. */
	}
}
