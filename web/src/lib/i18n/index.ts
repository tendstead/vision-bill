import { addMessages, init, locale, t, unwrapFunctionStore } from 'svelte-i18n';
import de from './de.json';
import en from './en.json';

export const LOCALES = ['en', 'de'] as const;
export type Locale = (typeof LOCALES)[number];

const STORAGE_KEY = 'vb-lang';

function detectLocale(): Locale {
	const stored = localStorage.getItem(STORAGE_KEY);
	if (stored === 'en' || stored === 'de') {
		return stored;
	}
	return navigator.language.toLowerCase().startsWith('de') ? 'de' : 'en';
}

addMessages('en', en);
addMessages('de', de);

init({
	initialLocale: detectLocale(),
	fallbackLocale: 'en'
});

export function setLocale(code: Locale): void {
	localStorage.setItem(STORAGE_KEY, code);
	document.documentElement.lang = code;
	void locale.set(code);
}

/**
 * Synchronous translation helper for `<script>` blocks.
 *
 * `$t` only works in markup; `translate` unwraps the underlying formatter
 * store so non-template code (snackbars, fetch messages, …) can translate too.
 */
export const translate = unwrapFunctionStore(t);

export { locale, t };
