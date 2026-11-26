import { addMessages, init, locale, t, unwrapFunctionStore } from 'svelte-i18n';
import de from './de.json';
import en from './en.json';

export const LOCALES = ['en', 'de'] as const;
export type Locale = (typeof LOCALES)[number];

const STORAGE_KEY = 'vb-lang';

function detectLocale(): Locale {
	let stored: string | null = null;
	try {
		stored = localStorage.getItem(STORAGE_KEY);
	} catch {
		/* Browser language remains available without storage. */
	}
	if (stored === 'en' || stored === 'de') {
		return stored;
	}
	return navigator.language.toLowerCase().startsWith('de') ? 'de' : 'en';
}

addMessages('en', en);
addMessages('de', de);

const initialLocale = detectLocale();
document.documentElement.lang = initialLocale;

init({
	initialLocale,
	fallbackLocale: 'en'
});

export function setLocale(code: Locale): void {
	try {
		localStorage.setItem(STORAGE_KEY, code);
	} catch {
		/* Keep session-only language changes usable. */
	}
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
