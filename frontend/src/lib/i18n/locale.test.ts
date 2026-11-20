import { get } from 'svelte/store';
import { afterEach, expect, it, vi } from 'vitest';

afterEach(() => {
	vi.unstubAllGlobals();
	vi.restoreAllMocks();
	vi.resetModules();
	document.documentElement.lang = 'en';
});
it.each(['en', 'de'])('initializes the document and formatter from saved %s', async (code) => {
	vi.stubGlobal('localStorage', { getItem: () => code, setItem: vi.fn() });
	const { locale } = await import('./index');
	expect(document.documentElement.lang).toBe(code);
	expect(get(locale)).toBe(code);
});
it.each(['en-US', 'de-DE'])(
	'uses browser %s when storage is denied and permits session-only changes',
	async (language) => {
		vi.stubGlobal('navigator', { language });
		vi.stubGlobal('localStorage', {
			getItem: () => {
				throw new Error('denied');
			},
			setItem: () => {
				throw new Error('denied');
			}
		});
		const { locale, setLocale } = await import('./index');
		const expected = language.startsWith('de') ? 'de' : 'en';
		expect(document.documentElement.lang).toBe(expected);
		expect(get(locale)).toBe(expected);
		expect(() => setLocale(expected === 'en' ? 'de' : 'en')).not.toThrow();
		expect(document.documentElement.lang).toBe(expected === 'en' ? 'de' : 'en');
	}
);
