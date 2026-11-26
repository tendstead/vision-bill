import { describe, expect, it } from 'vitest';
import html from '../app.html?raw';

describe('appearance before first paint', () => {
	function boot(saved: string | null, dark = false, denied = false) {
		const root = { dataset: {} as Record<string, string>, style: {} as Record<string, string> };
		const script = html.match(/<script>([\s\S]*?)<\/script>/)![1];
		new Function('localStorage', 'window', 'document', script)(
			{
				getItem: () => {
					if (denied) throw new Error('denied');
					return saved;
				}
			},
			{ matchMedia: () => ({ matches: dark }) },
			{ documentElement: root }
		);
		return root;
	}
	it.each(['light', 'black', 'dark'])('preserves saved %s', (theme) => {
		expect(boot(theme).dataset.theme).toBe(theme);
		expect(boot(theme).style.colorScheme).toBe(theme === 'light' ? 'light' : 'dark');
	});
	it('falls back safely when storage is unavailable', () => {
		expect(boot(null, true, true).dataset.theme).toBe('dark');
	});
});
