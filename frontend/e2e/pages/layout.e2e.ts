import { expect, test } from '@playwright/test';
import { registerUser } from '../helpers/auth';

test.describe('app chrome', () => {
	test('all three appearances persist', async ({ page, context }) => {
		await registerUser(context, 'e2e-layout');
		await page.goto('/');
		for (const theme of ['black', 'dark', 'light']) {
			await page.getByRole('combobox', { name: 'Appearance' }).selectOption(theme);
			await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
			await page.reload();
			await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
			await expect(page.getByRole('combobox', { name: 'Appearance' })).toHaveValue(theme);
			const contrasts = await page.evaluate(() => {
				const styles = getComputedStyle(document.documentElement);
				function colorToken(name: string) {
					const token = styles.getPropertyValue(name).trim().toLowerCase();
					if (/^#[a-f0-9]{3}$/.test(token)) {
						return '#' + Array.from(token.slice(1), (channel) => channel.repeat(2)).join('');
					}
					return token;
				}
				const luminance = (name: string) => {
					const hex = colorToken(`--vb-${name}`);
					if (!/^#[a-f0-9]{6}$/.test(hex)) throw new Error(`Invalid color token ${name}: ${hex}`);
					const linear = hex
						.match(/[a-f0-9]{2}/gi)!
						.map((value) => parseInt(value, 16) / 255)
						.map((value) => (value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4));
					return linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722;
				};
				const ratio = (a: string, b: string) => {
					const x = luminance(a),
						y = luminance(b);
					return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05);
				};
				return {
					text: [
						['on-surface', 'surface'],
						['on-surface-variant', 'surface-container-low'],
						['on-primary', 'action'],
						['on-error', 'error-action'],
						['on-warning', 'warning-action'],
						['on-error-container', 'error-container'],
						['on-success-container', 'success-container'],
						['on-warning-container', 'warning-container']
					].map(([a, b]) => ratio(a, b)),
					border: ratio('outline-variant', 'surface'),
					native: styles.colorScheme,
					tokens: Object.fromEntries(
						Array.from(styles)
							.filter((name) => name.startsWith('--vb-') && /^#[a-f0-9]{6}$/.test(colorToken(name)))
							.map((name) => [name.slice(5), colorToken(name)])
					),
					foregrounds: [
						'primary',
						'secondary',
						'tertiary',
						'error',
						'success',
						'warning',
						'on-surface',
						'on-surface-variant',
						'on-primary',
						'on-error',
						'on-warning',
						'on-primary-container',
						'on-secondary-container',
						'on-tertiary-container',
						'on-error-container',
						'on-success-container',
						'on-warning-container'
					].map((name) => colorToken(`--vb-${name}`)),
					surface: colorToken('--vb-surface')
				};
			});
			for (const contrast of contrasts.text) expect(contrast).toBeGreaterThanOrEqual(4.5);
			expect(contrasts.border).toBeGreaterThanOrEqual(3);
			expect(contrasts.native).toBe(theme === 'light' ? 'light' : 'dark');
			if (theme !== 'dark') {
				const foreground = theme === 'light' ? '#000000' : '#ffffff';
				expect(contrasts.foregrounds.every((value) => value === foreground)).toBe(true);
				expect(contrasts.surface).toBe(theme === 'light' ? '#ffffff' : '#000000');
				for (const [name, value] of Object.entries(contrasts.tokens)) {
					expect(value.slice(1, 3), name).toBe(value.slice(3, 5));
					expect(value.slice(3, 5), name).toBe(value.slice(5, 7));
				}
			}
		}
	});

	test('mobile navigation works at 375px', async ({ page, context }) => {
		await registerUser(context, 'e2e-layout');
		await page.setViewportSize({ width: 375, height: 812 });
		await page.goto('/');

		await expect(page.getByRole('navigation', { name: 'Mobile navigation' })).toBeVisible();
		await page.getByRole('button', { name: 'Menu', exact: true }).click();
		const drawer = page.getByRole('dialog');
		await expect(drawer.getByRole('link', { name: 'Search', exact: true })).toBeVisible();
		await expect(drawer.getByRole('button', { name: 'Sign out' })).toBeVisible();
		await expect(drawer.getByRole('combobox', { name: 'Appearance' })).toBeVisible();

		await drawer.getByRole('button', { name: 'Sign out' }).click();
		await expect(page).toHaveURL(/\/login$/);
	});
});
