import { expect, test } from '@playwright/test';

test('shell controls follow the selected language while the mobile drawer is open', async ({
	page
}) => {
	await page.route('**/api/v1/**', async (route) => {
		const path = new URL(route.request().url()).pathname;
		if (path.endsWith('/auth/me')) {
			await route.fulfill({ json: { id: 'test-user', username: 'reviewer', is_admin: false } });
		} else if (path.endsWith('/collections/active')) {
			await route.fulfill({ status: 204 });
		} else if (path.endsWith('/system/ui-config')) {
			await route.fulfill({ json: { bypass_review_default: false, registration_open: true } });
		} else {
			await route.fulfill({ json: [] });
		}
	});
	await page.setViewportSize({ width: 375, height: 812 });
	await page.goto('/upload');
	await page.getByRole('button', { name: 'Menu', exact: true }).click();
	const drawer = page.getByRole('dialog');
	await drawer.locator('#locale-select').selectOption('de');
	await expect(drawer).toHaveAccessibleName('Alle Ziele');
	await expect(
		drawer.getByRole('navigation', { name: 'Gesamte Navigation', exact: true })
	).toBeVisible();
	await drawer.getByRole('button', { name: 'Menü schließen', exact: true }).click();
	await expect(page.getByRole('button', { name: 'Menü', exact: true })).toBeFocused();
	await expect(
		page.getByRole('navigation', { name: 'Mobile Navigation', exact: true })
	).toBeVisible();
	await page.reload();
	await page.getByRole('button', { name: 'Menü', exact: true }).click();
	await drawer.locator('#locale-select').selectOption('en');
	await expect(drawer).toHaveAccessibleName('All destinations');
	await drawer.getByRole('button', { name: 'Close menu', exact: true }).click();
	await expect(page.getByRole('button', { name: 'Menu', exact: true })).toBeFocused();
	await page.setViewportSize({ width: 1280, height: 900 });
	await page.locator('#locale-select').selectOption('de');
	await expect(
		page.getByRole('navigation', { name: 'Hauptnavigation', exact: true })
	).toBeVisible();
	await expect(page.getByRole('link', { name: 'Zum Inhalt springen', exact: true })).toHaveCount(1);
});

// Exercise the real layout/shell with deterministic session and API responses.
// A missing destination, inconsistent ordering or leaked admin link must fail.
for (const admin of [false, true]) {
	test(`desktop and mobile share ordered destinations (admin=${admin})`, async ({ page }) => {
		await page.route('**/api/v1/**', async (route) => {
			const path = new URL(route.request().url()).pathname;
			if (path.endsWith('/auth/me')) {
				await route.fulfill({ json: { id: 'test-user', username: 'reviewer', is_admin: admin } });
			} else if (path.endsWith('/collections/active')) {
				await route.fulfill({ status: 204 });
			} else if (path.endsWith('/system/ui-config')) {
				await route.fulfill({ json: { bypass_review_default: false, registration_open: true } });
			} else {
				await route.fulfill({ json: [] });
			}
		});
		const expected = [
			'/',
			'/upload',
			'/collections',
			'/receipts',
			'/search',
			'/statistics',
			'/queue',
			...(admin ? ['/benchmarks/results', '/settings'] : [])
		];
		await page.setViewportSize({ width: 1280, height: 900 });
		await page.goto('/upload');
		const desktop = page.getByRole('navigation', { name: 'Main navigation', exact: true });
		await expect(desktop).toBeVisible();
		await expect(desktop.getByRole('link')).toHaveCount(expected.length);
		expect(
			await desktop
				.getByRole('link')
				.evaluateAll((links) => links.map((link) => link.getAttribute('href')))
		).toEqual(expected);
		await page.setViewportSize({ width: 375, height: 812 });
		const bar = page.getByRole('navigation', { name: 'Mobile navigation', exact: true });
		await expect(bar).toBeVisible();
		expect(
			await bar
				.getByRole('link')
				.evaluateAll((links) => links.map((link) => link.getAttribute('href')))
		).toEqual(['/', '/upload', '/receipts']);
		await page.getByRole('button', { name: 'Menu', exact: true }).click();
		const drawer = page.getByRole('dialog');
		await expect(drawer).toBeVisible();
		const mobile = drawer.getByRole('navigation', { name: 'All navigation', exact: true });
		expect(
			await mobile
				.getByRole('link')
				.evaluateAll((links) => links.map((link) => link.getAttribute('href')))
		).toEqual(expected);
		await mobile.getByRole('link', { name: 'Search', exact: true }).click();
		await expect(page).toHaveURL(/\/search$/);
		await expect(drawer).toHaveCount(0);
		await page.getByRole('button', { name: 'Menu', exact: true }).click();
		await page.getByRole('dialog').getByRole('button', { name: 'Toggle theme' }).click();
		await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
		await page.keyboard.press('Escape');
		await expect(page.getByRole('button', { name: 'Menu', exact: true })).toBeFocused();
	});
}
