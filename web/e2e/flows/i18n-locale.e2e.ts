import { expect, test } from '@playwright/test';
import { registerUser } from '../helpers/auth';

test.describe('i18n', () => {
	test('switch to German, persists across reload, switch back', async ({ page, context }) => {
		test.setTimeout(90_000);
		await registerUser(context, 'e2e-i18n');
		await page.goto('/');

		await page.locator('#locale-select').selectOption('de');
		await expect(page.getByRole('link', { name: 'Suche' })).toBeVisible();
		await expect(page.getByRole('button', { name: 'Abmelden' })).toBeVisible();

		// Locale is persisted in localStorage.
		await page.reload();
		await expect(page.getByRole('button', { name: 'Abmelden' })).toBeVisible();

		await page.locator('#locale-select').selectOption('en');
		await expect(page.getByRole('button', { name: 'Sign out' })).toBeVisible();
	});
});
