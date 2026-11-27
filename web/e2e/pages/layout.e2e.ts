import { expect, test } from '@playwright/test';
import { registerUser } from '../helpers/auth';

test.describe('app chrome', () => {
	test('theme toggle persists', async ({ page, context }) => {
		await registerUser(context, 'e2e-layout');
		await page.goto('/');
		await page.getByRole('button', { name: 'Toggle theme' }).click();
		expect(await page.evaluate(() => document.documentElement.dataset.theme)).toBe('dark');
		await page.reload();
		expect(await page.evaluate(() => document.documentElement.dataset.theme)).toBe('dark');
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
		await expect(drawer.getByRole('button', { name: 'Toggle theme' })).toBeVisible();

		await drawer.getByRole('button', { name: 'Sign out' }).click();
		await expect(page).toHaveURL(/\/login$/);
	});
});
