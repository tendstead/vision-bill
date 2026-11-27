import { expect, test } from '@playwright/test';
import { loginAdmin } from '../helpers/auth';

test.describe('admin: settings + registration toggle', () => {
	test('admin can close and reopen registration', async ({ page, context, browser }) => {
		test.setTimeout(150_000);
		await loginAdmin(context);
		await page.goto('/settings');

		// Source labels from the e2e stack: provider/host/model are
		// environment-controlled (locked), temperature comes from config.
		await expect(page.getByText('Environment').first()).toBeVisible();
		await expect(page.getByText('Config').first()).toBeVisible();
		await expect(page.getByLabel('Provider')).toBeDisabled();
		await expect(page.getByLabel('Host')).toBeDisabled();
		await expect(page.getByLabel('Default model')).toBeDisabled();
		await expect(page.getByLabel('Temperature')).toBeEnabled();

		try {
			// Close registration.
			await page.getByLabel('Allow new registrations').uncheck();
			await page.getByRole('button', { name: 'Save', exact: true }).click();
			await expect(page.getByText('Settings saved.')).toBeVisible({ timeout: 15_000 });

			// The login page hides the registration link...
			const contextB = await browser.newContext();
			const pageB = await contextB.newPage();
			await pageB.goto('/login');
			await expect(pageB.getByRole('button', { name: 'Create one' })).toHaveCount(0);
			// ...and the API rejects new accounts.
			const response = await contextB.request.post('/api/v1/auth/register', {
				data: { username: 'e2e-closed', password: 'e2e-pw-12345' }
			});
			expect(response.status()).toBe(403);
			expect(await response.text()).toContain('Registration is disabled');
			await contextB.close();
		} finally {
			// Reopen registration (leave the suite in a usable state).
			await page.getByLabel('Allow new registrations').check();
			await page.getByRole('button', { name: 'Save', exact: true }).click();
			await expect(page.getByText('Settings saved.')).toBeVisible({ timeout: 15_000 });
		}
	});
});
