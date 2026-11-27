import { expect, test } from '@playwright/test';
import { loginAdmin, registerUser, uniqueUsername } from '../helpers/auth';
import { putSettings } from '../helpers/api';

// Auth UI flows against the real stack. Each test registers a fresh user
// via the API (shared cookie jar) or drives the form directly.
test.describe('login', () => {
	test('register via form lands on the app, logout returns to login', async ({ page }) => {
		await page.goto('/login');
		await expect(page.getByRole('heading', { name: 'Sign in' })).toBeVisible();

		// Switch to registration mode.
		await page.getByRole('button', { name: 'Create one' }).click();
		const username = `e2e-pw-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
		await page.getByLabel('Username').fill(username);
		await page.getByLabel('Password').fill('e2e-pw-12345');
		await page.getByRole('button', { name: 'Create account' }).click();

		// After registration the user is signed in (cookie set) and routed in.
		await page.waitForURL((url) => !url.pathname.startsWith('/login'));

		// Sign out via the nav.
		await page.getByRole('button', { name: /Sign out/i }).click();
		await page.waitForURL('**/login');
	});

	test('wrong password shows an error and does not sign in', async ({ page, context }) => {
		// Register a real user via the API first, then clear the session
		// cookie: the layout guard bounces any signed-in browser away from
		// /login, which would defeat the point of this test.
		const username = `e2e-pwbad-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
		const register = await context.request.post('/api/v1/auth/register', {
			data: { username, password: 'real-pw-12345' }
		});
		expect(register.status()).toBe(201);
		await context.clearCookies();

		await page.goto('/login');
		await page.getByLabel('Username').fill(username);
		await page.getByLabel('Password').fill('definitely-wrong');
		await page.getByRole('button', { name: 'Sign in' }).click();

		// Still on /login with the API's 401 detail shown.
		await expect(page).toHaveURL(/\/login/);
		await expect(page.getByText('Invalid username or password')).toBeVisible();
	});

	test('duplicate username shows the conflict message', async ({ page, context }) => {
		const creds = await registerUser(context, 'e2e-login');
		// Clear the session cookie: the layout guard bounces signed-in users
		// away from /login, which would hide the "Create one" button.
		await context.clearCookies();
		await page.goto('/login');
		await page.getByRole('button', { name: 'Create one' }).click();
		await page.getByLabel('Username').fill(creds.username);
		await page.getByLabel('Password').fill('e2e-pw-12345');
		await page.getByRole('button', { name: 'Create account' }).click();
		await expect(page.getByRole('alert')).toContainText('Username is already taken');
	});

	test('unknown user shows the error message', async ({ page }) => {
		await page.goto('/login');
		await page.getByLabel('Username').fill('no-such-user-e2e');
		await page.getByLabel('Password').fill('e2e-pw-12345');
		await page.getByRole('button', { name: 'Sign in' }).click();
		await expect(page.getByRole('alert')).toContainText('Invalid username or password');
	});

	test('registration closed hides the register link and blocks the API', async ({
		page,
		context,
		browser
	}) => {
		const adminContext = await browser.newContext();
		await loginAdmin(adminContext);

		try {
			await putSettings(adminContext, { allow_registration: false });

			const response = await context.request.post('/api/v1/auth/register', {
				data: { username: uniqueUsername('e2e-closed'), password: 'e2e-pw-12345' }
			});
			expect(response.status()).toBe(403);

			await page.goto('/login');
			await expect(page.getByRole('button', { name: 'Create one' })).toHaveCount(0);
		} finally {
			await putSettings(adminContext, { allow_registration: true });
		}
		await adminContext.close();
	});
});
