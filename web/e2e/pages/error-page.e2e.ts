import { expect, test } from '@playwright/test';
import { registerUser } from '../helpers/auth';

test.describe('error page', () => {
	test('unknown deep route shows the 404 page', async ({ page, context }) => {
		await registerUser(context, 'e2e-error');
		await page.goto('/definitely/not/a/route');
		await expect(page.getByRole('heading', { name: 'Page not found' })).toBeVisible();
		await expect(page.getByRole('link', { name: 'Return to dashboard' })).toBeVisible();
	});
});
