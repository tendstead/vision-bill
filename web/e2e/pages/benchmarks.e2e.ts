import { expect, test } from '@playwright/test';
import { loginAdmin } from '../helpers/auth';

test.describe('benchmarks create form (admin)', () => {
	test('lists the model and disables create without a selection', async ({ page, context }) => {
		await loginAdmin(context);
		await page.goto('/benchmarks');
		await expect(page.getByRole('checkbox', { name: 'e2e-vision' })).toBeChecked();
		await page.getByRole('checkbox', { name: 'e2e-vision' }).uncheck();
		await expect(page.getByRole('button', { name: 'Create benchmark' })).toBeDisabled();
	});

	test('invalid receipt id shows an error and stays on the page', async ({ page, context }) => {
		await loginAdmin(context);
		await page.goto('/benchmarks');
		await page.getByPlaceholder('Comma-separated receipt UUIDs').fill('not-a-uuid');
		await page.getByRole('button', { name: 'Create benchmark' }).first().click();
		await page.getByRole('button', { name: 'Create benchmark' }).nth(1).click();
		// The 422 detail (pydantic UUID validation) is surfaced to the user.
		await expect(page.getByRole('alert')).toContainText(/UUID/i);
		await expect(page).toHaveURL(/\/benchmarks$/);
	});
});
