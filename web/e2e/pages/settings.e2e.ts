import { expect, test } from '@playwright/test';
import { loginAdmin } from '../helpers/auth';
import { getSettings, putSettings } from '../helpers/api';

test.describe('settings (admin)', () => {
	test('loads with source labels and locked env fields', async ({ page, context }) => {
		await loginAdmin(context);
		await page.goto('/settings');
		await expect(page.getByText('Vision model')).toBeVisible();
		await expect(page.getByText('Environment').first()).toBeVisible();
		await expect(page.getByText('Config').first()).toBeVisible();
		await expect(page.getByLabel('Provider')).toBeDisabled();
		await expect(page.getByLabel('Host')).toBeDisabled();
		await expect(page.getByLabel('Default model')).toBeDisabled();
		await expect(page.getByLabel('Temperature')).toBeEnabled();
	});

	test('temperature edit saves and persists', async ({ page, context }) => {
		test.setTimeout(90_000);
		await loginAdmin(context);
		const before = await getSettings(context);
		const next = Math.round((before.llm.temperature + 0.2) * 100) / 100;

		await page.goto('/settings');
		await page.getByLabel('Temperature').fill(String(next));
		await page.getByRole('button', { name: 'Save', exact: true }).click();
		await expect(page.getByText('Settings saved.')).toBeVisible({ timeout: 15_000 });

		await page.reload();
		await expect(page.getByLabel('Temperature')).toHaveValue(String(next));

		// Restore the original value for the other tests.
		await putSettings(context, { temperature: before.llm.temperature });
	});
});
