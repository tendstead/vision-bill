import { expect, test } from '@playwright/test';
import { loginAdmin } from '../helpers/auth';
import { getSettings, putSettings } from '../helpers/api';

test.describe('settings (admin)', () => {
	test('loads with source labels and editable environment defaults', async ({ page, context }) => {
		await loginAdmin(context);
		await page.goto('/settings');
		await expect(page.getByRole('group', { name: 'Receipt formatter', exact: true })).toBeVisible();
		await expect(page.getByText('Environment default').first()).toBeVisible();
		await expect(page.getByText('Config').first()).toBeVisible();
		await expect(page.getByLabel('Provider')).toBeEnabled();
		await expect(page.getByLabel('Host')).toBeEnabled();
		await expect(page.getByLabel('Default model')).toBeEnabled();
		await expect(page.getByLabel('Temperature')).toBeEnabled();
	});

	test('opens the installed OCR recovery guide and returns to Settings', async ({
		page,
		context
	}) => {
		await loginAdmin(context);
		await page.goto('/settings');
		await page.getByRole('link', { name: 'GPU setup and recovery guide' }).click();
		await expect(page).toHaveURL(/\/settings\/ocr-guide\/?$/);
		await expect(page.getByRole('heading', { name: 'Recover a failed setup' })).toBeVisible();
		const manual = await page
			.getByRole('link', { name: 'Source installation (.md)' })
			.evaluate((link) => (link as HTMLAnchorElement).href);
		const response = await page.request.get(manual);
		expect(response.ok()).toBe(true);
		expect(await response.text()).toContain('## Dedicated OCR from source');
		await page.getByRole('article').getByRole('link', { name: 'Settings' }).click();
		await expect(page).toHaveURL(/\/settings\/?$/);
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
