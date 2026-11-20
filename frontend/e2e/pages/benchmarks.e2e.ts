import { expect, test } from '@playwright/test';
import { loginAdmin } from '../helpers/auth';

test.describe('benchmarks create form (admin)', () => {
	test('lists the model and disables create without a selection', async ({ page, context }) => {
		await loginAdmin(context);
		const capabilityResponse = await context.request.get('/api/v1/benchmarks/capabilities');
		expect(capabilityResponse.status()).toBe(200);
		const capabilities = await capabilityResponse.json();
		await page.goto('/benchmarks');
		await expect(page.getByRole('checkbox', { name: 'e2e-vision' })).toBeChecked();
		const models = page
			.getByRole('group', {
				name: capabilities.ocr_enabled ? 'OCR text formatter models' : 'Vision models',
				exact: true
			})
			.getByRole('checkbox');
		for (const model of await models.all()) {
			await model.uncheck();
			await expect(model).not.toBeChecked();
		}
		await expect(page.getByRole('button', { name: 'Create benchmark' })).toBeDisabled();
	});

	test('invalid receipt id shows an error and stays on the page', async ({ page, context }) => {
		await loginAdmin(context);
		const before = await context.request.get('/api/v1/benchmarks');
		expect(before.status()).toBe(200);
		const beforeIds = ((await before.json()) as { id: string }[]).map((run) => run.id);
		await page.goto('/benchmarks');
		await page.getByPlaceholder('Comma-separated receipt UUIDs').fill('not-a-uuid');
		await page.getByRole('button', { name: 'Create benchmark' }).first().click();
		const validation = page.waitForResponse(
			(response) =>
				response.url().endsWith('/api/v1/benchmarks') && response.request().method() === 'POST'
		);
		await page.getByRole('button', { name: 'Confirm and start', exact: true }).click();
		expect((await validation).status()).toBe(422);
		// The 422 detail (pydantic UUID validation) is surfaced to the user.
		await expect(page.getByRole('alert')).toContainText(/UUID/i);
		await expect(page).toHaveURL(/\/benchmarks$/);
		const after = await context.request.get('/api/v1/benchmarks');
		expect(after.status()).toBe(200);
		expect(((await after.json()) as { id: string }[]).map((run) => run.id)).toEqual(beforeIds);
	});
});
