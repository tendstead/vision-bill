import { expect, test } from '@playwright/test';
import { uploadViaApi, verifyReceiptViaApi } from '../helpers/api';
import { loginAdmin } from '../helpers/auth';
import { testEnvironment } from '../helpers/environment';
import { loadFixture } from '../helpers/fixtures';
import { setStubScenario, stubGuard } from '../helpers/stub';

async function requestCount(stub: 'VB_E2E_STUB_URL' | 'VB_E2E_OCR_URL'): Promise<number> {
	const response = await fetch(testEnvironment(stub) + '/__requests');
	expect(response.status).toBe(200);
	return (await response.json()).requests.length;
}

test.describe('benchmark input and pipeline selection', () => {
	stubGuard();

	test('pick a verified receipt by search and run only the text model with OCR', async ({
		page,
		context
	}) => {
		test.setTimeout(120_000);
		test.skip(testEnvironment('VB_E2E_POLICY') === 'off', 'OCR mode requires the OCR stub');
		await loginAdmin(context);
		const fixture = await loadFixture('b');
		const { body } = await uploadViaApi(context, 'receipt.png', fixture.bytes);
		const id = String(body.receipt_id);
		await verifyReceiptViaApi(context, id);
		await setStubScenario('complete');
		const beforeOcr = await requestCount('VB_E2E_OCR_URL');

		await page.goto('/benchmarks');
		await page
			.getByRole('searchbox', { name: 'Find a receipt by merchant or receipt number' })
			.fill(fixture.expected.merchant_name);
		const row = page
			.getByRole('listitem')
			.filter({ has: page.locator('a[href="/receipts/' + id + '"]') });
		await row.getByRole('checkbox', { name: new RegExp(fixture.expected.merchant_name) }).check();
		await page.getByRole('button', { name: 'Preview human references' }).click();
		await page.getByRole('checkbox', { name: /I checked these current receipts/ }).check();
		await page.getByRole('radio', { name: 'OCR + LLM', exact: true }).check();
		await page.getByRole('checkbox', { name: 'e2e-text', exact: true }).check();
		await page.getByRole('checkbox', { name: 'Include categories and tags' }).uncheck();
		await page.getByRole('button', { name: 'Create benchmark' }).click();
		await page.getByRole('button', { name: 'Confirm and start' }).click();

		await expect(page).toHaveURL(/\/benchmarks\/results\?run=/);
		await expect(page.getByText('1 / 1')).toBeVisible({ timeout: 60_000 });
		await expect(page.getByRole('cell', { name: /e2e-text/ })).toBeVisible();
		await expect(page.getByRole('table', { name: 'Pipeline comparison' })).toHaveCount(0);
		expect(await requestCount('VB_E2E_STUB_URL')).toBe(1);
		expect(await requestCount('VB_E2E_OCR_URL')).toBe(beforeOcr + 1);
	});

	test('confirm a photo, opt in to comparison, and read each model’s result without rerunning it', async ({
		page,
		context
	}) => {
		test.setTimeout(120_000);
		test.skip(testEnvironment('VB_E2E_POLICY') === 'off', 'Comparison requires the OCR stub');
		await loginAdmin(context);
		await setStubScenario('direct_merchant_mismatch');
		const fixture = await loadFixture('a');
		const name = 'Shop counter ' + crypto.randomUUID();
		const beforeOcr = await requestCount('VB_E2E_OCR_URL');

		await page.goto('/benchmarks');
		await page.getByRole('link', { name: 'Upload photos and manage benchmark cases' }).click();
		await page.getByRole('button', { name: 'New case' }).click();
		await page.getByLabel('Case name', { exact: true }).fill(name);
		await page.getByLabel('Receipt photos', { exact: true }).setInputFiles({
			name: 'counter.png',
			mimeType: fixture.mimeType,
			buffer: fixture.bytes
		});
		await page.getByRole('button', { name: 'Create case' }).click();
		await page.getByLabel('Merchant', { exact: true }).fill(fixture.expected.merchant_name);
		await page.getByRole('button', { name: 'Save draft' }).click();
		await expect(page.getByText('Draft saved.', { exact: true })).toBeVisible();
		await page.getByLabel('I checked the expected values against the original photos.').check();
		await page.getByRole('button', { name: 'Confirm reference' }).click();
		await page.getByRole('link', { name: 'Use in benchmark' }).click();
		await expect(page.getByRole('checkbox', { name: new RegExp(name) })).toBeChecked();
		await expect(page.getByRole('radio', { name: 'Image directly to LLM' })).toBeChecked();
		expect(await requestCount('VB_E2E_STUB_URL')).toBe(0);
		expect(await requestCount('VB_E2E_OCR_URL')).toBe(beforeOcr);

		await page.getByRole('radio', { name: 'Compare both' }).check();
		await page.getByRole('checkbox', { name: 'e2e-vision', exact: true }).check();
		await page.getByRole('checkbox', { name: /e2e-text/ }).check();
		await page.getByRole('checkbox', { name: 'Include categories and tags' }).uncheck();
		await page.getByRole('button', { name: 'Create benchmark' }).click();
		await page.getByRole('button', { name: 'Confirm and start' }).click();
		await expect(page).toHaveURL(/\/benchmarks\/results\?run=/);
		const table = page.getByRole('table', { name: 'Pipeline comparison' });
		const vision = table.getByRole('row', { name: /e2e-vision/ });
		await expect(vision).toContainText('+100.00 pp', { timeout: 60_000 });
		await expect(table.getByRole('row', { name: /e2e-text/ })).toContainText('Not applicable');

		await page.reload();
		await expect(vision).toContainText('+100.00 pp');
		await page.getByRole('button', { name: 'Load individual results' }).click();
		const cases = page.getByRole('table', { name: 'Case comparison' });
		await expect(cases.getByRole('row', { name: new RegExp('e2e-vision.*' + name) })).toContainText(
			'+100.00 pp'
		);
		expect(await requestCount('VB_E2E_STUB_URL')).toBe(3);
		expect(await requestCount('VB_E2E_OCR_URL')).toBe(beforeOcr + 2);
	});
});
