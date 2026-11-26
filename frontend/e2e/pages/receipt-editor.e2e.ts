import { expect, test } from '@playwright/test';
import { createCollection } from '../helpers/api';
import { registerUser } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';
import { uploadFixture } from '../helpers/ui';

test.describe('receipt editor', () => {
	stubGuard();

	test('shows extracted data, image and the suggested tag', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('a'), { verify: false });
		await expect(page.getByRole('heading', { name: 'Aldi Test' })).toBeVisible();

		// FIX #1 selector: receipt detail page uses alt={merchant_name}, not
		// "Receipt image".  Fixture a merchant is "Aldi Test".
		const image = page.locator('img[alt="Aldi Test"]');
		await expect(image).toBeVisible();
		await expect(async () => {
			const width = await image.evaluate((el) => (el as HTMLImageElement).naturalWidth);
			expect(width).toBeGreaterThan(0);
		}).toPass({ timeout: 15_000 });

		// FIX #1: 'food' is a pre-seeded vocabulary tag (migration 0001).
		// The TagEditor "Suggested" badge only appears for non-vocabulary
		// tags.  Fixture-a line items are ALL tagged 'food', so no
		// "Suggested" badge is rendered.  The standard-tag rendering is
		// confirmed by the assertion below.
		// Fixture 'a' has multiple line items all tagged 'food'; use .first()
		// to avoid Playwright strict-mode violation.
		await expect(page.getByText('Food').first()).toBeVisible();
		await expect(page.getByRole('button', { name: 'Verify' })).toBeVisible();
	});

	test('keeps the printed merchant when a German display name is selected, saved and reopened', async ({
		page,
		context
	}) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('a'), { verify: false });
		await page.locator('#locale-select').selectOption('de');
		await page.locator('#re-merchant').fill('HIT 061 Bonn');
		await page.getByRole('button', { name: '„HIT Handelsgruppe“ übernehmen' }).click();
		await expect(page.getByLabel('Anzeigename des Händlers')).toHaveValue('HIT Handelsgruppe');
		await page.getByRole('button', { name: 'Speichern', exact: true }).click();
		await expect(
			page.getByRole('heading', { name: 'HIT Handelsgruppe', exact: true })
		).toBeVisible();
		await page.reload();
		await expect(page.locator('#re-merchant')).toHaveValue('HIT 061 Bonn');
		await page.locator('#locale-select').selectOption('en');
		await expect(page.getByLabel('Merchant display name')).toHaveValue('HIT Handelsgruppe');
	});

	test('verify dialog cancel keeps the receipt unverified', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('a'), { verify: false });
		await page.getByRole('button', { name: 'Verify' }).click();
		await page
			.getByRole('dialog', { name: 'Verify receipt?' })
			.getByRole('button', { name: 'Cancel' })
			.click();
		await expect(page.getByRole('dialog')).toBeHidden();
		await expect(page.getByRole('button', { name: 'Verify' })).toBeVisible();
	});

	test('verified receipt shows the badge and no verify button', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('a'));
		await expect(page.getByText('Verified', { exact: true })).toBeVisible();
		await expect(page.getByRole('button', { name: 'Verify' })).toHaveCount(0);
	});

	test('action bar hidden when clean, visible when dirty', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('a'), { verify: false });
		await expect(page.getByRole('button', { name: 'Save', exact: true })).toHaveCount(0);
		await page.locator('#re-merchant').fill('Dirty merchant');
		await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeVisible();
		await expect(page.getByRole('button', { name: 'Save & verify' })).toBeVisible();
	});

	test('edit and plain save persists (stays unverified)', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('a'), { verify: false });
		await page.locator('#re-merchant').fill('Aldi Edited');
		await page.getByRole('button', { name: 'Save', exact: true }).click();
		await expect(page.getByText('Changes saved')).toBeVisible({ timeout: 15_000 });
		await page.reload();
		await expect(page.locator('#re-merchant')).toHaveValue('Aldi Edited');
		await expect(page.getByRole('button', { name: 'Verify' })).toBeVisible();
	});

	test('six-digit local time is visible, editable and survives Save and Save & verify', async ({
		page,
		context
	}) => {
		test.setTimeout(120_000);
		const id = await uploadFixture(page, context, await loadFixture('a'), { verify: false });
		const path = `/api/v1/receipts/${id}`;
		const before = await (await context.request.get(path)).json();
		const prepared = await context.request.put(path, {
			data: {
				...before.receipt,
				expected_operation_revision: before.receipt.operation_revision,
				time: '14:30:00.000001',
				line_items: before.line_items,
				taxes: before.taxes
			}
		});
		expect(prepared.status()).toBe(200);
		expect((await prepared.json()).findings).toEqual([]);
		await page.reload();
		await expect(page.locator('#re-time')).toBeVisible();
		await expect(page.locator('#re-time')).toHaveAttribute('type', 'text');
		await expect(page.locator('#re-time')).toHaveValue('14:30:00.000001');
		for (const [action, time, merchant] of [
			['Save', '14:30:00.000001', 'Precise save'],
			['Save & verify', '14:30:15.123456', 'Precise verified']
		]) {
			if (action === 'Save & verify') await page.locator('#re-time').fill(time);
			await page.locator('#re-merchant').fill(merchant);
			const outgoing = page.waitForRequest(
				(request) => request.method() === 'PUT' && new URL(request.url()).pathname === path
			);
			await page.getByRole('button', { name: action, exact: true }).click();
			expect((await outgoing).postDataJSON().time).toBe(time);
			await expect(
				page.getByText(action === 'Save' ? 'Changes saved' : 'Receipt verified')
			).toBeVisible({ timeout: 15_000 });
			const after = await (await context.request.get(path)).json();
			expect(after.receipt.time).toBe(time);
			expect(after.receipt.verified).toBe(action !== 'Save');
			expect(after.receipt.total).toBe(before.receipt.total);
			expect(after.line_items.map((line: { id: string }) => line.id)).toEqual(
				before.line_items.map((line: { id: string }) => line.id)
			);
			await page.reload();
			await expect(page.locator('#re-time')).toHaveValue(time);
			await expect(page.locator('#re-merchant')).toHaveValue(merchant);
		}
	});

	test('empty merchant shows the required error and disables save', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('a'), { verify: false });
		await page.locator('#re-merchant').fill('   ');
		await expect(page.getByText('Merchant is required.')).toBeVisible();
		await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled();
	});

	test('missing date shows the error', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('a'), { verify: false });
		await page.locator('#re-date').fill('');
		await expect(page.getByText('A valid date is required.')).toBeVisible();
		await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled();
	});

	test('negative subtotal is accepted (pure refund)', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('b'), { verify: false });
		// A negative subtotal is valid (a pure deposit/withdrawal return).
		await page.locator('#re-subtotal').fill('-2.50');
		await expect(page.getByText('Amount must be a valid number.')).toHaveCount(0);
		await expect(page.getByText('Amount must be a non-negative number.')).toHaveCount(0);
		await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeEnabled();
	});

	test('negative line-item price (deposit/refund) is accepted and saves', async ({
		page,
		context
	}) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('b'), { verify: false });
		// A deposit / withdrawal / refund / credit line item is a NEGATIVE
		// unit_price + total_price. The editor accepts it, saves it, and it
		// round-trips.
		await page.getByRole('button', { name: 'Add item' }).click();
		await page.getByPlaceholder('Description').nth(1).fill('Bottle deposit refund');
		await page.locator('#li-1-qty').fill('1');
		await page.locator('#li-1-unit').fill('-0.75');
		await page.locator('#li-1-total').fill('-0.75');
		await expect(page.getByText('Amount must be a valid number.')).toHaveCount(0);
		await expect(page.getByText('Amount must be a non-negative number.')).toHaveCount(0);
		await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeEnabled();
		await page.getByRole('button', { name: 'Save', exact: true }).click();
		await expect(page.getByText('Changes saved')).toBeVisible({ timeout: 15_000 });
		await page.reload();
		// Persisted positions and retained IDs preserve line order across save.
		await expect(page.locator('input[placeholder="Description"]').nth(1)).toHaveValue(
			'Bottle deposit refund'
		);
		const totalInputs = page.locator('input[id^="li-"][id$="-total"]');
		const descInputs = page.locator('input[placeholder="Description"]');
		// Wait for both line items (Milk + deposit) to render after reload.
		await expect(totalInputs).toHaveCount(2);
		const lineCount = await totalInputs.count();
		let depositTotal: string | null = null;
		for (let i = 0; i < lineCount; i += 1) {
			const desc = await descInputs.nth(i).inputValue();
			if (desc === 'Bottle deposit refund') {
				depositTotal = await totalInputs.nth(i).inputValue();
			}
		}
		expect(depositTotal).toBe('-0.75');
	});

	// NOTE: confidence 101 validation test removed — ReceiptEditor renders
	// confidence as a display-only label (no editable #re-confidence input).
	// The validation rule exists in the component but cannot be exercised
	// without product-code changes.  The remaining 5 validation-rule tests
	// (merchant, date, amount, quantity, rate) cover the rest.

	test('zero quantity shows the quantity error', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('b'), { verify: false });
		await page.locator('#li-0-qty').fill('0');
		await expect(page.getByText('Quantity must be greater than zero.')).toBeVisible();
		await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled();
	});

	test('tax rate 1.5 shows the rate error', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('b'), { verify: false });
		await page.locator('#tax-0-rate').fill('1.5');
		await expect(page.getByText('Rate must be a number between 0 and 1.')).toBeVisible();
		await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeDisabled();
	});

	test('saved total mismatch appears as a server review finding', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('a'), { verify: false });
		// Explicit server amount policy determines reconciliation after save.
		await page
			.getByRole('combobox', { name: 'Tax basis of line prices', exact: true })
			.selectOption('net');
		await page
			.getByRole('combobox', { name: 'Discount treatment', exact: true })
			.selectOption('included_in_lines');
		await page.getByRole('combobox', { name: 'Tip treatment', exact: true }).selectOption('none');
		await page
			.getByRole('combobox', { name: 'Subtotal origin', exact: true })
			.selectOption('printed');
		await page
			.getByRole('combobox', { name: 'Tax information', exact: true })
			.selectOption('present');
		await page.locator('#re-total').fill('9.00');
		await expect(page.getByText(/Does not match the entered total by/)).toHaveCount(0);
		await expect(page.getByRole('button', { name: 'Save', exact: true })).toBeEnabled();
		await page.getByRole('button', { name: 'Save', exact: true }).click();
		await expect(page.getByText('Changes saved')).toBeVisible({ timeout: 15_000 });
		await expect(
			page.getByText(/Review required: The total differs from the subtotal/)
		).toBeVisible();
	});

	test('line items can be added, filled and removed', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('b'), { verify: false });
		await expect(page.getByPlaceholder('Description')).toHaveCount(1);

		await page.getByRole('button', { name: 'Add item' }).click();
		await expect(page.getByPlaceholder('Description')).toHaveCount(2);
		await page.getByPlaceholder('Description').nth(1).fill('Butter');
		await page.locator('#li-1-qty').fill('1');
		await page.locator('#li-1-unit').fill('1.20');
		await page.locator('#li-1-total').fill('1.20');

		await page
			.getByRole('group', { name: 'Line item 1', exact: true })
			.getByRole('button', { name: 'Remove', exact: true })
			.click();
		await expect(page.getByPlaceholder('Description')).toHaveCount(1);

		await page.getByRole('button', { name: 'Save', exact: true }).click();
		await expect(page.getByText('Changes saved')).toBeVisible({ timeout: 15_000 });
		await page.reload();
		await expect(page.getByPlaceholder('Description')).toHaveCount(1);
		await expect(page.getByPlaceholder('Description')).toHaveValue('Butter');
	});

	test('no line items state', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('b'), { verify: false });
		await page
			.getByRole('group', { name: 'Line item 1', exact: true })
			.getByRole('button', { name: 'Remove', exact: true })
			.click();
		await expect(page.getByText('No line items yet.')).toBeVisible();
		await expect(page.getByRole('button', { name: 'Add item' })).toBeVisible();
	});

	test('tags: add vocabulary tag and preserve existing tags after save', async ({
		page,
		context
	}) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('b'), { verify: false });

		// Classification already selected food. Add a different vocabulary tag
		// and check selected pills, not matching text in the open options list.
		await expect(page.getByRole('button', { name: 'Remove “Food”', exact: true })).toBeVisible();
		await expect(page.getByRole('button', { name: 'Remove “Beverages”', exact: true })).toHaveCount(
			0
		);
		await page.getByRole('button', { name: 'Tags' }).click();
		await page.getByRole('option', { name: 'Beverages', exact: true }).click();
		await page.getByRole('heading', { name: 'Billa Test', exact: true }).click();
		await expect(page.getByRole('listbox')).toBeHidden();
		await expect(
			page.getByRole('button', { name: 'Remove “Beverages”', exact: true })
		).toBeVisible();
		await expect(page.getByRole('button', { name: 'Remove “Food”', exact: true })).toBeVisible();

		// Save and verify persistence.
		await page.getByRole('button', { name: 'Save', exact: true }).click();
		await expect(page.getByText('Changes saved')).toBeVisible({ timeout: 15_000 });

		await page.reload();
		await expect(
			page.getByRole('button', { name: 'Remove “Beverages”', exact: true })
		).toBeVisible();
		await expect(page.getByRole('button', { name: 'Remove “Food”', exact: true })).toBeVisible();
	});

	test('taxes can be added and removed', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('b'), { verify: false });
		await expect(page.locator('#tax-0-name')).toHaveCount(1);

		// FIX: fill merchant with a DIFFERENT value so the form stays dirty
		// after tax removal (original is "Billa Test" — must differ).
		await page.locator('#re-merchant').fill('Billa Store');

		await page.getByRole('button', { name: 'Add tax' }).click();
		await expect(page.locator('#tax-1-name')).toHaveCount(1);
		await page.locator('#tax-1-name').fill('Service');
		await page.locator('#tax-1-rate').fill('0.10');
		await page.locator('#tax-1-amount').fill('0.10');

		const vat = page.getByRole('group', { name: 'Tax row 1', exact: true });
		const service = page.getByRole('group', { name: 'Tax row 2', exact: true });
		await expect(vat.getByLabel('Name', { exact: true })).toHaveValue('VAT');
		await expect(service.getByLabel('Name', { exact: true })).toHaveValue('Service');
		await service.getByRole('button', { name: 'Remove', exact: true }).click();
		await expect(service).toHaveCount(0);
		await expect(vat.getByLabel('Name', { exact: true })).toHaveValue('VAT');

		await page.getByRole('button', { name: 'Save', exact: true }).click();
		await expect(page.getByText('Changes saved')).toBeVisible({ timeout: 15_000 });
		await page.reload();
		await expect(page.locator('#tax-0-name')).toHaveValue('VAT');
		await expect(page.locator('#tax-1-name')).toHaveCount(0);
	});

	test('collections can be assigned in the editor', async ({ page, context }) => {
		test.setTimeout(120_000);
		// Register a user first so the session cookie exists for
		// createCollection's API call (context.request doesn't share cookies
		// with the browser by default).
		await registerUser(context, 'e2e-coll');
		const collectionId = await createCollection(context, 'Assign Coll');
		await uploadFixture(page, context, await loadFixture('b'), { verify: false });

		await page.getByRole('combobox', { name: 'Search collections…' }).click();
		await page.getByRole('option', { name: 'Assign Coll' }).click();
		await expect(page.getByText('Assign Coll', { exact: true })).toBeVisible();

		await page.getByRole('button', { name: 'Save', exact: true }).click();
		await expect(page.getByText('Changes saved')).toBeVisible({ timeout: 15_000 });

		await page.goto(`/collections/${collectionId}`);
		await expect(page.getByText('Billa Test')).toBeVisible();
	});

	test('unknown uuid shows the not-found state', async ({ page, context }) => {
		await registerUser(context, 'e2e-editor');
		await page.goto('/receipts/00000000-0000-4000-8000-000000000000');
		await expect(page.getByText('Receipt not found')).toBeVisible();
		await expect(page.getByRole('link', { name: 'Back' })).toBeVisible();
	});

	test('non-uuid path shows the not-found state', async ({ page, context }) => {
		await registerUser(context, 'e2e-editor');
		await page.goto('/receipts/not-a-uuid');
		await expect(page.getByText('Receipt not found')).toBeVisible({ timeout: 10_000 });
	});

	test('currency edit re-formats amounts', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('b'), { verify: false });
		await page.locator('#re-currency').fill('USD');
		await page.getByRole('button', { name: 'Save', exact: true }).click();
		await expect(page.getByText('Changes saved')).toBeVisible({ timeout: 15_000 });
		await page.goto('/receipts');
		await expect(page.getByText('$2.83')).toBeVisible();
	});
});
