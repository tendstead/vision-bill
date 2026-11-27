import { expect, test } from '@playwright/test';
import { registerUser } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';
import { uploadFixture } from '../helpers/ui';

// End-to-end smoke test for the collections feature, run against an
// already-running vision-bill app (see playwright.config.ts for the base URL).
// A fresh user is registered per run so the suite is isolated from any
// existing data; registration is the only auth path needed (it is open by
// default).
test.describe('collections', () => {
	stubGuard();

	test('create a collection, open it, and see the statistics filter', async ({ page, context }) => {
		// Register a fresh, isolated user. context.request shares the browser
		// context's cookie jar, so the vb_session Set-Cookie from this response
		// is stored automatically and sent on every subsequent navigation — no
		// manual cookie transfer is needed.
		const username = `e2e-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
		const register = await context.request.post('/api/v1/auth/register', {
			data: { username, password: 'e2e-password-123' }
		});
		expect(register.status()).toBe(201);

		// Create a collection via the overview page dialog.
		await page.goto('/collections');
		await page.getByRole('button', { name: /New collection/i }).click();
		await page.getByLabel(/Name/i).fill('Berlin Trip');
		await page.getByRole('button', { name: /Save/i }).click();
		await expect(page.getByText('Berlin Trip').first()).toBeVisible();

		// Open the collection and confirm the (empty) receipts section renders.
		await page.getByRole('link', { name: /Berlin Trip/i }).click();
		await expect(page.getByText(/Receipts|No receipts in this collection/i).first()).toBeVisible();

		// Modify the collection (edit dialog → PATCH). Regression: an
		// unreferenced $1 placeholder in the UPDATE SQL made every PATCH
		// return 500 (IndeterminateDatatypeError) right after create+modify.
		await page.getByRole('button', { name: /Edit collection/i }).click();
		await page.getByRole('textbox', { name: /Name/i }).fill('Berlin Trip 2026');
		await page.getByRole('button', { name: /^Save$/i }).click();
		await expect(page.getByRole('heading', { name: 'Berlin Trip 2026' })).toBeVisible();

		// The statistics page exposes the collection filter.
		await page.goto('/statistics');
		await expect(page.getByText(/Collection/).first()).toBeVisible();
	});

	test('create with dates and a preset color', async ({ page, context }) => {
		const username = `e2e-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
		const register = await context.request.post('/api/v1/auth/register', {
			data: { username, password: 'e2e-password-123' }
		});
		expect(register.status()).toBe(201);

		await page.goto('/collections');
		await page.getByRole('button', { name: /New collection/i }).click();
		const dialog = page.getByRole('dialog', { name: 'New collection' });
		await dialog.getByLabel(/Name/i).fill('Date Trip');
		await dialog.getByLabel(/Start date/i).fill('2026-01-01');
		await dialog.getByLabel(/End date/i).fill('2026-01-31');
		// Pick the teal preset (default is red #EF4444).
		await dialog.getByRole('button', { name: '#14B8A6' }).click();
		await dialog.getByRole('button', { name: /^Save$/ }).click();
		await expect(page.getByText('Date Trip')).toBeVisible();
		await expect(page.getByText('2026-01-01 – 2026-01-31')).toBeVisible();
	});

	test('whitespace-only name shows the client-side error', async ({ page, context }) => {
		const username = `e2e-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
		const register = await context.request.post('/api/v1/auth/register', {
			data: { username, password: 'e2e-password-123' }
		});
		expect(register.status()).toBe(201);

		await page.goto('/collections');
		await page.getByRole('button', { name: /New collection/i }).click();
		const dialog = page.getByRole('dialog', { name: 'New collection' });
		// "   " passes the native `required` check (non-empty) but fails the
		// app's trim check -> inline error, dialog stays open.
		await dialog.getByLabel(/Name/i).fill('   ');
		await dialog.getByRole('button', { name: /^Save$/ }).click();
		await expect(dialog.getByRole('alert')).toContainText('Name');
		await expect(dialog).toBeVisible();
	});

	test('start date after end date shows the date error', async ({ page, context }) => {
		const username = `e2e-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
		const register = await context.request.post('/api/v1/auth/register', {
			data: { username, password: 'e2e-password-123' }
		});
		expect(register.status()).toBe(201);

		await page.goto('/collections');
		await page.getByRole('button', { name: /New collection/i }).click();
		const dialog = page.getByRole('dialog', { name: 'New collection' });
		await dialog.getByLabel(/Name/i).fill('Backwards');
		await dialog.getByLabel(/Start date/i).fill('2026-02-01');
		await dialog.getByLabel(/End date/i).fill('2026-01-01');
		await dialog.getByRole('button', { name: /^Save$/ }).click();
		await expect(dialog.getByRole('alert')).toContainText(
			'Start date must be on or before end date.'
		);
		await expect(dialog).toBeVisible();
	});

	test('delete keeps the receipts', async ({ page, context }) => {
		test.setTimeout(120_000);
		await uploadFixture(page, context, await loadFixture('a'));
		const receiptId = await uploadFixture(page, context, await loadFixture('b'));
		const collectionId = (
			await context.request.post('/api/v1/collections', { data: { name: 'To Delete' } })
		).json() as Promise<{ id: string }>;
		const { id } = await collectionId;
		await context.request.post(`/api/v1/collections/${id}/receipts/${receiptId}`);

		await page.goto('/collections');
		await page.getByRole('button', { name: 'Remove' }).click();
		await page
			.getByRole('dialog', { name: 'Delete collection' })
			.getByRole('button', { name: 'Delete' })
			.click();
		await expect(page.getByText('To Delete')).toHaveCount(0);

		// The receipt survived.
		await page.goto('/receipts');
		await expect(page.getByText('Billa Test')).toBeVisible();
	});

	test('detail page lists receipts and unassign works', async ({ page, context }) => {
		test.setTimeout(150_000);
		await registerUser(context, 'e2e-coll-detail');
		await page.goto('/collections'); // authenticate the page
		const receiptId = await uploadFixture(page, context, await loadFixture('a'));

		// Use page.request so the session cookie is available.
		const collResp = await page.request.post('/api/v1/collections', {
			data: { name: 'Unassign Coll', color: '#2563EB', start_date: null, end_date: null }
		});
		const collBody = (await collResp.json()) as { id: string };
		const id = collBody.id;
		await page.request.post(`/api/v1/collections/${id}/receipts/${receiptId}`);

		await page.goto(`/collections/${id}`);
		await expect(page.getByText('Aldi Test').first()).toBeVisible();
		await page
			.getByRole('button', { name: 'Remove \u201cAldi Test\u201d from this collection' })
			.click();
		await expect(page.getByText('Removed from this collection')).toBeVisible({
			timeout: 15_000
		});
		await expect(page.getByText('No receipts in this collection yet.')).toBeVisible();
	});

	test('random uuid shows the not-found state', async ({ page, context }) => {
		await registerUser(context, 'e2e-collections');
		await page.goto('/collections/00000000-0000-4000-8000-000000000000');
		await expect(page.getByText('Collection not found')).toBeVisible();
		await expect(page.getByRole('link', { name: 'Back' })).toBeVisible();
	});

	test('non-uuid path shows the not-found state', async ({ page, context }) => {
		await registerUser(context, 'e2e-collections');
		await page.goto('/collections/not-a-collection');
		await expect(page.getByText('Collection not found')).toBeVisible({ timeout: 10_000 });
	});
});
