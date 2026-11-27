import { expect, test } from '@playwright/test';
import { registerUser } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';
import { uploadFixture } from '../helpers/ui';

test.describe('multi-user scoping (browser-level)', () => {
	stubGuard();

	test('user B sees none of user A data', async ({ page, context, browser }) => {
		test.setTimeout(180_000);
		// User A: one verified receipt.
		await uploadFixture(page, context, await loadFixture('a'));
		await page.goto('/receipts');
		await expect(page.getByText('Aldi Test')).toBeVisible();

		// User B: fresh browser context + fresh user.
		const contextB = await browser.newContext();
		const pageB = await contextB.newPage();
		await registerUser(contextB, 'e2e-b');

		await pageB.goto('/receipts');
		await expect(pageB.getByText('No receipts yet.')).toBeVisible();

		await pageB.goto('/search');
		await pageB.getByRole('searchbox').fill('Milk');
		await expect(pageB.getByText('No purchases match “Milk”.')).toBeVisible({
			timeout: 15_000
		});

		await pageB.goto('/statistics');
		await expect(pageB.getByText('No statistics yet')).toBeVisible();

		await pageB.goto('/');
		await expect(pageB.getByText('No verified receipts yet')).toBeVisible();

		await pageB.goto('/collections');
		await expect(
			pageB.getByText('No collections yet — create one to start grouping receipts.')
		).toBeVisible();

		await contextB.close();

		// User A's data is untouched.
		await page.goto('/receipts');
		await expect(page.getByText('Aldi Test')).toBeVisible();
	});
});
