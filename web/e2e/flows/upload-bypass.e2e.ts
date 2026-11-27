import { expect, test } from '@playwright/test';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';
import { uploadFixture } from '../helpers/ui';

test.describe('bypass review', () => {
	stubGuard();

	test('skip review saves directly as verified', async ({ page, context }) => {
		test.setTimeout(120_000);
		const fixture = await loadFixture('a');
		await uploadFixture(page, context, fixture, { bypass: true });
		await expect(page.getByText('Verified', { exact: true })).toBeVisible();
		await expect(page.getByRole('button', { name: 'Verify' })).toHaveCount(0);

		// Counted in the verified-only dashboard stats.
		await page.goto('/');
		await expect(page.getByText('€3.86')).toBeVisible();
	});
});
