import { expect, test } from '@playwright/test';
import { registerUser } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { setStubMode, stubGuard } from '../helpers/stub';
import { confirmDialog } from '../helpers/ui';

test.describe('persistently broken model', () => {
	stubGuard();

	test('broken model -> upload failed -> failed image -> delete', async ({ page, context }) => {
		test.setTimeout(150_000);
		await setStubMode('broken');
		await registerUser(context, 'e2e-broken');
		const fixture = await loadFixture('a');

		await page.goto('/upload');
		await page.locator('input[type="file"]').first().setInputFiles({
			name: 'a.png',
			mimeType: fixture.mimeType,
			buffer: fixture.bytes
		});
		await page.getByRole('button', { name: 'Upload' }).click();
		// The provider IS reachable but returns invalid JSON: self-correction
		// exhausts its retries, the image is marked failed, the API 500s.
		await expect(page.getByText('Upload failed')).toBeVisible({ timeout: 30_000 });

		await page.goto('/queue');
		await expect(page.getByText('a.png')).toBeVisible();
		await expect(page.getByText('Failed', { exact: true })).toBeVisible();

		// The failed image can be cleaned up.
		await page.getByRole('button', { name: 'Remove' }).click();
		await confirmDialog(page, 'Delete image?', 'Delete');
		await expect(page.getByText('Image deleted')).toBeVisible({ timeout: 15_000 });
		await expect(page.getByText('The queue is empty')).toBeVisible();
	});
});
