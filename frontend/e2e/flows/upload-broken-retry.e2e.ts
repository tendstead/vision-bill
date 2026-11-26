import { expect, test } from '@playwright/test';
import { registerUser } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { setStubMode, stubGuard } from '../helpers/stub';
import { awaitSubmission } from '../helpers/api';
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
		await expect(page).toHaveURL(/\/upload\?job=[a-f0-9-]+/);
		const id = new URL(page.url()).searchParams.get('job')!;
		const job = await awaitSubmission(context, id, 120_000);
		expect(job.status).toBe('failed');
		expect(job.attempt_count).toBe(3);

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
