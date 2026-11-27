import { expect, test } from '@playwright/test';
import { registerUser } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';

function textFile(name: string, text: string) {
	const bytes = new TextEncoder().encode(text);
	const b64 = btoa(String.fromCharCode(...bytes));
	return { name, mimeType: 'text/plain', buffer: Object.assign(bytes, { toString: () => b64 }) };
}

test.describe('upload page', () => {
	stubGuard();

	test('preview shows the file and remove resets the form', async ({ page, context }) => {
		await registerUser(context, 'e2e-upload');
		const fixture = await loadFixture('a');
		await page.goto('/upload');
		await page.locator('input[type="file"]').first().setInputFiles({
			name: 'a.png',
			mimeType: fixture.mimeType,
			buffer: fixture.bytes
		});
		// Preview renders the picked file (img alt is the file name).
		await expect(page.locator('img[alt="a.png"]')).toBeVisible();
		await expect(page.getByText('a.png')).toBeVisible();
		await expect(page.getByRole('button', { name: 'Upload' })).toBeEnabled();

		await page.getByRole('button', { name: 'Remove' }).click();
		await expect(page.locator('img[alt="a.png"]')).toHaveCount(0);
		await expect(page.getByRole('button', { name: 'Upload' })).toBeDisabled();
	});

	test('unsupported file type shows the 415 message', async ({ page, context }) => {
		await registerUser(context, 'e2e-upload');
		await page.goto('/upload');
		await page
			.locator('input[type="file"]')
			.first()
			.setInputFiles(textFile('notes.txt', 'not an image'));
		await page.getByRole('button', { name: 'Upload' }).click();
		await expect(page.getByText('Unsupported image type')).toBeVisible({ timeout: 15_000 });
	});
});
