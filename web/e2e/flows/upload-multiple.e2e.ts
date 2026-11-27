import { expect, test } from '@playwright/test';
import { registerUser } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { setStubMode, stubGuard } from '../helpers/stub';

test.describe('multiple photos of one receipt', () => {
	stubGuard();

	test('starts immediately, survives reload, removes verified overlap and retains photos', async ({
		page,
		context
	}) => {
		await registerUser(context, 'multi-photo');
		await setStubMode('slow');
		const fixture = await loadFixture('a');
		await page.goto('/upload');
		await page
			.locator('input[type="file"]')
			.first()
			.setInputFiles([
				{ name: 'top.png', mimeType: fixture.mimeType, buffer: fixture.bytes },
				{ name: 'bottom.png', mimeType: fixture.mimeType, buffer: fixture.bytes }
			]);
		await page.getByRole('button', { name: 'Move photo up' }).nth(1).click();
		await expect(page.getByText('1. bottom.png')).toBeVisible();
		await page.getByRole('button', { name: 'Upload', exact: true }).click();
		await expect(page).toHaveURL(/\/upload\?job=[0-9a-f-]+/);
		const jobUrl = page.url();
		await page.reload();
		await expect(page.getByText('Your receipt is ready', { exact: true })).toBeVisible({
			timeout: 20_000
		});
		const imageId = new URL(jobUrl).searchParams.get('job');
		const job = await (await context.request.get(`/api/v1/images/${imageId}`)).json();
		expect(job.original_filename).toBe('bottom.png');
		expect(job.additional_images).toHaveLength(1);
		const receipt = await (await context.request.get(`/api/v1/receipts/${job.receipt_id}`)).json();
		expect(receipt.line_items).toHaveLength(fixture.expected.line_items.length);
		await page.getByRole('link', { name: 'Open and review receipt' }).click();
		await page.getByRole('button', { name: 'Photo 2', exact: true }).click();
		await expect(page.locator('img[src*="part=1"]')).toBeVisible();
		const verified = await context.request.post(`/api/v1/receipts/${job.receipt_id}/verify`);
		expect(verified.ok()).toBeTruthy();
		expect((await context.request.get(`/api/v1/images/${imageId}/file?part=1`)).ok()).toBeTruthy();
		await page.goto(jobUrl);
		await expect(page.getByRole('link', { name: 'Open and review receipt' })).toBeVisible();
	});

	test('unreadable photos show a persistent error', async ({ page, context }) => {
		await registerUser(context, 'unreadable-photo');
		await setStubMode('unreadable');
		const fixture = await loadFixture('a');
		await page.goto('/upload');
		await page
			.locator('input[type="file"]')
			.first()
			.setInputFiles({ name: 'blurry.png', mimeType: fixture.mimeType, buffer: fixture.bytes });
		await page.getByRole('button', { name: 'Upload', exact: true }).click();
		await expect(page).toHaveURL(/\/upload\?job=/);
		await page.reload();
		await expect(
			page.getByText('The receipt could not be read reliably', { exact: true })
		).toBeVisible();
		await expect(page.getByText('Photo is too blurry', { exact: true })).toBeVisible();
	});
});
