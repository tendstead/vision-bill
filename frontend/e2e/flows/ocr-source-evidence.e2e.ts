import { expect, test, type BrowserContext, type Page } from '@playwright/test';
import { loadFixture } from '../helpers/fixtures';
import { setStubScenario, stubGuard } from '../helpers/stub';
import { testEnvironment } from '../helpers/environment';
import { uploadViaApi, verifyReceiptViaApi } from '../helpers/api';
import { registerUser } from '../helpers/auth';

async function ownerId(context: BrowserContext): Promise<string> {
	const response = await context.request.get('/api/v1/auth/me');
	expect(response.status()).toBe(200);
	const { id } = (await response.json()) as { id: string };
	expect(id).toMatch(/^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/);
	return id;
}

function ownerMediaUrl(path: string, owner: string): URL {
	const url = new URL(path, testEnvironment('VB_E2E_BASE_URL'));
	url.searchParams.set('vb_expected_user', owner);
	return url;
}

async function removeOwnedFile(id: string, kind: 'original' | 'thumbnail') {
	const response = await fetch(testEnvironment('VB_E2E_OCR_URL') + '/__files', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ image_id: id, kind })
	});
	expect(response.status).toBe(200);
}
async function assertWorkingView(
	page: Page,
	view: { sha256: string; width: number; height: number },
	boundUrl: URL,
	bbox: number[][]
) {
	const actual = await page.evaluate(async (url) => {
		const response = await fetch(url);
		if (!response.ok) throw new Error(`working PNG ${response.status}`);
		const bytes = await response.arrayBuffer();
		const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
		const digest = await crypto.subtle.digest('SHA-256', bytes);
		return {
			width: bitmap.width,
			height: bitmap.height,
			sha256: Array.from(new Uint8Array(digest))
				.map((b) => b.toString(16).padStart(2, '0'))
				.join('')
		};
	}, boundUrl.href);
	expect(actual).toEqual({ width: view.width, height: view.height, sha256: view.sha256 });
	const image = page.getByRole('img', { name: 'Source view', exact: true });
	await expect(image).toHaveAttribute('src', boundUrl.pathname + boundUrl.search);
	await expect(image).toHaveAttribute('width', String(view.width));
	await expect(image).toHaveAttribute('height', String(view.height));
	const svg = page.getByLabel('Source boxes', { exact: true });
	await expect(svg).toHaveAttribute('viewBox', `0 0 ${view.width} ${view.height}`);
	await expect(svg.locator('polygon')).toHaveAttribute(
		'points',
		bbox.map((p) => p.join(',')).join(' ')
	);
}
async function create(context: BrowserContext) {
	const fixture = await loadFixture('a');
	return (await uploadViaApi(context, 'a.png', fixture.bytes, fixture.mimeType)).body;
}

test.describe('saved OCR evidence and original integrity', () => {
	stubGuard();
	test('draft source is the authenticated working PNG; unknown and display failure have honest states', async ({
		page,
		context
	}) => {
		test.skip(
			testEnvironment('VB_E2E_POLICY') === 'off',
			'Working OCR geometry requires the explicitly enabled stack'
		);
		await registerUser(context, 'ocr-geometry');
		const owner = await ownerId(context);
		await setStubScenario('partial');
		const job = await create(context);
		expect(job.status).toBe('needs_review');
		const draft = await (await context.request.get(`/api/v1/images/${job.id}/draft`)).json();
		const source = draft.field_sources.merchant_name[0];
		const view = draft.source_views.find((v: { view_id: string }) => v.view_id === source.view_id);
		const workingUrl = ownerMediaUrl(view.url, owner);
		expect(source.coordinate_space).toBe('working');
		await page.goto(`/upload?job=${job.id}&draft=1`);
		const field = page.locator('p').filter({ hasText: /^Merchant\s/ });
		await field.getByRole('button', { name: 'Show source', exact: true }).click();
		await assertWorkingView(page, view, workingUrl, source.bbox);
		// The separately selected original never borrows working-coordinate polygons.
		await page.getByRole('button', { name: 'Original photos', exact: true }).click();
		const original = page.getByRole('img', { name: 'Original photos', exact: true });
		const originalUrl = ownerMediaUrl(`/api/v1/images/${job.id}/file`, owner);
		await expect(original).toHaveAttribute('src', originalUrl.pathname + originalUrl.search);
		await expect(original.locator('..').locator('polygon')).toHaveCount(0);
		await field.getByRole('button', { name: 'Mark source unknown' }).click();
		await field.getByRole('button', { name: 'Show source', exact: true }).click();
		await expect(page.getByText('Source mapping unknown', { exact: true })).toBeVisible();
		await expect(page.locator('svg polygon')).toHaveCount(0);
		await page.route(
			(url) => url.href === workingUrl.href,
			(route) => route.fulfill({ status: 404 })
		);
		await page.reload();
		await page.getByRole('button', { name: 'Working image', exact: true }).click();
		await expect(page.getByText('Source image unavailable', { exact: true })).toBeVisible();
		await expect(page.getByText(/The original photo is missing/)).toHaveCount(0);
	});

	test('accepted boundary overlap projects final C row to its own exact block and polygon', async ({
		page,
		context
	}) => {
		test.skip(
			testEnvironment('VB_E2E_POLICY') === 'off',
			'Working OCR geometry requires the explicitly enabled stack'
		);
		await registerUser(context, 'ocr-boundary');
		const owner = await ownerId(context);
		await setStubScenario('overlap');
		const fixture = await loadFixture('a');
		const { body: job } = await uploadViaApi(context, 'a.png', fixture.bytes, fixture.mimeType, {
			photos: [0, 1].map((i) => ({
				name: `${i}.png`,
				bytes: fixture.bytes,
				mimeType: fixture.mimeType
			}))
		});
		expect(job.status).toBe('analyzed');
		const detail = await (await context.request.get(`/api/v1/receipts/${job.receipt_id}`)).json();
		expect(detail.line_items.map((line: { description: string }) => line.description)).toEqual([
			'A',
			'B',
			'C'
		]);
		const artifact = await (
			await context.request.get(`/api/v1/images/${job.id}/extractions/${job.latest_extraction_id}`)
		).json();
		expect(artifact.envelope.source_projection_version).toBe('2');
		expect(
			artifact.envelope.draft.line_items.map((line: { description: string }) => line.description)
		).toEqual(['A', 'B', 'B', 'C']);
		expect(artifact.envelope.overlap_decisions).toEqual([
			{ first_index: 1, second_index: 2, verified: true, removed_index: 2 }
		]);
		const source = detail.receipt.field_sources['line_items.2.description'][0];
		const removed = artifact.envelope.raw_field_sources['line_items.2.description'][0];
		const rawC = artifact.envelope.raw_field_sources['line_items.3.description'][0];
		expect(source).toEqual(rawC);
		expect(source.block_id).toBe('photo-1-2');
		expect(source.block_id).not.toBe(removed.block_id);
		expect(source.photo_index).toBe(1);
		await page.goto(`/receipts/${job.receipt_id}`);
		await page.reload();
		await page.getByText('Source decisions', { exact: true }).click();
		await page
			.locator('p')
			.filter({ hasText: /^line_items.2.description\s/ })
			.getByRole('button', { name: 'Show source', exact: true })
			.click();
		const views = job.source_views as {
			view_id: string;
			url: string;
			width: number;
			height: number;
			sha256: string;
		}[];
		const view = views.find((v) => v.view_id === source.view_id)!;
		await assertWorkingView(page, view, ownerMediaUrl(view.url, owner), source.bbox);
		await page.getByRole('button', { name: 'Original photos', exact: true }).click();
		await page.getByRole('button', { name: 'Photo 2', exact: true }).click();
		const originalUrl = ownerMediaUrl(`/api/v1/images/${job.id}/file?part=1`, owner);
		const original = page.locator(`img[src="${originalUrl.pathname}${originalUrl.search}"]`);
		await expect(original).toBeVisible();
		await expect(original.locator('..').locator('polygon')).toHaveCount(0);
	});

	test('server detects missing original despite retained working image and thumbnail', async ({
		page,
		context
	}) => {
		await registerUser(context, 'missing-original');
		const job = await create(context);
		expect(job.status).toBe('analyzed');
		const views = job.source_views as { url: string }[];
		await removeOwnedFile(job.id as string, 'original');
		expect((await context.request.get(views[0].url)).status()).toBe(200);
		expect((await context.request.get(`/api/v1/images/${job.id}/thumb`)).status()).toBe(200);
		const before = await (await context.request.get(`/api/v1/receipts/${job.receipt_id}`)).json();
		// Material review rechecks actual originals and persists the integrity finding.
		const reviewed = await context.request.put(`/api/v1/receipts/${job.receipt_id}`, {
			data: {
				...before.receipt,
				time: before.receipt.time?.slice(0, 5) ?? null,
				line_items: before.line_items,
				taxes: before.taxes,
				collection_ids: before.collection_ids,
				expected_operation_revision: before.receipt.operation_revision,
				merchant_name: 'Integrity recheck'
			}
		});
		expect(reviewed.status()).toBe(200);
		const detail = await (await context.request.get(`/api/v1/receipts/${job.receipt_id}`)).json();
		expect(
			detail.receipt.findings.some(
				(f: { code: string; severity: string }) =>
					f.code === 'original_integrity_failed' && f.severity === 'integrity'
			)
		).toBe(true);
		await page.goto(`/receipts/${job.receipt_id}`);
		await expect(
			page.getByText(
				'Verification blocked: The original is unavailable or its bytes have changed. Verification is blocked.',
				{ exact: true }
			)
		).toBeVisible();
		await expect(
			page.getByRole('textbox', { name: /Review comment: The original is unavailable/ })
		).toHaveCount(0);
		await page.getByRole('button', { name: 'Verify', exact: true }).click();
		const dialog = page.getByRole('dialog', { name: 'Verify receipt?' });
		await expect(dialog.getByRole('button', { name: 'Verify', exact: true })).toBeDisabled();
		await expect(dialog.getByRole('textbox')).toHaveCount(0);
		const denied = await context.request.post(`/api/v1/receipts/${job.receipt_id}/verify`, {
			data: {
				expected_operation_revision: detail.receipt.operation_revision,
				expected_review_revision: detail.receipt.review_revision,
				acknowledgements: []
			}
		});
		expect(denied.status()).toBe(422);
	});

	test('missing thumbnail on mobile loads full original and still allows verification', async ({
		page,
		context
	}) => {
		await registerUser(context, 'missing-thumbnail');
		const owner = await ownerId(context);
		const job = await create(context);
		expect(job.status).toBe('analyzed');
		await removeOwnedFile(job.id as string, 'thumbnail');
		await page.setViewportSize({ width: 390, height: 844 });
		const thumbnailUrl = ownerMediaUrl(`/api/v1/images/${job.id}/thumb`, owner);
		expect((await context.request.get(thumbnailUrl.href)).status()).toBe(404);
		await page.goto(`/receipts/${job.receipt_id}`);
		await page.getByRole('button', { name: 'Receipt', exact: true }).click();
		const originalUrl = ownerMediaUrl(`/api/v1/images/${job.id}/file`, owner);
		const original = page.locator(`img[src="${originalUrl.pathname}${originalUrl.search}"]`);
		await expect(original).toBeVisible();
		expect(
			await original.evaluate((img) => (img as HTMLImageElement).naturalWidth)
		).toBeGreaterThan(0);
		await verifyReceiptViaApi(context, job.receipt_id as string);
		await page.reload();
		await expect(page.getByText('Verified', { exact: true })).toBeVisible();
	});
});
