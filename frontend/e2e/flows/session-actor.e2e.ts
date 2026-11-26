import { expect, test, type BrowserContext, type Page } from '@playwright/test';
import { ADMIN, loginAdmin, loginAs, registerUser, type Creds } from '../helpers/auth';
import { loadFixture } from '../helpers/fixtures';
import { stubGuard } from '../helpers/stub';
import { uploadFixture } from '../helpers/ui';

async function accountChange(page: Page, successor: Creds): Promise<void> {
	await page.getByRole('button', { name: 'Sign out' }).click();
	await expect(page).toHaveURL(/\/login$/);
	await page.getByLabel('Username').fill(successor.username);
	await page.getByLabel('Password').fill(successor.password);
	await page.getByRole('button', { name: 'Sign in' }).click();
	await expect(page.getByText(`Signed in as ${successor.username}`, { exact: true })).toBeVisible();
}

async function imageIds(context: BrowserContext): Promise<string[]> {
	const response = await context.request.get('/api/v1/images');
	expect(response.status()).toBe(200);
	return ((await response.json()) as { id: string }[]).map((image) => image.id);
}

test.describe('browser session actor', () => {
	stubGuard();

	test('rejects stale tab media, a held upload, reference qualification and run creation', async ({
		browser,
		context,
		page,
		baseURL
	}) => {
		const admin = await browser.newContext({ baseURL });
		try {
			await loginAdmin(admin);
			const adminPage = await admin.newPage();
			const goldId = await uploadFixture(adminPage, admin, await loadFixture('c'));
			await adminPage.close();
			const preview = await admin.request.get(`/api/v1/benchmarks/references/${goldId}`);
			expect(preview.status()).toBe(200);
			const { review_revision, document_sha256 } = await preview.json();
			const alice = await registerUser(context, 'actor-alice');
			const fixture = await loadFixture('a');
			const receiptId = await uploadFixture(page, context, fixture, { verify: false });
			const { id: aliceId } = await (await context.request.get('/api/v1/auth/me')).json();
			const stale = await context.newPage();
			await stale.addInitScript(() => {
				// Simulate a tab that missed the cookie notification and has not resumed.
				Object.defineProperty(window, 'BroadcastChannel', { value: undefined });
				for (const type of ['focus', 'pageshow'])
					window.addEventListener(type, (event) => event.stopImmediatePropagation());
				document.addEventListener('visibilitychange', (event) => event.stopImmediatePropagation());
				const actualFetch = window.fetch.bind(window);
				window.fetch = async (input, init) => {
					const url = new URL(String(input), location.href);
					if (url.pathname === '/api/v1/images' && init?.method === 'POST') {
						const release = new Promise<void>((resolve) =>
							window.addEventListener('resume-actor-upload', () => resolve(), { once: true })
						);
						document.documentElement.dataset.actorUploadHeld = 'yes';
						await release;
					}
					return actualFetch(input, init);
				};
			});
			const warmMedia = stale.waitForResponse((response) =>
				new URL(response.url()).pathname.endsWith('/file')
			);
			await stale.goto(`/receipts/${receiptId}`);
			const photo = stale.locator('img[src*="/file"]').first();
			await expect(photo).toBeVisible();
			const oldUrl = (await photo.getAttribute('src'))!;
			const warmResponse = await warmMedia;
			expect(warmResponse.status()).toBe(200);
			await stale.goto('/upload');
			await expect(
				stale.getByText(`Signed in as ${alice.username}`, { exact: true })
			).toBeVisible();
			await stale.locator('input[type=file]').first().setInputFiles({
				name: 'e2e-stale.png',
				mimeType: fixture.mimeType,
				buffer: fixture.bytes
			});
			await stale.getByRole('button', { name: 'Upload' }).click();
			await expect(stale.locator('html')).toHaveAttribute('data-actor-upload-held', 'yes');
			const uploadIdsBefore = await imageIds(admin);
			const runsBefore = await (await admin.request.get('/api/v1/benchmarks')).json();

			// Both pages share the HttpOnly cookie. Change it after multipart preflight
			// and before the stale tab calls the actual browser fetch.
			await accountChange(page, ADMIN);
			await expect(
				stale.getByText(`Signed in as ${alice.username}`, { exact: true })
			).toBeVisible();
			const replay = new URL(oldUrl, stale.url());
			replay.searchParams.set('actor_test_replay', '1');
			const media = stale.waitForResponse((response) => response.url() === replay.href);
			await stale.evaluate((src) => {
				const image = new Image();
				image.src = src;
				document.body.appendChild(image);
			}, replay.href);
			// Use the actor captured before the cookie changed to check both admin
			// mutation entry points under the successor's real shared browser cookie.
			const benchmark = await stale.evaluate(
				async ({ actor, receipt, revision, digest }) => {
					const post = async (path: string, data: unknown) => {
						const response = await fetch(path, {
							method: 'POST',
							headers: {
								'Content-Type': 'application/json',
								'X-Vision-Bill-Expected-User': actor
							},
							body: JSON.stringify(data)
						});
						return { status: response.status, body: await response.json() };
					};
					const qualification = await post(`/api/v1/benchmarks/references/${receipt}`, {
						expected_review_revision: revision,
						expected_document_sha256: digest
					});
					const run = await post('/api/v1/benchmarks', {
						model_ids: ['e2e-vision'],
						receipt_ids: [receipt]
					});
					return { qualification, run };
				},
				{ actor: aliceId, receipt: goldId, revision: review_revision, digest: document_sha256 }
			);
			const upload = stale.waitForResponse(
				(response) =>
					new URL(response.url()).pathname === '/api/v1/images' &&
					response.request().method() === 'POST'
			);
			await stale.evaluate(() => window.dispatchEvent(new Event('resume-actor-upload')));
			const [staleMedia, staleUpload] = await Promise.all([media, upload]);
			expect([
				staleUpload.status(),
				staleMedia.status(),
				benchmark.qualification.status,
				benchmark.run.status
			]).toEqual([409, 409, 409, 409]);
			expect(benchmark.qualification.body).toEqual({ detail: 'session_actor_changed' });
			expect(benchmark.run.body).toEqual({ detail: 'session_actor_changed' });
			expect(await staleUpload.json()).toEqual({ detail: 'session_actor_changed' });
			expect(await staleUpload.request().headerValue('X-Vision-Bill-Expected-User')).toBe(aliceId);
			expect(await warmResponse.headerValue('Cache-Control')).toBe('private, no-store');
			expect(await staleMedia.json()).toEqual({ detail: 'session_actor_changed' });
			expect(new URL(staleMedia.url()).searchParams.get('vb_expected_user')).toBe(aliceId);
			expect(await imageIds(admin)).toEqual(uploadIdsBefore);
			expect(await (await admin.request.get('/api/v1/benchmarks')).json()).toEqual(runsBefore);
			const stillUnqualified = await admin.request.post('/api/v1/benchmarks', {
				data: { model_ids: ['e2e-vision'], receipt_ids: [goldId] }
			});
			expect(stillUnqualified.status()).toBe(422);
			expect((await stillUnqualified.json()).detail).toContain('qualified');
			await expect(stale.getByText('Signed in as admin', { exact: true })).toBeVisible();
			await expect(stale).toHaveURL(/\/upload$/);
		} finally {
			await admin.close();
		}
	});

	test('applies a cross-tab account notification and rechecks a tab that missed it', async ({
		browser,
		context,
		page,
		baseURL
	}) => {
		const independent = await browser.newContext({ baseURL });
		try {
			const bob = await registerUser(independent, 'actor-bob');
			const alice = await registerUser(context, 'actor-alice');
			const listening = await context.newPage();
			const resumed = await context.newPage();
			await resumed.addInitScript(() => {
				Object.defineProperty(window, 'BroadcastChannel', { value: undefined });
			});
			for (const tab of [page, listening, resumed]) {
				await tab.goto('/receipts');
				await expect(
					tab.getByText(`Signed in as ${alice.username}`, { exact: true })
				).toBeVisible();
			}
			await accountChange(page, bob);
			await expect(
				listening.getByText(`Signed in as ${bob.username}`, { exact: true })
			).toBeVisible();

			// Recreate a missed event deterministically without relying on headless focus.
			await loginAs(context, alice);
			await resumed.reload();
			await expect(
				resumed.getByText(`Signed in as ${alice.username}`, { exact: true })
			).toBeVisible();
			await loginAs(context, bob);
			await resumed.evaluate(() => window.dispatchEvent(new Event('focus')));
			await expect(
				resumed.getByText(`Signed in as ${bob.username}`, { exact: true })
			).toBeVisible();
		} finally {
			await independent.close();
		}
	});

	test('keeps a committed deletion from notifying the successor tab', async ({
		browser,
		context,
		page,
		baseURL
	}) => {
		const independent = await browser.newContext({ baseURL });
		try {
			const bob = await registerUser(independent, 'actor-delete-bob');
			await registerUser(context, 'actor-delete-alice');
			const receipt = await uploadFixture(page, context, await loadFixture('a'), {
				verify: false
			});
			const endpoint = `/api/v1/receipts/${receipt}`;
			let markCommitted!: (status: number) => void;
			const committed = new Promise<number>((resolve) => (markCommitted = resolve));
			let release!: () => void;
			const held = new Promise<void>((resolve) => (release = resolve));
			await page.route(`**${endpoint}`, async (route) => {
				if (route.request().method() !== 'DELETE') {
					await route.continue();
					return;
				}
				const response = await route.fetch();
				markCommitted(response.status());
				await held;
				await route.fulfill({ response });
			});
			const switcher = await context.newPage();
			await switcher.goto('/');
			await page.goto('/receipts');
			await page.getByRole('button', { name: 'Remove' }).click();
			await page
				.getByRole('dialog', { name: 'Delete receipt?' })
				.getByRole('button', { name: 'Delete' })
				.click();
			expect(await committed).toBe(200);
			expect((await context.request.get(endpoint)).status()).toBe(404);

			await accountChange(switcher, bob);
			await expect(page.getByText(`Signed in as ${bob.username}`, { exact: true })).toBeVisible();
			const delivery = page.waitForResponse(
				(response) => response.url().endsWith(endpoint) && response.request().method() === 'DELETE'
			);
			release();
			expect((await delivery).status()).toBe(200);
			await page.bringToFront();
			await page.evaluate(
				() =>
					new Promise<void>((resolve) =>
						requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
					)
			);
			await expect(page.getByText('Deleting failed', { exact: true })).toHaveCount(0);
			await expect(page.getByText('Receipt deleted', { exact: true })).toHaveCount(0);
		} finally {
			await independent.close();
		}
	});

	test('stops the old benchmark after a real qualification finishes across an account change', async ({
		browser,
		context,
		page,
		baseURL
	}) => {
		const independent = await browser.newContext({ baseURL });
		try {
			const successor = await registerUser(independent, 'actor-benchmark');
			await loginAdmin(context);
			const receipt = await uploadFixture(page, context, await loadFixture('b'));
			const reference = `/api/v1/benchmarks/references/${receipt}`;
			let resume!: () => void;
			const held = new Promise<void>((resolve) => (resume = resolve));
			let markQualificationComplete!: () => void;
			const qualified = new Promise<void>((resolve) => (markQualificationComplete = resolve));
			await page.route(`**${reference}`, async (route) => {
				if (route.request().method() !== 'POST') {
					await route.continue();
					return;
				}
				const response = await route.fetch();
				expect(response.status()).toBe(200);
				markQualificationComplete();
				await held;
				await route.fulfill({ response });
			});
			const subsequentPosts: string[] = [];
			page.on('request', (request) => {
				const path = new URL(request.url()).pathname;
				if (request.method() === 'POST' && path === '/api/v1/benchmarks')
					subsequentPosts.push(path);
			});
			await page.getByRole('link', { name: 'Use this receipt in a benchmark' }).click();
			await expect(page).toHaveURL(new RegExp('\\?receipt=' + receipt + '$'));
			await page.getByRole('checkbox', { name: 'e2e-vision', exact: true }).check();
			await page.getByRole('button', { name: 'Preview human references' }).click();
			await page
				.getByRole('checkbox', {
					name: 'I checked these current receipts and originals and explicitly qualify them as benchmark references.'
				})
				.check();
			await page.getByRole('button', { name: 'Create benchmark' }).click();
			const first = page.waitForRequest(
				(request) => request.url().endsWith(reference) && request.method() === 'POST'
			);
			await page.getByRole('button', { name: 'Confirm and start' }).click();
			await first;
			await qualified;
			const persisted = await context.request.get(reference);
			expect(persisted.status()).toBe(200);
			expect((await persisted.json()).currently_qualified).toBe(true);

			// Starting closes confirmation so account controls remain available while
			// the committed qualification response is still pending.
			await expect(page.getByRole('dialog', { name: 'Create benchmark?' })).toHaveCount(0);
			await expect(page.getByRole('status')).toHaveText('Starting benchmark…');
			await expect(page.getByRole('button', { name: 'Create benchmark' })).toBeDisabled();
			await accountChange(page, successor);
			const completed = page.waitForResponse(
				(response) => response.url().endsWith(reference) && response.request().method() === 'POST'
			);
			resume();
			expect((await completed).status()).toBe(200);
			await page.evaluate(
				() =>
					new Promise<void>((resolve) =>
						requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
					)
			);
			expect(subsequentPosts).toEqual([]);
			await expect(page).toHaveURL(/\/$/);
		} finally {
			await independent.close();
		}
	});
});
