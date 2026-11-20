import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, expect, it, vi } from 'vitest';
import { goto } from '$app/navigation';
import { tick } from 'svelte';
import { session, signIn } from '$lib/auth';
import Page from './+page.svelte';
import { setLocale } from '$lib/i18n';
vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));
afterEach(() => {
	cleanup();
	setLocale('en');
	session.set('loading');
	vi.clearAllMocks();
	vi.unstubAllGlobals();
});

it('requires unchecked explicit consent and sends the preview revision/hash exactly', async () => {
	session.set({ id: 'owner', username: 'admin', is_admin: true, can_see_all: false });
	const calls: { url: string; init?: RequestInit }[] = [];
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string, init?: RequestInit) => {
			calls.push({ url, init });
			if (url === '/api/v1/llm/models') return Response.json([{ id: 'model' }]);
			if (url.endsWith('/capabilities'))
				return Response.json({ ocr_available: false, thinking_control: false });
			if (url.includes('/references/'))
				return Response.json({
					receipt_identity: 'receipt',
					review_revision: 7,
					document_sha256: 'a'.repeat(64),
					scored_fields: ['total'],
					expected: { total: '2.00' },
					sources: [],
					findings: []
				});
			return Response.json({});
		})
	);
	render(Page);
	await screen.findByText('model');
	expect(screen.queryByLabelText(/council/i)).toBeNull();
	expect(screen.getByLabelText('Request timeout (seconds)')).toBeTruthy();
	await fireEvent.input(screen.getByLabelText('Explicit receipt IDs'), {
		target: { value: 'receipt, receipt' }
	});
	await fireEvent.click(screen.getByRole('button', { name: 'Preview human references' }));
	const consent = await screen.findByLabelText(
		'I checked these current receipts and originals and explicitly qualify them as benchmark references.'
	);
	expect((consent as HTMLInputElement).checked).toBe(false);
	expect(calls.some((c) => c.init?.method === 'POST')).toBe(false);
	await fireEvent.click(consent);
	await fireEvent.click(screen.getByRole('button', { name: 'Create benchmark' }));
	expect(screen.getByRole('dialog', { name: 'Create benchmark?' })).toBeTruthy();
	expect(screen.getByText(/Selected models: model/)).toBeTruthy();
	await fireEvent.click(screen.getByRole('button', { name: 'Confirm and start' }));
	await waitFor(() => expect(calls.filter((c) => c.init?.method === 'POST')).toHaveLength(2));
	const writes = calls.filter((c) => c.init?.method === 'POST');
	expect(
		calls.filter((call) => call.url.endsWith('/references/receipt') && call.init?.method !== 'POST')
	).toHaveLength(1);
	expect(JSON.parse(writes[1].init!.body as string).receipt_ids).toEqual(['receipt']);
	expect(writes[0].url).toBe('/api/v1/benchmarks/references/receipt');
	expect(JSON.parse(writes[0].init!.body as string)).toEqual({
		expected_review_revision: 7,
		expected_document_sha256: 'a'.repeat(64)
	});
});

it('loads text formatter models for forced OCR and offers no misleading fallback override', async () => {
	session.set({ id: 'owner', username: 'admin', is_admin: true, can_see_all: false });
	const calls: string[] = [];
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string) => {
			calls.push(url);
			if (url.endsWith('/capabilities'))
				return Response.json({ ocr_available: true, ocr_enabled: false, thinking_control: false });
			return Response.json([
				{ id: url.includes('capability=text') ? 'text-only' : 'vision-model' }
			]);
		})
	);
	render(Page);
	await screen.findByText('vision-model');
	expect(screen.getByRole('group', { name: 'Vision models' })).not.toBeNull();
	await fireEvent.change(screen.getByLabelText('Pipeline'), { target: { value: 'ocr' } });
	await screen.findByText('text-only');
	expect(screen.getByRole('group', { name: 'OCR text formatter models' })).not.toBeNull();
	expect(calls).toContain('/api/v1/llm/models?capability=text');
	expect(screen.queryByLabelText(/Allow direct vision fallback/)).toBeNull();
	await fireEvent.change(screen.getByLabelText('Pipeline'), { target: { value: 'direct_vision' } });
	await screen.findByText('vision-model');
	expect(screen.getByRole('group', { name: 'Vision models' })).not.toBeNull();
});

it('labels the default upload policy as a text formatter when OCR is enabled', async () => {
	session.set({ id: 'owner', username: 'admin', is_admin: true, can_see_all: false });
	const calls: string[] = [];
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string) => {
			calls.push(url);
			if (url.endsWith('/capabilities'))
				return Response.json({ ocr_available: true, ocr_enabled: true, thinking_control: false });
			return Response.json([
				{ id: url.includes('capability=text') ? 'upload-formatter' : 'vision-model' }
			]);
		})
	);
	render(Page);
	await screen.findByText('upload-formatter');
	expect(screen.getByRole('group', { name: 'OCR text formatter models' })).not.toBeNull();
	expect(calls).toContain('/api/v1/llm/models?capability=text');
});

it('shows the localized GPU memory error when benchmark OCR qualification fails', async () => {
	setLocale('de');
	session.set({ id: 'owner', username: 'admin', is_admin: true, can_see_all: false });
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string) => {
			if (url.endsWith('/capabilities'))
				return Response.json({ ocr_available: true, ocr_enabled: true, thinking_control: false });
			if (url === '/api/v1/benchmarks')
				return Response.json({ detail: 'ocr_gpu_out_of_memory' }, { status: 503 });
			return Response.json([{ id: 'ocr-formatter' }]);
		})
	);
	render(Page);
	await screen.findByText('ocr-formatter');
	await fireEvent.click(screen.getByRole('button', { name: 'Benchmark anlegen' }));
	await fireEvent.click(screen.getByRole('button', { name: 'Bestätigen und starten' }));
	expect((await screen.findByRole('alert')).textContent).toBe(
		'Die lokale OCR-GPU hat nicht genug Grafikspeicher.'
	);
	expect(goto).not.toHaveBeenCalled();
});

it.each(['account change', 'view destruction'])(
	'stops after the first outstanding qualification on %s',
	async (boundary) => {
		const alice = { id: 'a', username: 'alice', is_admin: true, can_see_all: false };
		const bob = { ...alice, id: 'b', username: 'bob' };
		session.set(alice);
		let finishQualification!: (response: Response) => void;
		const qualification = new Promise<Response>((resolve) => {
			finishQualification = resolve;
		});
		const protectedWrites: string[] = [];
		vi.stubGlobal(
			'fetch',
			vi.fn(async (url: string, init?: RequestInit) => {
				if (url.endsWith('/auth/login')) return Response.json(bob);
				if (url.endsWith('/capabilities'))
					return Response.json({
						ocr_available: false,
						ocr_enabled: false,
						thinking_control: false
					});
				if (url.endsWith('/llm/models')) return Response.json([{ id: 'model' }]);
				if (init?.method === 'POST') {
					protectedWrites.push(url);
					return protectedWrites.length === 1 ? qualification : Response.json({});
				}
				return Response.json({
					receipt_identity: url.endsWith('/first') ? 'first' : 'second',
					review_revision: 7,
					document_sha256: 'a'.repeat(64),
					scored_fields: ['total'],
					expected: { total: '2.00' },
					sources: [],
					findings: []
				});
			})
		);
		const view = render(Page);
		await screen.findByText('model');
		await fireEvent.input(screen.getByLabelText('Explicit receipt IDs'), {
			target: { value: 'first,second' }
		});
		await fireEvent.click(screen.getByRole('button', { name: 'Preview human references' }));
		await fireEvent.click(
			await screen.findByLabelText(
				'I checked these current receipts and originals and explicitly qualify them as benchmark references.'
			)
		);
		await fireEvent.click(screen.getByRole('button', { name: 'Create benchmark' }));
		await fireEvent.click(screen.getByRole('button', { name: 'Confirm and start' }));
		await waitFor(() => expect(protectedWrites).toEqual(['/api/v1/benchmarks/references/first']));
		expect(screen.queryByRole('dialog')).toBeNull();
		expect(
			(screen.getByRole('button', { name: 'Create benchmark' }) as HTMLButtonElement).disabled
		).toBe(true);
		expect(screen.getByRole('status').textContent).toBe('Starting benchmark…');
		await fireEvent.submit(
			screen.getByRole('button', { name: 'Create benchmark' }).closest('form')!
		);
		expect(screen.queryByRole('dialog')).toBeNull();
		if (boundary === 'account change') await signIn('bob', 'x');
		else view.unmount();
		finishQualification(Response.json({}));
		await qualification;
		await tick();
		await new Promise((resolve) => setTimeout(resolve, 0));
		expect(protectedWrites).toEqual(['/api/v1/benchmarks/references/first']);
		expect(goto).not.toHaveBeenCalled();
	}
);

it('localizes creation and the selected references summary in German', async () => {
	setLocale('de');
	session.set({ id: 'owner', username: 'admin', is_admin: true, can_see_all: false });
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string) =>
			url.endsWith('/capabilities')
				? Response.json({ ocr_available: false, ocr_enabled: false, thinking_control: false })
				: Response.json([{ id: 'model' }])
		)
	);
	render(Page);
	await screen.findByText('model');
	expect(screen.getByLabelText('Verarbeitungskette')).toBeTruthy();
	await fireEvent.input(screen.getByLabelText('Explizite Beleg-IDs'), {
		target: { value: 'first,second,first' }
	});
	await fireEvent.click(screen.getByRole('button', { name: 'Benchmark anlegen' }));
	expect(screen.getByRole('dialog', { name: 'Benchmark anlegen?' })).toBeTruthy();
	expect(screen.getByText('Gewählte Modelle: model')).toBeTruthy();
	expect(screen.getByText('Explizite Referenzen: 2')).toBeTruthy();
	await fireEvent.click(screen.getByRole('button', { name: 'Abbrechen' }));
	expect(screen.queryByRole('dialog')).toBeNull();
});
