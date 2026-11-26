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
			if (url.endsWith('/benchmarks/cases') || url.includes('/receipts?')) return Response.json([]);
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
	await fireEvent.click(await screen.findByLabelText('model'));
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
			if (url.endsWith('/benchmarks/cases') || url.includes('/receipts?')) return Response.json([]);
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
	await fireEvent.click(screen.getByRole('radio', { name: 'OCR + LLM' }));
	await screen.findByText('text-only');
	expect(screen.getByRole('group', { name: 'OCR text formatter models' })).not.toBeNull();
	expect(calls).toContain('/api/v1/llm/models?capability=text');
	expect(screen.queryByLabelText(/Allow direct vision fallback/)).toBeNull();
	await fireEvent.click(screen.getByRole('radio', { name: 'Image directly to LLM' }));
	await screen.findByText('vision-model');
	expect(screen.getByRole('group', { name: 'Vision models' })).not.toBeNull();
});

it('defaults to direct images even when production OCR is enabled', async () => {
	session.set({ id: 'owner', username: 'admin', is_admin: true, can_see_all: false });
	const calls: string[] = [];
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string) => {
			if (url.endsWith('/benchmarks/cases') || url.includes('/receipts?')) return Response.json([]);
			calls.push(url);
			if (url.endsWith('/capabilities'))
				return Response.json({ ocr_available: true, ocr_enabled: true, thinking_control: false });
			return Response.json([
				{ id: url.includes('capability=text') ? 'upload-formatter' : 'vision-model' }
			]);
		})
	);
	render(Page);
	await screen.findByText('vision-model');
	expect(screen.getByRole('group', { name: 'Vision models' })).not.toBeNull();
	expect(
		(screen.getByRole('radio', { name: 'Image directly to LLM' }) as HTMLInputElement).checked
	).toBe(true);
	expect(calls).toContain('/api/v1/llm/models?capability=text');
});

it('shows the localized GPU memory error when benchmark OCR qualification fails', async () => {
	setLocale('de');
	session.set({ id: 'owner', username: 'admin', is_admin: true, can_see_all: false });
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string) => {
			if (url.endsWith('/benchmarks/cases') || url.includes('/receipts?')) return Response.json([]);
			if (url.endsWith('/capabilities'))
				return Response.json({ ocr_available: true, ocr_enabled: true, thinking_control: false });
			if (url === '/api/v1/benchmarks')
				return Response.json({ detail: 'ocr_gpu_out_of_memory' }, { status: 503 });
			return Response.json([{ id: 'ocr-formatter' }]);
		})
	);
	render(Page);
	await screen.findByLabelText('ocr-formatter');
	const mode = screen.getByRole('radio', { name: 'OCR + LLM' }) as HTMLInputElement;
	await fireEvent.click(mode);
	expect(mode.checked).toBe(true);
	await fireEvent.click(screen.getByLabelText('ocr-formatter'));
	await fireEvent.input(screen.getByLabelText('Explizite Beleg-IDs'), {
		target: { value: 'receipt' }
	});
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
				if (url.endsWith('/benchmarks/cases') || url.includes('/receipts?'))
					return Response.json([]);
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
		await fireEvent.click(await screen.findByLabelText('model'));
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
			url.endsWith('/benchmarks/cases') || url.includes('/receipts?')
				? Response.json([])
				: url.endsWith('/capabilities')
					? Response.json({ ocr_available: false, ocr_enabled: false, thinking_control: false })
					: Response.json([{ id: 'model' }])
		)
	);
	render(Page);
	await fireEvent.click(await screen.findByLabelText('model'));
	expect(screen.getByRole('group', { name: 'Verarbeitungskette' })).toBeTruthy();
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

it('submits one explicit paired request for the selected receipt and model after confirmation', async () => {
	session.set({ id: 'owner', username: 'admin', is_admin: true, can_see_all: false });
	const writes: RequestInit[] = [];
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string, init?: RequestInit) => {
			if (url.endsWith('/capabilities'))
				return Response.json({ ocr_available: true, ocr_enabled: true, thinking_control: false });
			if (url.includes('/llm/models'))
				return Response.json(
					url.includes('capability=text')
						? [{ id: 'vision-model' }, { id: 'text-only' }]
						: [{ id: 'vision-model' }, { id: 'other-model' }]
				);
			if (url.includes('/receipts?'))
				return Response.json([
					{
						id: 'receipt',
						merchant_name: 'Corner shop',
						date: '2026-01-12',
						total: '2.00',
						currency: 'EUR',
						image_id: 'photo',
						verified: true,
						status: 'verified',
						verification_source: 'human'
					}
				]);
			if (url.endsWith('/benchmarks/cases'))
				return Response.json([
					{ id: 'case', name: 'Private shop', confirmed: true, photo_count: 2 }
				]);
			if (init?.method === 'POST') writes.push(init);
			return Response.json({ id: 'new-run' });
		})
	);
	render(Page);
	const initial = await screen.findByRole('radio', { name: 'Image directly to LLM' });
	expect((initial as HTMLInputElement).checked).toBe(true);
	const model = await screen.findByLabelText('vision-model');
	expect((model as HTMLInputElement).checked).toBe(false);
	await fireEvent.click(model);
	await fireEvent.click(screen.getByRole('radio', { name: 'OCR + LLM' }));
	await waitFor(() =>
		expect((screen.getByLabelText('vision-model') as HTMLInputElement).checked).toBe(true)
	);
	await fireEvent.click(screen.getByRole('radio', { name: 'Compare both' }));
	await fireEvent.click(screen.getByLabelText('text-only'));
	await fireEvent.input(screen.getByLabelText('Temperature'), { target: { value: '0.35' } });
	await fireEvent.click(await screen.findByRole('checkbox', { name: /Corner shop/ }));
	await fireEvent.click(await screen.findByRole('checkbox', { name: /Private shop/ }));
	await fireEvent.click(screen.getByRole('button', { name: 'Create benchmark' }));
	expect(screen.getByText('Planned extractions: 6')).toBeTruthy();
	expect(writes).toEqual([]);
	await fireEvent.click(screen.getByRole('button', { name: 'Confirm and start' }));
	await waitFor(() => expect(goto).toHaveBeenCalledWith('/benchmarks/results?run=new-run'));
	expect(writes).toHaveLength(1);
	expect(JSON.parse(writes[0].body as string)).toMatchObject({
		comparison: true,
		model_ids: ['vision-model', 'text-only'],
		receipt_ids: ['receipt'],
		case_ids: ['case'],
		experiment: { pipeline: 'direct_vision', temperature: 0.35 }
	});
});

it('keeps selected receipts across searches, disables ineligible rows and reuses qualified Gold without a write', async () => {
	session.set({ id: 'owner', username: 'admin', is_admin: true, can_see_all: false });
	const selectedRow = {
		id: 'receipt',
		merchant_name: 'Corner shop',
		date: '2026-01-12',
		total: '2.00',
		currency: 'EUR',
		image_id: 'photo',
		thumbnail_path: 'private.jpg',
		verified: true,
		status: 'verified',
		verification_source: 'human'
	};
	const writes: string[] = [];
	const queries: URLSearchParams[] = [];
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string, init?: RequestInit) => {
			if (url.includes('/receipts?')) {
				const params = new URL(url, 'http://localhost').searchParams;
				queries.push(params);
				if (params.get('search')) return Response.json([]);
				if (params.get('offset') === '10')
					return Response.json([
						{ ...selectedRow, id: 'other', merchant_name: 'Online store', image_id: null }
					]);
				return Response.json([
					selectedRow,
					...Array.from({ length: 10 }, (_, index) => ({
						...selectedRow,
						id: 'auto-' + index,
						merchant_name: 'Unreviewed ' + index,
						verification_source: 'automatic_bypass'
					}))
				]);
			}
			if (url.endsWith('/capabilities'))
				return Response.json({ ocr_available: false, ocr_enabled: false, thinking_control: false });
			if (url.includes('/llm/models')) return Response.json([{ id: 'model' }]);
			if (url.endsWith('/benchmarks/cases')) return Response.json([]);
			if (url.endsWith('/references/stale'))
				return Response.json({ detail: 'original_integrity_failed' }, { status: 409 });
			if (url.endsWith('/references/receipt'))
				return Response.json({
					receipt_identity: 'receipt',
					review_revision: 7,
					document_sha256: 'a'.repeat(64),
					scored_fields: ['total'],
					expected: { total: '2.00' },
					sources: [],
					findings: [],
					currently_qualified: true
				});
			if (init?.method === 'POST') writes.push(url);
			return Response.json({ id: 'one' });
		})
	);
	render(Page);
	const filterOptIn = screen.getByLabelText(
		'Use existing qualified references matching the filters'
	) as HTMLInputElement;
	await fireEvent.click(filterOptIn);
	expect(filterOptIn.checked).toBe(true);
	await fireEvent.click(await screen.findByRole('checkbox', { name: /Corner shop/ }));
	expect(filterOptIn.checked).toBe(false);
	expect(document.querySelector('img')?.getAttribute('src')).toContain(
		'/images/photo/thumb?vb_expected_user=owner'
	);
	await fireEvent.click(screen.getByRole('button', { name: 'Preview reference for Corner shop' }));
	await screen.findByText(/Current reference qualified/);
	expect(
		(screen.getByRole('checkbox', { name: /Unreviewed 0/ }) as HTMLInputElement).disabled
	).toBe(true);
	await fireEvent.click(screen.getByRole('button', { name: 'Next receipts' }));
	expect(
		((await screen.findByRole('checkbox', { name: /Online store/ })) as HTMLInputElement).disabled
	).toBe(true);
	await fireEvent.input(screen.getByRole('searchbox'), { target: { value: 'Absent shop' } });
	await screen.findByText('No matching receipts.');
	expect(queries.at(-1)?.get('offset')).toBe('0');
	expect(queries.at(-1)?.get('search')).toBe('Absent shop');
	expect(screen.getByRole('button', { name: 'Remove Corner shop' })).toBeTruthy();
	await fireEvent.click(screen.getByLabelText('model'));
	await fireEvent.input(screen.getByLabelText('Explicit receipt IDs'), {
		target: { value: 'receipt,stale' }
	});
	await fireEvent.click(screen.getByRole('button', { name: 'Preview human references' }));
	await screen.findByText(/original_integrity_failed/);
	expect(screen.getByText(/Current reference qualified/)).toBeTruthy();
	expect(
		(screen.getByRole('button', { name: 'Create benchmark' }) as HTMLButtonElement).disabled
	).toBe(true);
	await fireEvent.click(screen.getByRole('button', { name: 'Remove stale' }));
	await fireEvent.click(screen.getByRole('button', { name: 'Preview human references' }));
	await screen.findByText(/Current reference qualified/);
	expect(screen.queryByLabelText(/I checked these current receipts/)).toBeNull();
	await fireEvent.click(screen.getByRole('button', { name: 'Create benchmark' }));
	expect(writes).toEqual([]);
	await fireEvent.click(screen.getByRole('button', { name: 'Confirm and start' }));
	await waitFor(() => expect(goto).toHaveBeenCalledWith('/benchmarks/results?run=one'));
	expect(writes).toEqual(['/api/v1/benchmarks']);
});

it('keeps the created run reachable when results navigation fails', async () => {
	session.set({ id: 'owner', username: 'admin', is_admin: true, can_see_all: false });
	const writes: string[] = [];
	vi.mocked(goto).mockRejectedValueOnce(new Error('navigation failed'));
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string, init?: RequestInit) => {
			if (url.endsWith('/capabilities'))
				return Response.json({ ocr_available: false, ocr_enabled: false, thinking_control: false });
			if (url.includes('/llm/models')) return Response.json([{ id: 'vision-model' }]);
			if (init?.method === 'POST') {
				writes.push(url);
				return Response.json({ id: 'created' });
			}
			return Response.json([]);
		})
	);
	render(Page);
	await fireEvent.click(await screen.findByLabelText('vision-model'));
	await fireEvent.input(screen.getByLabelText('Explicit receipt IDs'), {
		target: { value: 'receipt' }
	});
	await fireEvent.click(screen.getByRole('button', { name: 'Create benchmark' }));
	await fireEvent.click(screen.getByRole('button', { name: 'Confirm and start' }));
	const result = await screen.findByRole('link', { name: 'Open the created run' });
	expect(result.getAttribute('href')).toBe('/benchmarks/results?run=created');
	expect(
		(screen.getByRole('button', { name: 'Create benchmark' }) as HTMLButtonElement).disabled
	).toBe(true);
	expect(writes).toEqual(['/api/v1/benchmarks']);
});

it('preserves selections across rediscovery and requires explicit removal of a missing model', async () => {
	session.set({ id: 'owner', username: 'admin', is_admin: true, can_see_all: false });
	let ids = ['alpha', 'beta'];
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string) => {
			if (url.endsWith('/capabilities')) return Response.json({ ocr_available: false });
			if (url.endsWith('/llm/models')) return Response.json(ids.map((id) => ({ id })));
			return Response.json([]);
		})
	);
	render(Page);
	await fireEvent.click(await screen.findByLabelText('alpha'));
	await fireEvent.click(screen.getByLabelText('beta'));
	await fireEvent.input(screen.getByLabelText('Explicit receipt IDs'), {
		target: { value: 'receipt' }
	});
	const create = screen.getByRole('button', { name: 'Create benchmark' }) as HTMLButtonElement;
	expect(create.disabled).toBe(false);
	ids = ['alpha', 'gamma'];
	await fireEvent.click(screen.getByRole('button', { name: 'Refresh models' }));
	await screen.findByLabelText('gamma');
	expect((screen.getByLabelText('alpha') as HTMLInputElement).checked).toBe(true);
	expect(create.disabled).toBe(true);
	await fireEvent.click(screen.getByRole('button', { name: 'Remove beta' }));
	expect(create.disabled).toBe(false);
	ids = ['delta'];
	window.dispatchEvent(new Event('focus'));
	await screen.findByLabelText('delta');
	expect(create.disabled).toBe(true);
});
