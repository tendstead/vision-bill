import { cleanup, fireEvent, render, screen, within } from '@testing-library/svelte';
import { afterEach, expect, it, vi } from 'vitest';
import { session } from '$lib/auth';
import { setLocale } from '$lib/i18n';
import Page from './+page.svelte';

vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));
const admin = { id: 'alice', username: 'alice', is_admin: true, can_see_all: false };
const base = '/api/v1/benchmarks/cases';
const specimen = {
	id: 'sample',
	name: 'Corner shop',
	revision: 1,
	expected: {},
	confirmed: false,
	photo_count: 2,
	owner_id: 'alice',
	created_at: '2026-01-01',
	updated_at: '2026-01-01'
};
afterEach(() => {
	cleanup();
	setLocale('en');
	session.set('loading');
	vi.unstubAllGlobals();
});

it('uploads photos in displayed order, edits nullable values and explicitly confirms a revision', async () => {
	session.set(admin);
	const writes: RequestInit[] = [];
	const requests: string[] = [];
	let detail = { ...specimen };
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string, init: RequestInit) => {
			requests.push(url);
			expect(init.credentials).toBe('same-origin');
			expect(new Headers(init.headers).get('X-Vision-Bill-Expected-User')).toBe('alice');
			if (init.method === 'POST') {
				writes.push(init);
				return Response.json(detail, { status: 201 });
			}
			if (init.method === 'PATCH') {
				writes.push(init);
				const update = JSON.parse(init.body as string);
				detail = {
					...detail,
					revision: detail.revision + 1,
					expected: update.expected,
					confirmed: update.confirm
				};
				return Response.json(detail);
			}
			return Response.json([]);
		})
	);
	render(Page);
	await fireEvent.click(await screen.findByRole('button', { name: 'New case' }));
	await fireEvent.input(screen.getByLabelText('Case name'), { target: { value: 'Corner shop' } });
	await fireEvent.change(screen.getByLabelText('Receipt photos'), {
		target: {
			files: [
				new File(['bottom'], 'bottom.png', { type: 'image/png' }),
				new File(['top'], 'top.png', { type: 'image/png' })
			]
		}
	});
	await fireEvent.click(screen.getAllByRole('button', { name: 'Move photo up' })[1]);
	await fireEvent.click(screen.getByRole('button', { name: 'Create case' }));
	const total = within(await screen.findByRole('group', { name: 'Amounts' })).getByLabelText(
		'Total'
	);
	const form = writes[0].body as FormData;
	expect(form.get('name')).toBe('Corner shop');
	expect(form.getAll('photos').map((f) => (f as File).name)).toEqual(['top.png', 'bottom.png']);
	expect(screen.getByRole('img').getAttribute('src')).toBe(
		base + '/sample/photos/0?vb_expected_user=alice'
	);
	await fireEvent.click(screen.getByRole('button', { name: 'Photo 2' }));
	expect(screen.getByRole('img').getAttribute('src')).toBe(
		base + '/sample/photos/1?vb_expected_user=alice'
	);
	await fireEvent.input(total, { target: { value: '10.50' } });
	await fireEvent.input(screen.getByLabelText('Tip'), { target: { value: '0.00' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Add item' }));
	await fireEvent.input(screen.getByLabelText('Description'), { target: { value: 'Bread roll' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	await screen.findByText('Draft saved.');
	const payload = JSON.parse(writes[1].body as string);
	expect(payload.expected_revision).toBe(1);
	expect(payload.expected).toMatchObject({
		total: '10.50',
		tip: '0.00',
		line_items: [{ description: 'Bread roll' }]
	});
	expect(payload.expected.taxes ?? null).toBeNull();
	expect(payload.expected.line_items[0].quantity ?? null).toBeNull();
	expect(payload.confirm).toBe(false);
	const confirm = screen.getByRole('button', { name: 'Confirm reference' }) as HTMLButtonElement;
	expect(confirm.disabled).toBe(true);
	await fireEvent.click(
		screen.getByLabelText('I checked the expected values against the original photos.')
	);
	await fireEvent.click(confirm);
	expect((await screen.findByRole('link', { name: 'Use in benchmark' })).getAttribute('href')).toBe(
		'/benchmarks?case=sample'
	);
	expect(JSON.parse(writes[2].body as string)).toMatchObject({
		expected_revision: 2,
		confirm: true
	});
	expect(requests.every((url) => url === base || url === base + '/sample')).toBe(true);
});

it('preserves edits on revision conflict and reloads only on explicit request', async () => {
	session.set(admin);
	let reads = 0;
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string, init: RequestInit) => {
			if (url === base) return Response.json([specimen]);
			if (init.method === 'PATCH')
				return Response.json({ detail: 'case_revision_changed' }, { status: 409 });
			reads += 1;
			return Response.json({
				...specimen,
				revision: reads,
				expected: { merchant_name: reads === 1 ? 'Original' : 'Changed elsewhere' }
			});
		})
	);
	render(Page);
	await fireEvent.click(await screen.findByRole('button', { name: 'Corner shop' }));
	const merchant = (await screen.findByLabelText('Merchant')) as HTMLInputElement;
	await fireEvent.input(merchant, { target: { value: 'My edits' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	await screen.findByRole('alert');
	expect(merchant.value).toBe('My edits');
	await fireEvent.click(
		screen.getByRole('button', { name: 'Discard edits and load current version' })
	);
	await screen.findByDisplayValue('Changed elsewhere');
});

it('preserves German decimal input and distinguishes confirmed empty lists from unknowns', async () => {
	session.set(admin);
	setLocale('de');
	let write: RequestInit | undefined;
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string, init: RequestInit) => {
			if (url === base) return Response.json([specimen]);
			if (init.method === 'PATCH') write = init;
			return Response.json({ ...specimen, confirmed: !!write });
		})
	);
	render(Page);
	await fireEvent.click(await screen.findByRole('button', { name: 'Corner shop' }));
	const consent = await screen.findByLabelText(
		'Ich habe die Sollwerte mit den Originalfotos abgeglichen.'
	);
	await fireEvent.click(consent);
	await fireEvent.click(screen.getByRole('button', { name: 'Referenz bestätigen' }));
	expect(write).toBeUndefined();
	expect((await screen.findByRole('alert')).textContent).toContain('Sollwert');
	await fireEvent.click(
		screen.getByLabelText('Ich habe geprüft, dass keine Steuerzeilen vorhanden sind.')
	);
	expect((consent as HTMLInputElement).checked).toBe(false);
	const total = within(screen.getByRole('group', { name: 'Beträge' })).getByLabelText('Gesamt');
	await fireEvent.input(total, { target: { value: '10,50' } });
	await fireEvent.click(consent);
	await fireEvent.click(screen.getByRole('button', { name: 'Referenz bestätigen' }));
	await screen.findByRole('link', { name: 'Im Benchmark verwenden' });
	const doc = JSON.parse(write!.body as string).expected;
	expect(doc).toMatchObject({ taxes: [], total: '10.50' });
	expect(doc.line_items ?? null).toBeNull();
});

it('hides case contents when the actor changes during a pending private read', async () => {
	session.set(admin);
	let finish!: (r: Response) => void;
	const response = new Promise<Response>((r) => (finish = r));
	const fetch = vi.fn(async (url: string) => (url === base ? Response.json([specimen]) : response));
	vi.stubGlobal('fetch', fetch);
	render(Page);
	await fireEvent.click(await screen.findByRole('button', { name: 'Corner shop' }));
	session.set({ ...admin, id: 'bob' });
	finish(Response.json({ ...specimen, expected: { merchant_name: 'Private merchant' } }));
	await response;
	expect(screen.queryByDisplayValue('Private merchant')).toBeNull();
	expect(screen.queryByRole('img')).toBeNull();
	expect(screen.queryByText(specimen.name)).toBeNull();
	expect(fetch).toHaveBeenCalledTimes(2);
});

it('authors optional receipt and item labels without changing the shared tag vocabulary', async () => {
	session.set(admin);
	const requests: string[] = [];
	let updated: {
		expected?: {
			category: string;
			line_items: { spending_category: string; tags: string[] | null }[];
		};
	} = {};
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string, init: RequestInit) => {
			requests.push(url);
			if (url === base) return Response.json([specimen]);
			if (init.method === 'PATCH') updated = JSON.parse(init.body as string);
			return Response.json({
				...specimen,
				...updated,
				expected: updated.expected ?? {
					line_items: [{ description: 'milk' }, { description: 'apple' }, { description: 'bread' }]
				}
			});
		})
	);
	render(Page);
	await fireEvent.click(await screen.findByRole('button', { name: 'Corner shop' }));
	await fireEvent.change(await screen.findByLabelText('Category'), {
		target: { value: 'grocery' }
	});
	await fireEvent.change(screen.getAllByLabelText('Spending category')[0], {
		target: { value: 'milk' }
	});
	const flags = screen.getAllByLabelText('Score tags for this item');
	await fireEvent.click(flags[0]);
	await fireEvent.click(flags[1]);
	const tags = screen.getAllByLabelText('Tags (one per line)')[0] as HTMLTextAreaElement;
	setLocale('de');
	await fireEvent.input(tags, { target: { value: 'Kaffee\nMy WEEKLY tag\nfrischware' } });
	expect(tags.value).toBe('Kaffee\nMy WEEKLY tag\nfrischware');
	await fireEvent.blur(tags);
	expect(tags.value).toBe('Kaffee\nMy WEEKLY tag\nFrischware');
	setLocale('en');
	await screen.findByDisplayValue('Coffee My WEEKLY tag Fresh produce and perishables');
	await fireEvent.click(screen.getByRole('button', { name: 'Save draft' }));
	await screen.findByText('Draft saved.');
	expect(updated.expected!.category).toBe('grocery');
	expect(updated.expected!.line_items[0]).toMatchObject({
		spending_category: 'milk',
		tags: ['coffee', 'My WEEKLY tag', 'fresh']
	});
	expect(updated.expected!.line_items[1].tags).toEqual([]);
	expect(updated.expected!.line_items[2].tags ?? null).toBeNull();
	expect(requests).toEqual([base, base + '/sample', base + '/sample']);
});

it('retains the exact custom identity when removing its German display-name twin from a benchmark', async () => {
	session.set(admin);
	let saved: string[] | undefined;
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string, init: RequestInit) => {
			if (url === base) return Response.json([specimen]);
			if (init.method === 'PATCH')
				saved = JSON.parse(init.body as string).expected.line_items[0].tags;
			return Response.json({
				...specimen,
				expected: { line_items: [{ description: 'sample', tags: saved ?? ['milk', 'Milch'] }] }
			});
		})
	);
	render(Page);
	await fireEvent.click(await screen.findByRole('button', { name: 'Corner shop' }));
	setLocale('de');
	const field = (await screen.findByLabelText(
		'Schlagwörter (eines pro Zeile)'
	)) as HTMLTextAreaElement;
	expect(field.value).toBe('milk\nMilch');
	expect(screen.getByText('Milch (Eigenes Schlagwort)')).toBeTruthy();
	await fireEvent.input(field, { target: { value: 'Milch' } });
	await fireEvent.blur(field);
	setLocale('en');
	await fireEvent.click(await screen.findByRole('button', { name: 'Save draft' }));
	await screen.findByText('Draft saved.');
	expect(saved).toEqual(['Milch']);
});
