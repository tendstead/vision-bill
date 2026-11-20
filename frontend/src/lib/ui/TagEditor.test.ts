import { session } from '$lib/session-lifetime';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import TagEditor from './TagEditor.svelte';

const OPTIONS = ['alcohol', 'beverage', 'coffee', 'food', 'fresh', 'household'];

function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		status,
		headers: { 'Content-Type': 'application/json' }
	});
}

function setupFetch() {
	const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
		const url = String(input);
		if (url.endsWith('/tags')) {
			if (init?.method === 'POST') {
				const { name } = JSON.parse(String(init.body)) as { name: string };
				const normalized = name.trim().split(/\s+/).join(' ').toLowerCase();
				return jsonResponse({ name: normalized, created: true });
			}
			return jsonResponse(OPTIONS);
		}
		return jsonResponse({}, 404);
	});
	vi.stubGlobal('fetch', fetchMock);
	return fetchMock;
}

function mount(value: string[], tagOptions: string[] = OPTIONS) {
	return render(TagEditor, { value, tagOptions, id: 'li-0' });
}

async function openDropdown() {
	await fireEvent.click(screen.getByRole('combobox'));
	return screen.findByRole('listbox');
}

beforeEach(() => {
	vi.stubGlobal('fetch', setupFetch());
});

beforeEach(() =>
	session.set({ id: 'viewer', username: 'alice', is_admin: false, can_see_all: false })
);

afterEach(() => {
	// Vitest runs without `globals: true`, so testing-library's automatic
	// cleanup (which registers a global afterEach) is not active — unmount
	// manually so `screen` queries never see stale renders.
	cleanup();
	vi.unstubAllGlobals();
});

describe('TagEditor', () => {
	it('renders standard tags as chips and suggested tags with a badge', () => {
		mount(['food', 'custom-tag']);
		expect(screen.getByText('food')).not.toBeNull();
		expect(screen.getByText('Suggested')).not.toBeNull();
		expect(screen.getByText('custom-tag')).not.toBeNull();
		// The vocabulary itself is not rendered until the dropdown opens.
		expect(screen.queryByText('alcohol')).toBeNull();
	});

	it('opens the dropdown on click and lists every option', async () => {
		mount([]);
		expect(screen.queryByRole('listbox')).toBeNull();
		await openDropdown();
		for (const option of OPTIONS) {
			expect(screen.getByRole('option', { name: option })).not.toBeNull();
		}
	});

	it('marks selected options with aria-selected', async () => {
		mount(['food']);
		await openDropdown();
		expect(screen.getByRole('option', { name: 'food' }).getAttribute('aria-selected')).toBe('true');
		expect(screen.getByRole('option', { name: 'beverage' }).getAttribute('aria-selected')).toBe(
			'false'
		);
	});

	it('filters options while typing in the search field', async () => {
		mount([]);
		await openDropdown();
		const search = screen.getByPlaceholderText('Search tags…');
		await fireEvent.input(search, { target: { value: 'f' } });
		expect(screen.getByRole('option', { name: 'food' })).not.toBeNull();
		expect(screen.getByRole('option', { name: 'fresh' })).not.toBeNull();
		expect(screen.queryByRole('option', { name: 'beverage' })).toBeNull();
	});

	it('toggles tags when options are clicked and keeps the dropdown open', async () => {
		const value = ['food'];
		mount(value);
		await openDropdown();
		await fireEvent.click(screen.getByRole('option', { name: 'beverage' }));
		expect(value).toEqual(['food', 'beverage']);
		expect(screen.getByRole('listbox')).not.toBeNull();
		await fireEvent.click(screen.getByRole('option', { name: 'food' }));
		expect(value).toEqual(['beverage']);
	});

	it('creates a new tag from the create row', async () => {
		const fetchMock = vi.mocked(fetch);
		const value: string[] = [];
		mount(value);
		await openDropdown();
		const search = screen.getByPlaceholderText('Search tags…');
		await fireEvent.input(search, { target: { value: 'veggie' } });
		expect(screen.getByText('No tags match “veggie”.')).not.toBeNull();
		await fireEvent.click(screen.getByRole('option', { name: 'Create “veggie”' }));
		await waitFor(() =>
			expect(fetchMock).toHaveBeenCalledWith(
				expect.stringContaining('/tags'),
				expect.objectContaining({ method: 'POST' })
			)
		);
		await waitFor(() => expect(value).toEqual(['veggie']));
		// The search clears after `await api.createTag(...)` resolves, so poll
		// for the final DOM state rather than asserting synchronously.
		await waitFor(() => expect((search as HTMLInputElement).value).toBe(''));
	});

	it('does not offer a create row for an exact match', async () => {
		mount([]);
		await openDropdown();
		const search = screen.getByPlaceholderText('Search tags…');
		await fireEvent.input(search, { target: { value: 'FOOD' } });
		expect(screen.queryByRole('option', { name: /Create/ })).toBeNull();
		expect(screen.getByRole('option', { name: 'food' })).not.toBeNull();
	});

	it('removes the last chip with Backspace on an empty search', async () => {
		const value = ['food', 'beverage'];
		mount(value);
		await openDropdown();
		const search = screen.getByPlaceholderText('Search tags…');
		await fireEvent.keyDown(search, { key: 'Backspace' });
		expect(value).toEqual(['food']);
	});

	it('closes the dropdown with Escape and clears the search', async () => {
		mount([]);
		await openDropdown();
		const search = screen.getByPlaceholderText('Search tags…');
		await fireEvent.input(search, { target: { value: 'f' } });
		await fireEvent.keyDown(search, { key: 'Escape' });
		expect(screen.queryByRole('listbox')).toBeNull();
		// The dropdown (and its search field) is gone; the inline search input
		// back in the box must be empty again.
		expect((screen.getByPlaceholderText('Search or add tags…') as HTMLInputElement).value).toBe('');
	});

	it('navigates options with the arrow keys and toggles with Enter', async () => {
		const value: string[] = [];
		mount(value);
		await openDropdown();
		const search = screen.getByPlaceholderText('Search tags…');
		// The first row is highlighted as soon as the dropdown opens, so the
		// first ArrowDown moves the highlight to the second row.
		await fireEvent.keyDown(search, { key: 'ArrowDown' });
		await fireEvent.keyDown(search, { key: 'Enter' });
		expect(value).toEqual(['beverage']);
		await fireEvent.keyDown(search, { key: 'ArrowDown' });
		await fireEvent.keyDown(search, { key: 'Enter' });
		expect(value).toEqual(['beverage', 'coffee']);
	});

	it('promotes a suggested tag with Keep and removes it with Remove', async () => {
		const fetchMock = vi.mocked(fetch);
		const value = ['custom-tag'];
		mount(value);
		await fireEvent.click(
			screen.getByRole('button', { name: 'Keep “custom-tag” as standard tag' })
		);
		await waitFor(() =>
			expect(fetchMock).toHaveBeenCalledWith(
				expect.stringContaining('/tags'),
				expect.objectContaining({ method: 'POST' })
			)
		);
		const removeButton = screen.getByRole('button', { name: 'Remove “custom-tag”' });
		await fireEvent.click(removeButton);
		expect(value).toEqual([]);
	});

	it('shows the empty-vocabulary hint instead of options', async () => {
		mount([], []);
		await openDropdown();
		expect(screen.getByText('No tags yet — type to create one.')).not.toBeNull();
	});
});
it('blocks an already-open custom create option while its parent is saving', async () => {
	const value: string[] = [];
	const view = render(TagEditor, { value, tagOptions: OPTIONS, id: 'disabled-tag' });
	await fireEvent.click(screen.getByRole('button', { name: 'Tags' }));
	await fireEvent.input(screen.getByPlaceholderText('Search tags…'), { target: { value: 'late' } });
	await view.rerender({ value, tagOptions: OPTIONS, id: 'disabled-tag', disabled: true });
	await fireEvent.click(screen.getByRole('option', { name: 'Create “late”' }));
	expect(value).toEqual([]);
	expect(vi.mocked(fetch).mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(0);
});

it('does not attach a pending tag to a successor value after props change', async () => {
	const { api } = await import('$lib/api/client');
	let finish!: (value: { name: string; created: boolean }) => void;
	const create = vi.spyOn(api, 'createTag').mockImplementation(
		() =>
			new Promise((resolve) => {
				finish = resolve;
			})
	);
	const original: string[] = [],
		successor: string[] = [];
	const view = render(TagEditor, { value: original, tagOptions: OPTIONS, id: 'original' });
	await fireEvent.click(screen.getByRole('button', { name: 'Tags' }));
	await fireEvent.input(screen.getByPlaceholderText('Search tags…'), { target: { value: 'late' } });
	await fireEvent.click(screen.getByRole('option', { name: 'Create “late”' }));
	await view.rerender({ value: successor, tagOptions: OPTIONS, id: 'successor' });
	finish({ name: 'late', created: true });
	await new Promise((resolve) => setTimeout(resolve, 0));
	expect(successor).toEqual([]);
	expect(original).toEqual([]);
	create.mockRestore();
});

it.each(['create', 'keep'] as const)(
	'suppresses abandoned %s errors and pending effects after owner change',
	async (action) => {
		const { api } = await import('$lib/api/client');
		const { snackbar } = await import('$lib/ui/snackbar.svelte');
		let reject!: (error: Error) => void;
		const create = vi
			.spyOn(api, 'createTag')
			.mockImplementationOnce(
				() =>
					new Promise((_, r) => {
						reject = r;
					})
			)
			.mockImplementation(() => new Promise(() => {}));
		const notify = vi.spyOn(snackbar, 'notify');
		const pending = vi.fn();
		const original = action === 'keep' ? ['custom-tag'] : [];
		const view = render(TagEditor, {
			value: original,
			tagOptions: OPTIONS,
			id: 'a',
			onPendingChange: pending
		});
		if (action === 'keep')
			await fireEvent.click(
				screen.getByRole('button', { name: 'Keep “custom-tag” as standard tag' })
			);
		else {
			await openDropdown();
			await fireEvent.input(screen.getByPlaceholderText('Search tags…'), {
				target: { value: 'late' }
			});
			await fireEvent.click(screen.getByRole('option', { name: 'Create “late”' }));
		}
		await view.rerender({ value: [], tagOptions: OPTIONS, id: 'b', onPendingChange: pending });
		await openDropdown();
		await fireEvent.input(screen.getByPlaceholderText('Search tags…'), {
			target: { value: 'new' }
		});
		await fireEvent.click(screen.getByRole('option', { name: 'Create “new”' }));
		expect(create).toHaveBeenCalledTimes(2);
		reject(new Error('abandoned'));
		await new Promise((r) => setTimeout(r, 0));
		expect(notify).not.toHaveBeenCalled();
		expect(pending).toHaveBeenLastCalledWith(true);
		vi.restoreAllMocks();
	}
);
it.each(['create', 'keep'] as const)(
	'suppresses disposed %s errors including vocabulary refresh failure',
	async (action) => {
		const { api } = await import('$lib/api/client');
		const { queryClient } = await import('$lib/query/client');
		const { snackbar } = await import('$lib/ui/snackbar.svelte');
		let reject!: (error: Error) => void;
		vi.spyOn(api, 'createTag').mockResolvedValue({ name: 'late', created: true });
		const refresh = vi.spyOn(queryClient, 'invalidateQueries').mockImplementation(
			() =>
				new Promise((_, r) => {
					reject = r;
				})
		);
		const notify = vi.spyOn(snackbar, 'notify');
		const view = render(TagEditor, {
			value: action === 'keep' ? ['custom-tag'] : [],
			tagOptions: OPTIONS,
			id: 'a'
		});
		if (action === 'keep')
			await fireEvent.click(
				screen.getByRole('button', { name: 'Keep “custom-tag” as standard tag' })
			);
		else {
			await openDropdown();
			await fireEvent.input(screen.getByPlaceholderText('Search tags…'), {
				target: { value: 'late' }
			});
			await fireEvent.click(screen.getByRole('option', { name: 'Create “late”' }));
		}
		await waitFor(() => expect(refresh).toHaveBeenCalled());
		view.unmount();
		reject(new Error('refresh failed'));
		await new Promise((r) => setTimeout(r, 0));
		expect(notify).not.toHaveBeenCalled();
		vi.restoreAllMocks();
	}
);
