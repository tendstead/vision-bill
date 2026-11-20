import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/svelte';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { tick } from 'svelte';
import { page } from '$app/state';
import { goto } from '$app/navigation';
import { session } from '$lib/session-lifetime';
import { api } from '$lib/api/client';
import { queryClient } from '$lib/query/client';
import { snackbar } from '$lib/ui/snackbar.svelte';
import Page from './+page.svelte';
vi.mock('$app/state', async () => {
	const { SvelteURL } = await import('svelte/reactivity');
	return {
		page: {
			url: new SvelteURL('http://localhost/collections/00000000-0000-4000-8000-000000000001')
		}
	};
});
vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));
const a = '00000000-0000-4000-8000-000000000001',
	b = '00000000-0000-4000-8000-000000000002';
const collection = (id: string) => ({
	id,
	name: id === a ? 'Alpha' : 'Beta',
	color: null,
	start_date: null,
	end_date: null,
	active: false,
	created_at: null,
	receipts: [],
	totals: [],
	receipt_count: 0
});
beforeEach(() => {
	session.set({ id: 'owner', username: 'alice', is_admin: false, can_see_all: false });
	queryClient.clear();
	page.url.pathname = `/collections/${a}`;
	vi.spyOn(api, 'getCollection').mockImplementation(async (id) => collection(id));
	vi.spyOn(api, 'getStatistics').mockResolvedValue(null as never);
});
afterEach(() => {
	cleanup();
	queryClient.clear();
	vi.restoreAllMocks();
});
async function switchToB() {
	page.url.pathname = `/collections/${b}`;
	await tick();
	await screen.findByRole('heading', { name: 'Beta' });
}
it.each(['edit', 'delete'])('closes the originating %s dialog on route reuse', async (kind) => {
	const update = vi.spyOn(api, 'updateCollection');
	const remove = vi.spyOn(api, 'deleteCollection');
	render(Page);
	await screen.findByRole('heading', { name: 'Alpha' });
	await fireEvent.click(
		screen.getByRole('button', { name: kind === 'edit' ? 'Edit collection' : 'Delete' })
	);
	expect(screen.getByRole('dialog')).not.toBeNull();
	await switchToB();
	expect(screen.queryByRole('dialog')).toBeNull();
	expect(update).not.toHaveBeenCalled();
	expect(remove).not.toHaveBeenCalled();
	await fireEvent.click(screen.getByRole('button', { name: 'Edit collection' }));
	expect((screen.getByLabelText('Name') as HTMLInputElement).value).toBe('Beta');
});
it.each(['edit', 'delete'])(
	'keeps late successful A %s durable without changing the B view/cache/navigation',
	async (kind) => {
		let finish!: () => void;
		const pending = new Promise<void>((r) => {
			finish = r;
		});
		const mutation =
			kind === 'edit'
				? vi.spyOn(api, 'updateCollection').mockImplementation(async () => {
						await pending;
						return collection(a);
					})
				: vi.spyOn(api, 'deleteCollection').mockReturnValue(pending);
		render(Page);
		await screen.findByRole('heading', { name: 'Alpha' });
		await fireEvent.click(
			screen.getByRole('button', { name: kind === 'edit' ? 'Edit collection' : 'Delete' })
		);
		await fireEvent.click(
			within(screen.getByRole('dialog')).getByRole('button', {
				name: kind === 'edit' ? 'Save' : 'Delete'
			})
		);
		expect(mutation.mock.calls[0][0]).toBe(a);
		await switchToB();
		await fireEvent.click(screen.getByRole('button', { name: 'Edit collection' }));
		const invalidate = vi.spyOn(queryClient, 'invalidateQueries');
		finish();
		await pending;
		await tick();
		expect(screen.getByRole('dialog')).not.toBeNull();
		expect((screen.getByLabelText('Name') as HTMLInputElement).value).toBe('Beta');
		expect(goto).not.toHaveBeenCalled();
		expect(invalidate).not.toHaveBeenCalled();
	}
);
it('ignores an abandoned edit error and finally when B is saving', async () => {
	let reject!: (error: Error) => void;
	vi.spyOn(api, 'updateCollection')
		.mockImplementationOnce(
			() =>
				new Promise((_, r) => {
					reject = r;
				})
		)
		.mockImplementation(() => new Promise(() => {}));
	const notify = vi.spyOn(snackbar, 'notify');
	render(Page);
	await screen.findByRole('heading', { name: 'Alpha' });
	await fireEvent.click(screen.getByRole('button', { name: 'Edit collection' }));
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	await switchToB();
	await fireEvent.click(screen.getByRole('button', { name: 'Edit collection' }));
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	reject(new Error('late A'));
	await tick();
	await waitFor(() =>
		expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(true)
	);
	expect(notify).not.toHaveBeenCalled();
});
