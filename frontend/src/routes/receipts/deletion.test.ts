import { QueryObserver } from '@tanstack/svelte-query';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/svelte';
import { get } from 'svelte/store';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { api } from '$lib/api/client';
import { session, signIn } from '$lib/auth';
import { queryClient } from '$lib/query/client';
import { queryKeys } from '$lib/query/keys';
import { snackbar } from '$lib/ui/snackbar.svelte';
import Page from './+page.svelte';

vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));

const alice = { id: 'a', username: 'alice', is_admin: false, can_see_all: false };
const bob = { ...alice, id: 'b', username: 'bob' };
const receipt = {
	id: 'receipt',
	merchant_name: 'Shop',
	total: '12.00',
	currency: 'EUR',
	image_id: null,
	status: 'unverified'
};

function deferred<T>() {
	let resolve!: (value: T) => void;
	const promise = new Promise<T>((done) => {
		resolve = done;
	});
	return { promise, resolve };
}

beforeEach(() => {
	session.set(alice);
	vi.spyOn(api, 'login').mockResolvedValue(bob);
	vi.spyOn(api, 'listReceipts').mockResolvedValue([receipt] as never);
});

afterEach(() => {
	cleanup();
	snackbar.clear();
	queryClient.clear();
	session.set(null);
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
});

async function confirmDelete() {
	const receiptLink = await screen.findByRole('link', { name: /Shop/ });
	expect(receiptLink.tagName).toBe('A');
	expect(receiptLink.getAttribute('href')).toBe('/receipts/receipt');
	expect(receiptLink.querySelector('button')).toBeNull();
	await fireEvent.click(screen.getByRole('button', { name: 'Remove' }));
	await fireEvent.click(
		within(screen.getByRole('dialog', { name: 'Delete receipt?' })).getByRole('button', {
			name: 'Delete'
		})
	);
}

const settle = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

it.each(['account change', 'view destruction'])(
	'prevents a committed deletion response from notifying after %s',
	async (boundary) => {
		const held = deferred<Response>();
		const fetchMock = vi.fn().mockReturnValue(held.promise);
		vi.stubGlobal('fetch', fetchMock);

		const view = render(Page);
		await confirmDelete();
		expect(fetchMock).toHaveBeenCalledExactlyOnceWith(
			'/api/v1/receipts/receipt',
			expect.objectContaining({ method: 'DELETE' })
		);
		if (boundary === 'account change') await signIn('bob', 'secret');
		else view.unmount();

		held.resolve(Response.json({ deleted: 'receipt' }));
		await settle();
		expect(get(snackbar)).toEqual([]);
	}
);

it('does not refetch the successor images when the old receipt invalidation finishes', async () => {
	vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json({ deleted: 'receipt' })));
	const first = deferred<void>();
	const invalidate = queryClient.invalidateQueries.bind(queryClient);
	const invalidation = vi
		.spyOn(queryClient, 'invalidateQueries')
		.mockImplementationOnce(async (filters) => {
			await invalidate(filters);
			await first.promise;
		});
	render(Page);
	await confirmDelete();
	await waitFor(() => expect(invalidation).toHaveBeenCalledWith({ queryKey: ['receipts'] }));

	await signIn('bob', 'secret');
	const fetchImages = vi.fn().mockResolvedValue(['Bob image']);
	const observer = new QueryObserver(queryClient, {
		queryKey: queryKeys.images(),
		queryFn: fetchImages
	});
	const stop = observer.subscribe(() => undefined);
	try {
		await waitFor(() => expect(observer.getCurrentResult().data).toEqual(['Bob image']));
		first.resolve();
		await settle();
		expect(get(snackbar)).toEqual([]);
		expect(fetchImages).toHaveBeenCalledOnce();
	} finally {
		stop();
	}
});

it.each([
	{ status: 200, type: 'success', message: 'Receipt deleted' },
	{
		status: 409,
		type: 'error',
		message: "This receipt is used in a benchmark run and can't be deleted."
	}
])('keeps the current view notification for status $status', async ({ status, type, message }) => {
	const response =
		status === 200
			? Response.json({ deleted: 'receipt' })
			: Response.json({ detail: 'benchmark reference' }, { status });
	vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response));
	render(Page);
	await confirmDelete();
	await waitFor(() =>
		expect(get(snackbar)).toEqual([expect.objectContaining({ kind: type, text: message })])
	);
});
