import { cleanup, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, expect, it, vi } from 'vitest';
import { tick } from 'svelte';
import { api } from './api/client';
import { signIn, signOut } from './auth';
import { queryClient } from './query/client';
import { queryKeys } from './query/keys';
import Harness from './test-fixtures/SessionLayoutHarness.svelte';
vi.mock('$app/state', () => ({ page: { url: new URL('http://localhost/receipts') } }));
vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));
afterEach(() => {
	cleanup();
	queryClient.clear();
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
});
it('same mounted SPA drops A receipts and active header collection when B signs in', async () => {
	vi.stubGlobal(
		'ResizeObserver',
		class {
			observe() {}
			disconnect() {}
		}
	);
	vi.stubGlobal('matchMedia', () => ({
		matches: false,
		addEventListener() {},
		removeEventListener() {}
	}));
	const alice = { id: 'a', username: 'alice', is_admin: false, can_see_all: false },
		bob = { ...alice, id: 'b', username: 'bob' };
	vi.spyOn(api, 'me').mockResolvedValue(alice);
	vi.spyOn(api, 'login').mockResolvedValue(bob);
	vi.spyOn(api, 'logout').mockResolvedValue();
	vi.spyOn(api, 'listImages').mockResolvedValue(
		Array.from({ length: 51 }, (_, index) => ({ id: String(index) })) as never
	);
	vi.spyOn(api, 'getActiveCollection')
		.mockResolvedValueOnce({ id: 'a', name: 'Alice private collection', color: '#ff0000' } as never)
		.mockResolvedValue(null);
	vi.spyOn(api, 'listReceipts')
		.mockResolvedValueOnce([{ id: 'a', merchant_name: 'Alice private shop' }] as never)
		.mockResolvedValue([{ id: 'b', merchant_name: 'Bob shop' }] as never);
	render(Harness);
	await screen.findByText('Alice private shop');
	await waitFor(() =>
		expect(api.listImages).toHaveBeenCalledWith(expect.objectContaining({ limit: 51 }))
	);
	expect(screen.getAllByText('50+').length).toBeGreaterThan(0);
	expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1);
	await waitFor(() =>
		expect(screen.getByRole('link', { name: 'Alice private collection' })).not.toBeNull()
	);
	queryClient.setQueryData(queryKeys.statistics(), { private: 'Alice totals' });
	await signOut();
	await tick();
	await signIn('bob', 'x');
	await tick();
	expect(screen.queryByText('Alice private shop')).toBeNull();
	expect(screen.queryByRole('link', { name: 'Alice private collection' })).toBeNull();
	await screen.findByText('Bob shop');
	expect(queryClient.getQueryData(queryKeys.statistics())).toBeUndefined();
});
