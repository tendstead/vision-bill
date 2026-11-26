import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { tick } from 'svelte';
import { session } from '$lib/auth';
import Page from './+page.svelte';
vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));
const run = (id: string) => ({
	id,
	status: 'running',
	model_ids: [],
	receipt_ids: [id],
	scoring_version: id,
	experiment: {}
});
const status = (id: string, version = id) => ({
	run: { ...run(id), scoring_version: version },
	queued: 0,
	running: 0,
	retrying: 0,
	waiting_for_model: 0,
	terminal: 1,
	summaries: []
});
function deferred<T>() {
	let resolve!: (value: T) => void;
	const promise = new Promise<T>((r) => {
		resolve = r;
	});
	return { promise, resolve };
}
let poll: () => void;
beforeEach(() => {
	session.set({ id: 'owner', username: 'admin', is_admin: true, can_see_all: false });
	const original = window.setInterval.bind(window);
	vi.spyOn(window, 'setInterval').mockImplementation(((callback: TimerHandler, delay: number) => {
		if (delay === 3000) {
			poll = callback as () => void;
			return 42;
		}
		return original(callback, delay);
	}) as never);
});
afterEach(() => {
	cleanup();
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
});
async function setup() {
	const fetchMock = vi.fn(async (url: string) =>
		Response.json(
			url === '/api/v1/benchmarks' ? [run('A'), run('B')] : status(url.endsWith('/B') ? 'B' : 'A')
		)
	);
	vi.stubGlobal('fetch', fetchMock);
	render(Page);
	await screen.findByText(/Scoring version: A/);
	return fetchMock;
}
it('drops late A status when B is selected', async () => {
	const fetchMock = await setup(),
		late = deferred<Response>();
	fetchMock.mockImplementation(async (url) =>
		url === '/api/v1/benchmarks'
			? Response.json([run('A'), run('B')])
			: url.endsWith('/A')
				? late.promise
				: Response.json(status('B'))
	);
	fetchMock.mockClear();
	poll();
	await waitFor(() =>
		expect(fetchMock).toHaveBeenCalledWith('/api/v1/benchmarks/A', expect.any(Object))
	);
	await fireEvent.change(screen.getByLabelText('Benchmark run'), { target: { value: 'B' } });
	await screen.findByText(/Scoring version: B/);
	late.resolve(Response.json(status('A')));
	await tick();
	await new Promise((r) => setTimeout(r, 0));
	expect(screen.queryByText(/Scoring version: A/)).toBeNull();
	expect(screen.getByText(/Scoring version: B/)).not.toBeNull();
});
it.each([false, true])(
	'drops late A details and errors after switching (error=%s)',
	async (fail) => {
		const fetchMock = await setup(),
			late = deferred<Response>();
		fetchMock.mockImplementation(async (url) =>
			url.includes('/results?')
				? late.promise
				: url === '/api/v1/benchmarks'
					? Response.json([run('A'), run('B')])
					: Response.json(status('B'))
		);
		await fireEvent.click(screen.getByRole('button', { name: 'Load individual results' }));
		await fireEvent.change(screen.getByLabelText('Benchmark run'), { target: { value: 'B' } });
		await screen.findByText(/Scoring version: B/);
		late.resolve(
			fail
				? new Response('', { status: 500 })
				: Response.json([{ model_id: 'A-private', receipt_id: 'a', success: false }])
		);
		await tick();
		await new Promise((r) => setTimeout(r, 0));
		expect(screen.queryByText('A-private')).toBeNull();
		expect(screen.queryByRole('alert')).toBeNull();
		expect(
			(screen.getByRole('button', { name: 'Load individual results' }) as HTMLButtonElement)
				.disabled
		).toBe(false);
	}
);
it('coalesces timer ticks until a slow refresh has delivered its result', async () => {
	const fetchMock = await setup();
	const slow = deferred<Response>();
	fetchMock.mockImplementation(async (url) =>
		url === '/api/v1/benchmarks' ? Response.json([run('A'), run('B')]) : slow.promise
	);
	fetchMock.mockClear();
	poll();
	await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
	poll();
	poll();
	await tick();
	expect(fetchMock).toHaveBeenCalledTimes(1);
	slow.resolve(Response.json(status('A', 'slow result')));
	await screen.findByText(/Scoring version: slow result/);
	poll();
	await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
});

it('clears the removed run and fences its pending details when history refresh selects a successor', async () => {
	const fetchMock = await setup(),
		late = deferred<Response>();
	fetchMock.mockImplementation(async (url) =>
		url.includes('/results?')
			? late.promise
			: url === '/api/v1/benchmarks'
				? Response.json([run('B')])
				: Response.json(status('B'))
	);
	await fireEvent.click(screen.getByRole('button', { name: 'Load individual results' }));
	await fireEvent.click(screen.getByRole('button', { name: 'Refresh runs' }));
	await screen.findByText(/Scoring version: B/);
	late.resolve(Response.json([{ model_id: 'A-private', receipt_id: 'a', success: false }]));
	await tick();
	await new Promise((r) => setTimeout(r, 0));
	expect(screen.queryByText('A-private')).toBeNull();
	expect(
		(screen.getByRole('button', { name: 'Load individual results' }) as HTMLButtonElement).disabled
	).toBe(false);
});

it('retains live status through a transient poll failure and retries the active run', async () => {
	const fetchMock = await setup();
	fetchMock.mockResolvedValueOnce(new Response('', { status: 503 }));
	poll();
	await screen.findByRole('alert');
	expect(screen.getByText(/Scoring version: A/)).toBeTruthy();
	fetchMock.mockResolvedValueOnce(Response.json(status('A', 'recovered')));
	poll();
	await screen.findByText(/Scoring version: recovered/);
	expect(screen.queryByRole('alert')).toBeNull();
});

it('drops previous status after the run becomes inaccessible', async () => {
	const fetchMock = await setup();
	fetchMock.mockResolvedValueOnce(new Response('', { status: 404 }));
	poll();
	await screen.findByRole('alert');
	expect(screen.queryByText(/Scoring version: A/)).toBeNull();
});

it('refreshes the open single-run detail page when progress changes during its request', async () => {
	const fetchMock = await setup();
	const late = deferred<Response>();
	let secondPageRequests = 0;
	const rows = Array.from({ length: 50 }, (_, index) => ({
		model_id: 'm',
		receipt_id: String(index),
		success: false
	}));
	fetchMock.mockImplementation(async (path) => {
		if (path.endsWith('offset=0')) return Response.json(rows);
		if (path.endsWith('offset=50')) {
			secondPageRequests++;
			if (secondPageRequests === 1) return late.promise;
			return Response.json([
				{ model_id: 'final model', receipt_id: 'completed-case', success: false }
			]);
		}
		return Response.json({
			...status('A', 'finished'),
			run: { ...run('A'), status: 'completed', scoring_version: 'finished' },
			terminal: 52
		});
	});
	await fireEvent.click(screen.getByText('Load individual results', { selector: 'button' }));
	await waitFor(() =>
		expect(
			(screen.getByText('Next 50', { selector: 'button' }) as HTMLButtonElement).disabled
		).toBe(false)
	);
	await fireEvent.click(screen.getByText('Next 50', { selector: 'button' }));
	poll();
	await screen.findByText(/Scoring version: finished/);
	late.resolve(Response.json([]));
	await screen.findByText('final model');
	expect(secondPageRequests).toBe(2);
});
