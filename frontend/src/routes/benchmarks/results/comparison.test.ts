import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/svelte';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { session } from '$lib/auth';
import Page from './+page.svelte';

vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
vi.mock('$app/navigation', () => ({ goto: vi.fn(), replaceState: vi.fn() }));
const route = '/api/v1/benchmarks';
const receipts = Array.from({ length: 51 }, (_, index) => 'case-' + index);
const arm = (id: string, models: string[]) => ({
	id,
	model_ids: models,
	receipt_ids: receipts,
	scoring_version: '5',
	dataset_fingerprint: 'same',
	status: 'completed',
	experiment: {},
	comparison_id: 'group',
	comparison_arm: id,
	case_ids: receipts,
	case_names: { 'case-50': 'Weekend groceries' }
});
const vision = arm('direct_vision', ['model']);
const ocr = arm('ocr', ['model', 'text']);
const single = { ...arm('single', ['other']), comparison_id: null, comparison_arm: null };
function summary(model_id: string, completed: number, score: number) {
	return {
		model_id,
		completed,
		succeeded: completed - 1,
		failed: 1,
		all_case_average_score: score,
		average_latency_ms: 120
	};
}
function show({ completed = 51, second = ocr, history = [single, vision, ocr] } = {}) {
	window.history.replaceState(null, '', '?run=ocr');
	const fetchMock = vi.fn(async (path: string) => {
		if (path === route) return Response.json(history);
		if (path === route + '?comparison_id=group') return Response.json([vision, second]);
		const run = path.startsWith(route + '/ocr') ? second : vision;
		if (path.includes('/results?')) {
			const query = new URL(path, 'http://localhost').searchParams;
			const model = query.get('model_id');
			return Response.json(
				query.getAll('receipt_id').map((receipt_id) => ({
					version: '5',
					model_id: model,
					receipt_id,
					success: run === second,
					error: run === second ? null : 'unreadable',
					scores: { normalized: run === second ? 0.85 : 0 },
					item_fp: 0,
					item_fn: 2,
					exact_amounts: false,
					item_matches: [],
					scored_fields: ['line_items', 'total'],
					metadata: { elapsed_ms: 145, llm_ms: 110, telemetry: { stage_ms: { ocr: 30 } } }
				}))
			);
		}
		const count = run === second ? completed : 51;
		return Response.json({
			run,
			terminal: count,
			summaries: run.model_ids.map((m) => summary(m, count, run === second ? 0.85 : 0))
		});
	});
	vi.stubGlobal('fetch', fetchMock);
	render(Page);
	return fetchMock;
}
beforeEach(() => {
	session.set({ id: 'owner', username: 'admin', is_admin: true, can_see_all: false });
});
afterEach(() => {
	cleanup();
	window.history.replaceState(null, '', '/');
	vi.unstubAllGlobals();
	vi.restoreAllMocks();
});
it('opens the requested pair outside the history page and stops terminal network polling', async () => {
	let poll = () => {};
	vi.spyOn(window, 'setInterval').mockImplementation(((callback: () => void, delay: number) => {
		if (delay === 3000) poll = callback;
		return 42;
	}) as never);
	const fetchMock = show({ history: [single] });
	const table = await screen.findByRole('table', { name: 'Pipeline comparison' });
	expect(within(table).getByRole('row', { name: /^model / }).textContent).toContain('+85.00 pp');
	expect(within(table).getByRole('row', { name: /^text / }).textContent).toContain(
		'Not applicable'
	);
	expect(within(table).getByRole('row', { name: /^model / }).textContent).toContain('0.00%');
	fetchMock.mockClear();
	poll();
	await Promise.resolve();
	expect(fetchMock).not.toHaveBeenCalled();
});
it.each(['partial', 'different cases', 'different references'])(
	'withholds a model delta for %s',
	async (reason) => {
		show({
			completed: reason === 'partial' ? 10 : 51,
			second: {
				...ocr,
				receipt_ids: reason === 'different cases' ? [...receipts.slice(1), 'another'] : receipts,
				dataset_fingerprint: reason === 'different references' ? 'changed' : 'same'
			}
		});
		const table = await screen.findByRole('table', { name: 'Pipeline comparison' });
		expect(table.textContent).not.toContain('+85.00 pp');
		if (reason === 'partial') expect(table.textContent).toContain('Provisional');
	}
);
it('joins a named case beyond the first page, displays terminal failures and recorded stage times', async () => {
	const fetchMock = show();
	await fireEvent.click(await screen.findByRole('button', { name: 'Load individual results' }));
	await screen.findByText('Case comparison', { selector: 'caption' });
	await fireEvent.click(screen.getByText('Next 50', { selector: 'button' }));
	const row = (
		await screen.findByText('model · Weekend groceries', { selector: 'button' })
	).closest('tr')!;
	expect(row.textContent).toContain('+85.00 pp');
	expect(row.textContent).toContain('unreadable');
	expect(row.textContent).toContain('30 ms');
	expect(row.textContent).toContain('110 ms');
	expect(row.textContent).toContain('2');
	await waitFor(() =>
		expect(fetchMock.mock.calls.some(([url]) => url.includes('receipt_id=case-50'))).toBe(true)
	);
});

it('preserves benchmark-only case identity when the selection is reused', async () => {
	show();
	await fireEvent.click(await screen.findByRole('link', { name: 'Reuse this receipt selection' }));
	expect(sessionStorage.getItem('benchmark-reuse-receipts')).toBe('');
	expect(sessionStorage.getItem('benchmark-reuse-cases')).toBe(receipts.join(','));
	sessionStorage.clear();
});

it('withholds results when the sibling is inaccessible and recovers on manual refresh', async () => {
	window.history.replaceState(null, '', '?run=ocr');
	let available = false;
	vi.stubGlobal(
		'fetch',
		vi.fn(async (path: string) => {
			if (path === route) return Response.json([ocr]);
			if (path.includes('?comparison_id=')) return Response.json(available ? [ocr, vision] : [ocr]);
			const run = path.endsWith('/ocr') ? ocr : vision;
			return Response.json({ run, terminal: 51, summaries: [summary('model', 51, 0.5)] });
		})
	);
	render(Page);
	expect((await screen.findByRole('alert')).textContent).toContain('other run');
	expect(screen.queryByRole('table')).toBeNull();
	available = true;
	await fireEvent.click(screen.getByRole('button', { name: 'Refresh runs' }));
	await screen.findByRole('table', { name: 'Pipeline comparison' });
	expect(screen.queryByRole('alert')).toBeNull();
});

it('loads older history without replacing the selected single run', async () => {
	const runs = Array.from({ length: 51 }, (_, index) => ({
		...single,
		id: 'legacy-' + index,
		experiment: { options: { pipeline: 'direct_vision' } }
	}));
	vi.stubGlobal(
		'fetch',
		vi.fn(async (path: string) => {
			if (path === route) return Response.json(runs.slice(0, 50));
			if (path === route + '?offset=50') return Response.json(runs.slice(50));
			const run = runs.find((run) => path === route + '/' + run.id);
			return Response.json({ run, terminal: 0, summaries: [] });
		})
	);
	render(Page);
	await fireEvent.click(await screen.findByRole('button', { name: 'Load older runs' }));
	const choice = await screen.findByRole('option', { name: /#legacy-50/ });
	expect(choice.textContent).toContain('Direct vision model');
	const select = screen.getByRole('combobox') as HTMLSelectElement;
	expect(select.value).toBe('legacy-0');
	await fireEvent.change(select, { target: { value: 'legacy-50' } });
	await waitFor(() => expect(select.value).toBe('legacy-50'));
});

it('polls only the unfinished arm and stops once it completes', async () => {
	let poll = () => {};
	vi.spyOn(window, 'setInterval').mockImplementation(((callback: () => void, delay: number) => {
		if (delay === 3000) poll = callback;
		return 42;
	}) as never);
	const fetchMock = show({ completed: 1, second: { ...ocr, status: 'running' } });
	await screen.findByRole('table', { name: 'Pipeline comparison' });
	fetchMock.mockClear();
	fetchMock.mockImplementation(async () =>
		Response.json({ run: ocr, terminal: 51, summaries: [summary('model', 51, 0.9)] })
	);
	poll();
	await screen.findByText('+90.00 pp');
	expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([route + '/ocr']);
	poll();
	await Promise.resolve();
	expect(fetchMock).toHaveBeenCalledTimes(1);
});
