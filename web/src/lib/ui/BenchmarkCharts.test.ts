import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it } from 'vitest';
import { buildSeries, type CompareCell, type Comparison } from '$lib/benchmark-compare';
import BenchmarkScoreDots from './BenchmarkScoreDots.svelte';
import BenchmarkDumbbell from './BenchmarkDumbbell.svelte';
import BenchmarkLatencyStrip from './BenchmarkLatencyStrip.svelte';
import BenchmarkSummaryBars from './BenchmarkSummaryBars.svelte';

afterEach(cleanup);

const R1 = '11111111-1111-1111-1111-111111111111';
const R2 = '22222222-2222-2222-2222-222222222222';
const cell = (
	run: string,
	receipt: string,
	score: number | null,
	extra: Partial<CompareCell> = {}
): CompareCell => ({
	run_id: run,
	model_id: 'm1',
	model_digest: null,
	success: score !== null,
	score,
	scores: null,
	score_delta: null,
	latency_ms: 1000,
	load_ms: null,
	generation_ms: null,
	attempts: 1,
	error: null,
	error_kind: null,
	...extra
});
const info = (id: string, label: string) => ({
	id,
	status: 'completed',
	model_ids: ['m1'],
	receipt_ids: [R1, R2],
	label,
	parent_run_id: null,
	comparison_arm: null,
	created_at: ''
});
const summary = (score: number, failed: number) => [
	{
		model_id: 'm1',
		model_digest: null,
		completed: 2,
		succeeded: 2 - failed,
		failed,
		average_score: score,
		average_confidence: 90,
		council_findings: 0,
		average_latency_ms: 1200
	}
];
const comparison: Comparison = {
	baseline_run_id: 'a',
	comparable: { comparable: true, reasons: [] },
	varied: [],
	runs: [
		{
			run: info('a', 'before'),
			baseline_run_id: null,
			model_digests: {},
			summaries: summary(0.5, 0),
			deltas: []
		},
		{
			run: info('b', 'after'),
			baseline_run_id: 'a',
			model_digests: {},
			summaries: summary(0.9, 1),
			deltas: []
		}
	],
	receipts: [
		{
			receipt_id: R1,
			cells: [
				cell('a', R1, 0.5),
				cell('b', R1, 0.9, { score_delta: 0.4, load_ms: 300, generation_ms: 600, latency_ms: 950 })
			]
		},
		{ receipt_id: R2, cells: [cell('a', R2, 0.8), cell('b', R2, null, { error: 'TimeoutError' })] }
	]
};
const series = buildSeries(comparison);

describe('benchmark charts', () => {
	it('score dots: every marker links to its receipt and focus pins the values', async () => {
		render(BenchmarkScoreDots, { comparison, series });
		const links = screen.getAllByRole('link');
		expect(links).toHaveLength(4);
		expect(links[0].getAttribute('href')).toContain(`/receipts/${R1}`);
		expect(links[3].getAttribute('href')).toContain(`/receipts/${R2}`);
		expect(links[3].getAttribute('aria-label')).toContain('failed');
		expect(screen.getByTestId('chart-pinned').textContent).toContain('Hover or focus');
		await fireEvent.focus(links[1]);
		const pinned = screen.getByTestId('chart-pinned').textContent ?? '';
		expect(pinned).toContain('11111111');
		expect(pinned).toContain('after');
		expect(pinned).toContain('90.0%');
		expect(pinned).toContain('+40.0 pp');
		expect(screen.getByRole('group').getAttribute('aria-labelledby')).toBeTruthy();
	});

	it('dumbbell: one line per comparable receipt, improvements and totals labelled', () => {
		render(BenchmarkDumbbell, { comparison, series });
		// Receipt 2 failed in the later run, so only receipt 1 is paired.
		const links = screen.getAllByRole('link');
		expect(links).toHaveLength(1);
		expect(links[0].getAttribute('href')).toContain(R1);
		expect(screen.getByText('+40.0 pp')).toBeTruthy();
	});

	it('dumbbell: says so when nothing can be paired', () => {
		render(BenchmarkDumbbell, { comparison: { ...comparison, receipts: [] }, series });
		expect(screen.getByText(/No receipt has a score/)).toBeTruthy();
	});

	it('latency strip: links per marker and explains the load split', () => {
		render(BenchmarkLatencyStrip, { comparison, series });
		expect(screen.getAllByRole('link')).toHaveLength(4); // the failed cell still has a latency
		expect(screen.getByText(/Pale bar: model load time/)).toBeTruthy();
		cleanup();
		render(BenchmarkLatencyStrip, {
			comparison: {
				...comparison,
				receipts: comparison.receipts.map((r) => ({
					...r,
					cells: r.cells.map((c) => ({ ...c, load_ms: null, generation_ms: null }))
				}))
			},
			series
		});
		expect(screen.getByText(/reports no load\/generation split/)).toBeTruthy();
	});

	it('summary bars: score, latency and failures per series with text values', () => {
		render(BenchmarkSummaryBars, { comparison, series });
		const root = screen.getByTestId('chart-summary-bars');
		expect(root.textContent).toContain('50.0%');
		expect(root.textContent).toContain('90.0%');
		expect(root.textContent).toContain('1200 ms');
		expect(screen.getAllByRole('img')).toHaveLength(1);
		expect(root.textContent).toContain('Failures');
	});
});
