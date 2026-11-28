import { describe, expect, it } from 'vitest';
import {
	buildSeries,
	candidateHint,
	candidateTooltip,
	failureLabel,
	failureReason,
	labelColumnWidth,
	truncateMiddle,
	orderForCompare,
	defaultSelection,
	deltaTone,
	dumbbellRows,
	formatMs,
	formatPercent,
	formatPoints,
	niceTicks,
	linearScale,
	type CompareCell,
	type Comparison
} from './benchmark-compare';

const cell = (run: string, model: string, score: number | null): CompareCell => ({
	run_id: run,
	model_id: model,
	model_digest: null,
	success: score !== null,
	score,
	scores: null,
	score_delta: null,
	latency_ms: 100,
	load_ms: null,
	generation_ms: null,
	attempts: 1,
	error: null,
	error_kind: null
});
const run = (
	id: string,
	models: string[],
	baseline: string | null,
	label: string | null = null
) => ({
	run: {
		id,
		status: 'completed',
		model_ids: models,
		receipt_ids: ['r1', 'r2'],
		label,
		parent_run_id: null,
		comparison_arm: null,
		created_at: ''
	},
	baseline_run_id: baseline,
	model_digests: {},
	summaries: [],
	deltas: []
});
const cmp: Comparison = {
	baseline_run_id: 'a',
	comparable: { comparable: true, reasons: [] },
	varied: [],
	runs: [run('a', ['m1', 'm2'], null, 'before'), run('b', ['m1'], 'a')],
	receipts: [
		{
			receipt_id: 'r1',
			exists: true,
			image_id: null,
			cells: [cell('a', 'm1', 0.5), cell('a', 'm2', 0.9), cell('b', 'm1', 0.75)]
		},
		{
			receipt_id: 'r2',
			exists: true,
			image_id: null,
			cells: [cell('a', 'm1', 0.8), cell('b', 'm1', null)]
		}
	]
};

describe('formatting', () => {
	it('formats scores, differences and latency', () => {
		expect(formatPercent(0.8567)).toBe('85.7%');
		expect(formatPercent(null)).toBe('—');
		expect(formatPoints(0.025)).toBe('+2.5 pp');
		expect(formatPoints(-0.1)).toBe('-10.0 pp');
		expect(formatMs(250.4)).toBe('250 ms');
		expect(formatMs(12_500)).toBe('12.5 s');
		expect(formatMs(0.4)).toBe('0.4 ms');
		expect(formatMs(3)).toBe('3 ms');
	});
	it('judges the direction of a change', () => {
		expect(deltaTone(0.1, true)).toBe('good');
		expect(deltaTone(0.1, false)).toBe('bad');
		expect(deltaTone(-5, false)).toBe('good');
		expect(deltaTone(0, true)).toBe('neutral');
		expect(deltaTone(null, true)).toBe('neutral');
	});
});

describe('scales', () => {
	it('maps linearly and produces nice ticks', () => {
		expect(linearScale(0, 1, 10, 110)(0.5)).toBe(60);
		expect(niceTicks(1)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
		expect(niceTicks(0)).toEqual([0, 1]);
		const t = niceTicks(1830);
		expect(t[0]).toBe(0);
		expect(t[t.length - 1]).toBeGreaterThanOrEqual(1830);
	});
});

describe('series and differences', () => {
	it('names series by run and model only as far as needed', () => {
		const s = buildSeries(cmp);
		expect(s.map((x) => x.label)).toEqual(['m1 · before', 'm2 · before', 'm1 · #b']);
		expect(new Set(s.map((x) => x.color)).size).toBe(3);
		const single = { ...cmp, runs: [run('a', ['m1', 'm2'], null)] };
		expect(buildSeries(single).map((x) => x.label)).toEqual(['m1', 'm2']);
		const oneModel = {
			...cmp,
			runs: [run('a', ['m1'], null, 'before'), run('b', ['m1'], 'a', 'after')]
		};
		expect(buildSeries(oneModel).map((x) => x.label)).toEqual(['before', 'after']);
	});
	it('pairs the same model with its baseline and skips failures', () => {
		const rows = dumbbellRows(cmp);
		expect(rows).toHaveLength(1);
		expect(rows[0]).toMatchObject({
			receiptId: 'r1',
			modelId: 'm1',
			from: 0.5,
			to: 0.75,
			runId: 'b'
		});
		expect(rows[0].delta).toBeCloseTo(0.25);
	});
});

describe('defaultSelection', () => {
	it('follows the parent chain oldest first', () => {
		const runs = [
			{ id: 'c', parent_run_id: 'b' },
			{ id: 'b', parent_run_id: 'a' },
			{ id: 'a', parent_run_id: null },
			{ id: 'x', parent_run_id: null }
		];
		expect(defaultSelection(runs)).toEqual(['a', 'b', 'c']);
		expect(defaultSelection(runs, 2)).toEqual(['b', 'c']);
	});
	it('falls back to the previous run and handles empties', () => {
		expect(
			defaultSelection([
				{ id: 'n', parent_run_id: null },
				{ id: 'o', parent_run_id: null }
			])
		).toEqual(['o', 'n']);
		expect(defaultSelection([{ id: 'only', parent_run_id: null }])).toEqual(['only']);
		expect(defaultSelection([])).toEqual([]);
	});
});

describe('failures and candidates', () => {
	it('labels failure kinds', () => {
		expect(failureLabel('timeout')).toBe('Timed out');
		expect(failureLabel('invalid_response')).toBe('Invalid response');
		expect(failureLabel(null)).toBe('Error');
	});
	it('explains a blocked run', () => {
		const c = {
			run_id: 'x',
			comparable: false,
			selectable: false,
			reasons: [
				{ code: 'dataset_changed', message: 'The expected receipts differ', run_ids: ['x'] },
				{ code: 'model_digest_changed', message: 'updated', run_ids: ['x'], model_id: 'm1' }
			]
		};
		expect(candidateHint(c)).toBe('Different receipts or expected values, Model updated');
		expect(candidateTooltip(c)).toContain(
			'Different receipts or expected values: The expected receipts differ'
		);
		expect(candidateTooltip(c)).toContain('Model updated (m1): updated');
	});
	it('orders the selection oldest first', () => {
		const runs = [{ id: 'new' }, { id: 'mid' }, { id: 'old' }];
		expect(orderForCompare(runs, ['new', 'old', 'mid', 'new'])).toEqual(['old', 'mid', 'new']);
	});
});

describe('labels', () => {
	it('truncates in the middle and sizes the label column', () => {
		expect(truncateMiddle('short', 10)).toBe('short');
		const t = truncateMiddle('DeepSeek-V4-Pro-Qwen3.5-9B-Heretic-Q6_K · #7bf41841', 24);
		expect(t).toHaveLength(24);
		expect(t.startsWith('DeepSeek-V4')).toBe(true);
		expect(t.endsWith('#7bf41841')).toBe(true);
		expect(t).toContain('…');
		expect(labelColumnWidth(['abc'], 6, 100, 300)).toBe(100);
		expect(labelColumnWidth(['x'.repeat(30)], 6, 100, 300)).toBe(196);
		expect(labelColumnWidth(['x'.repeat(200)], 6, 100, 300)).toBe(300);
		expect(labelColumnWidth([], 6, 100, 300)).toBe(100);
	});
	it('explains old bare failure messages', () => {
		expect(failureReason({ error: 'TimeoutError', error_kind: 'timeout' })).toContain(
			'run timeout'
		);
		expect(failureReason({ error: null, error_kind: 'error' })).toContain('older version');
		expect(failureReason({ error: 'boom', error_kind: 'error' })).toBe('boom');
	});
	it('labels the context failure', () => {
		expect(failureLabel('context_too_small')).toBe('Context too small');
	});
});
