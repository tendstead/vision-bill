// Types and pure helpers for the benchmark compare page and its SVG charts.

export type CompareCell = {
	run_id: string;
	model_id: string;
	model_digest: string | null;
	success: boolean;
	score: number | null;
	scores: Record<string, number | null> | null;
	score_delta: number | null;
	latency_ms: number | null;
	load_ms: number | null;
	generation_ms: number | null;
	attempts: number;
	error: string | null;
};
export type CompareSummary = {
	model_id: string;
	model_digest: string | null;
	completed: number;
	succeeded: number;
	failed: number;
	average_score: number | null;
	average_latency_ms: number | null;
};
export type CompareRunInfo = {
	id: string;
	status: string;
	model_ids: string[];
	receipt_ids: string[];
	label: string | null;
	parent_run_id: string | null;
	comparison_arm: string | null;
	created_at: string;
};
export type CompareRun = {
	run: CompareRunInfo;
	baseline_run_id: string | null;
	model_digests: Record<string, string | null>;
	summaries: CompareSummary[];
	deltas: {
		model_id: string;
		average_score: number | null;
		average_latency_ms: number | null;
		failed: number;
	}[];
};
export type Comparison = {
	baseline_run_id: string;
	comparable: {
		comparable: boolean;
		reasons: { code: string; message: string; run_ids: string[]; model_id?: string }[];
	};
	varied: { field: string; values: Record<string, unknown> }[];
	runs: CompareRun[];
	receipts: { receipt_id: string; cells: CompareCell[] }[];
};

/** Okabe-Ito palette: distinguishable with the common colour-vision deficiencies. */
export const PALETTE = [
	'#0072B2',
	'#E69F00',
	'#009E73',
	'#CC79A7',
	'#D55E00',
	'#56B4E9',
	'#B8A400',
	'#777777'
];
export const SHAPES = ['circle', 'square', 'diamond', 'triangle'] as const;
export type Shape = (typeof SHAPES)[number];

export type Series = {
	key: string;
	runId: string;
	modelId: string;
	label: string;
	color: string;
	shape: Shape;
};

export const seriesKey = (runId: string, modelId: string) => `${runId}|${modelId}`;
export const shortId = (id: string) => id.slice(0, 8);
export const runName = (run: CompareRunInfo) =>
	run.label ?? run.comparison_arm ?? `#${shortId(run.id)}`;

/** One series per run and model; shapes repeat after four, so colour and shape together identify a series. */
export function buildSeries(cmp: Comparison): Series[] {
	const models = new Set(cmp.runs.flatMap((r) => r.run.model_ids));
	const out: Series[] = [];
	for (const r of cmp.runs) {
		for (const model of r.run.model_ids) {
			const i = out.length;
			const label =
				models.size === 1
					? runName(r.run)
					: cmp.runs.length === 1
						? model
						: `${model} · ${runName(r.run)}`;
			out.push({
				key: seriesKey(r.run.id, model),
				runId: r.run.id,
				modelId: model,
				label,
				color: PALETTE[i % PALETTE.length],
				shape: SHAPES[i % SHAPES.length]
			});
		}
	}
	return out;
}

export function cellFor(
	receipt: Comparison['receipts'][number],
	runId: string,
	modelId: string
): CompareCell | undefined {
	return receipt.cells.find((c) => c.run_id === runId && c.model_id === modelId);
}

export const formatPercent = (v: number | null | undefined, digits = 1) =>
	v == null ? '—' : `${(v * 100).toFixed(digits)}%`;
/** Score differences are shown in percentage points. */
export const formatPoints = (v: number | null | undefined) =>
	v == null ? '—' : `${v > 0 ? '+' : ''}${(v * 100).toFixed(1)} pp`;
export const formatMs = (v: number | null | undefined) =>
	v == null
		? '—'
		: v >= 10_000
			? `${(v / 1000).toFixed(1)} s`
			: v < 10
				? `${Number(v.toFixed(1))} ms`
				: `${Math.round(v)} ms`;
export const formatSignedMs = (v: number | null | undefined) =>
	v == null ? '—' : `${v > 0 ? '+' : ''}${formatMs(v)}`;

/** Higher scores are better; for latency and failures lower is better. */
export function deltaTone(delta: number | null | undefined, higherIsBetter: boolean) {
	if (delta == null || Math.abs(delta) < 1e-9) return 'neutral' as const;
	return delta > 0 === higherIsBetter ? ('good' as const) : ('bad' as const);
}

export function linearScale(d0: number, d1: number, r0: number, r1: number) {
	const span = d1 - d0 || 1;
	return (v: number) => r0 + ((v - d0) / span) * (r1 - r0);
}

/** "Nice" tick values from 0 to at least max. */
export function niceTicks(max: number, target = 5): number[] {
	if (!(max > 0)) return [0, 1];
	const raw = max / target;
	const mag = 10 ** Math.floor(Math.log10(raw));
	const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
	const out: number[] = [];
	for (let i = 0; i < 100; i++) {
		const v = Number((i * step).toPrecision(12));
		out.push(v);
		if (v >= max - 1e-9) break;
	}
	return out;
}

export type DumbbellRow = {
	receiptId: string;
	seriesKey: string;
	modelId: string;
	runId: string;
	baselineRunId: string;
	from: number;
	to: number;
	delta: number;
};

/** Previous and current score of the same model on one line per receipt. */
export function dumbbellRows(cmp: Comparison): DumbbellRow[] {
	const rows: DumbbellRow[] = [];
	for (const r of cmp.runs) {
		if (!r.baseline_run_id) continue;
		for (const model of r.run.model_ids) {
			for (const receipt of cmp.receipts) {
				const to = cellFor(receipt, r.run.id, model);
				const from = cellFor(receipt, r.baseline_run_id, model);
				if (to?.score == null || from?.score == null) continue;
				rows.push({
					receiptId: receipt.receipt_id,
					seriesKey: seriesKey(r.run.id, model),
					modelId: model,
					runId: r.run.id,
					baselineRunId: r.baseline_run_id,
					from: from.score,
					to: to.score,
					delta: to.score - from.score
				});
			}
		}
	}
	return rows;
}

const REASON_TEXT: Record<string, string> = {
	model_digest_changed: 'Model updated',
	dataset_changed: 'Different receipts or expected values',
	prompt_changed: 'Prompt changed',
	schema_changed: 'Validation schema changed',
	scoring_changed: 'Scoring version changed',
	council_policy_changed: 'Council policy changed',
	run_incomplete: 'Run not finished',
	legacy_run: 'Run without stored results'
};
export const reasonTitle = (code: string) => REASON_TEXT[code] ?? code;

/** The default selection: the newest run and its parent chain (or the previous run). */
export function defaultSelection(
	runs: { id: string; parent_run_id: string | null }[],
	limit = 4
): string[] {
	if (runs.length === 0) return [];
	const byId = new Map(runs.map((r) => [r.id, r]));
	const chain: string[] = [];
	let cur: string | null = runs[0].id;
	while (cur && byId.has(cur) && !chain.includes(cur) && chain.length < limit) {
		chain.push(cur);
		cur = byId.get(cur)?.parent_run_id ?? null;
	}
	if (chain.length < 2 && runs.length > 1) chain.push(runs[1].id);
	// The oldest run is the baseline (compare treats the first id as baseline).
	return chain.reverse();
}
