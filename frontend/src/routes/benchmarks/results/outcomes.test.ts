import { cleanup, fireEvent, render, screen, within } from '@testing-library/svelte';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { session } from '$lib/auth';
import type { IndividualResult } from '$lib/benchmark-experiments';
import type { BenchmarkSummary } from '$lib/types';
import { setLocale } from '$lib/i18n';
import Page from './+page.svelte';

vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));

const route = '/api/v1/benchmarks';
const summary: BenchmarkSummary = {
	model_id: 'model',
	model_digest: null,
	completed: 10,
	succeeded: 1,
	failed: 9,
	average_score: 1,
	all_case_average_score: 0.1,
	full_receipt_correct_rate: 0.1,
	average_confidence: null,
	average_attempts: null,
	average_latency_ms: null,
	council_candidates: 0,
	council_findings: 0
};

function showResults({
	runSummary = summary,
	version = '4',
	details = [] as IndividualResult[]
} = {}) {
	const run = {
		id: 'run',
		status: runSummary.completed === 10 ? 'completed' : 'running',
		model_ids: ['model'],
		receipt_ids: Array.from({ length: 10 }, (_, index) => String(index)),
		scoring_version: version,
		experiment: {}
	};
	const status = {
		run,
		queued: 10 - runSummary.completed,
		running: 0,
		retrying: 0,
		waiting_for_model: 0,
		terminal: runSummary.completed,
		summaries: [runSummary]
	};
	vi.stubGlobal(
		'fetch',
		vi.fn(async (url: string) => {
			if (url === route) return Response.json([run]);
			if (url === `${route}/run`) return Response.json(status);
			if (url === `${route}/run/results?limit=50&offset=0`) return Response.json(details);
			throw new Error(`Unexpected request: ${url}`);
		})
	);
	render(Page);
}

function column(heading: string) {
	const header = screen.getByRole('columnheader', { name: heading }) as HTMLTableCellElement;
	return within(screen.getAllByRole('row')[1]).getAllByRole('cell')[header.cellIndex];
}

beforeEach(() => {
	setLocale('en');
	session.set({ id: 'owner', username: 'admin', is_admin: true, can_see_all: false });
});
afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
	setLocale('en');
});

it('shows both v4 rates over completed cases and labels the successful-only average', async () => {
	showResults();
	await screen.findByRole('cell', { name: /model/ });
	expect(column('Fully correct receipts').textContent).toContain('10.00%');
	expect(column('Fully correct receipts').textContent).toContain('10 completed cases');
	expect(column('All-case score').textContent).toContain('10.00%');
	expect(column('All-case score').textContent).toContain('10 completed cases');
	expect(column('Average score (successful cases only)').textContent).toContain('100.00%');
	expect(column('Average score (successful cases only)').textContent).toContain(
		'1 successful case'
	);
});

it('uses the distinct backend rates and model completion count while cases are pending', async () => {
	showResults({
		runSummary: {
			...summary,
			completed: 4,
			succeeded: 3,
			failed: 1,
			average_score: 0.8,
			all_case_average_score: 0.6,
			full_receipt_correct_rate: 0.25
		}
	});
	await screen.findByRole('cell', { name: /model/ });
	expect(column('Fully correct receipts').textContent).toContain('25.00%');
	expect(column('All-case score').textContent).toContain('60.00%');
	expect(column('All-case score').textContent).toContain('4 completed cases');
	expect(column('Average score (successful cases only)').textContent).toContain(
		'3 successful cases'
	);
});

it('shows measured zero rates for an all-failed v4 model', async () => {
	showResults({
		runSummary: {
			...summary,
			succeeded: 0,
			failed: 10,
			average_score: null,
			all_case_average_score: 0,
			full_receipt_correct_rate: 0
		}
	});
	await screen.findByRole('cell', { name: /model/ });
	expect(column('Fully correct receipts').textContent).toContain('0.00%');
	expect(column('All-case score').textContent).toContain('0.00%');
	expect(column('Average score (successful cases only)').textContent).toContain('Unavailable');
});

it('shows historical v4 fields as unavailable and keeps the old average', async () => {
	showResults({
		version: '3',
		runSummary: { ...summary, all_case_average_score: null, full_receipt_correct_rate: null }
	});
	await screen.findByRole('cell', { name: /model/ });
	expect(column('Fully correct receipts').textContent).toBe('Unavailable');
	expect(column('All-case score').textContent).toBe('Unavailable');
	expect(column('Average score (successful cases only)').textContent).toContain('100.00%');
});

it('shows a readable message for a benchmark result with a native GPU memory failure', async () => {
	showResults({
		details: [
			{
				model_id: 'model',
				receipt_id: 'ocr-failure',
				success: false,
				error: 'ocr_gpu_out_of_memory'
			}
		]
	});
	await fireEvent.click(await screen.findByRole('button', { name: 'Load individual results' }));
	await fireEvent.click(await screen.findByText(/ocr-failure/));
	await screen.findByText('The local OCR GPU ran out of video memory.');
});

it('distinguishes a successful extraction with incorrect amounts from a fully correct receipt', async () => {
	showResults({
		details: [
			{
				model_id: 'model',
				receipt_id: 'wrong-amount',
				success: true,
				version: '4',
				legacy_detail_version: '3',
				scored_fields: ['line_items', 'total'],
				extraction_failed: false,
				scores: { normalized: 0.82, strict: 0.82, classification: 1 },
				rows: { missing: 0, extra: 0, matched: 2 },
				differences: [{ path: 'total', expected: '3.10', actual: '3.11', equivalent: false }],
				full_receipt_correct: false,
				item_tp: 1,
				item_fp: 1,
				item_fn: 1,
				item_matches: [[0, 1]],
				precision: 0.5,
				recall: 0.5,
				exact_item_count: true,
				exact_item_order: false,
				exact_amounts: false,
				metadata: {
					classification_enrichment: {
						state: 'unavailable',
						reason_code: 'invalid_response',
						missing_fields: ['category']
					}
				}
			}
		]
	});
	await fireEvent.click(await screen.findByRole('button', { name: 'Load individual results' }));
	await fireEvent.click(await screen.findByText(/wrong-amount/));
	expect(screen.getByText('Fully correct receipt').parentElement?.textContent).toContain('No');
	expect(screen.getByText('Exact amounts').parentElement?.textContent).toContain('No');
	expect(screen.getByText('Exact item order').parentElement?.textContent).toContain('No');
	expect(screen.getByText('Exact item count').parentElement?.textContent).toContain('Yes');
	expect(screen.getByText('Matched item occurrences').parentElement?.textContent).toContain('1');
	expect(screen.getByText('Extra item occurrences').parentElement?.textContent).toContain('1');
	expect(screen.getByText('Missing item occurrences').parentElement?.textContent).toContain('1');
	expect(screen.getByText('Item precision').parentElement?.textContent).toContain('50.0%');
	expect(screen.getByText('Item recall').parentElement?.textContent).toContain('50.0%');
	expect(screen.getByText('Matched item positions').parentElement?.textContent).toContain('1 → 2');
	expect(screen.getByText('"3.11"')).not.toBeNull();
	expect(screen.getByText(/Some optional labels were unavailable/).textContent).toContain(
		'category'
	);
	expect(screen.getByText(/Recorded classification score/).textContent).toContain('100.0%');
	expect(screen.queryByRole('alert')).toBeNull();
});

it('identifies absent label provenance in a historical result', async () => {
	showResults({
		version: '3',
		details: [
			{
				model_id: 'model',
				receipt_id: 'historical',
				success: true,
				version: '3',
				scores: { normalized: 1, strict: 1, classification: 1 }
			}
		]
	});
	await fireEvent.click(await screen.findByRole('button', { name: 'Load individual results' }));
	await fireEvent.click(await screen.findByText(/historical/));
	expect(screen.getByText(/Optional label availability was not recorded/)).not.toBeNull();
	expect(screen.queryByText('Fully correct receipt')).toBeNull();
});

it('localizes the v4 headings and unavailable text in German', async () => {
	setLocale('de');
	showResults({
		version: '3',
		runSummary: { ...summary, all_case_average_score: null, full_receipt_correct_rate: null }
	});
	await screen.findByRole('cell', { name: /model/ });
	expect(column('Vollständig korrekte Belege').textContent).toBe('Nicht verfügbar');
	expect(column('Bewertung aller Fälle').textContent).toBe('Nicht verfügbar');
});

it('offers compact model outcomes with secondary metrics collapsed', async () => {
	showResults();
	const card = await screen.findByRole('region', { name: 'Model outcomes: model' });
	expect(within(card).getByText('Fully correct receipts')).toBeTruthy();
	expect(within(card).getByText('Fully correct receipts').parentElement?.textContent).toContain(
		'10.00%'
	);
	expect(within(card).getByText('Failed cases: 9')).toBeTruthy();
	const more = within(card).getByText('More metrics').closest('details');
	expect(more?.open).toBe(false);
});

it.each([
	{ language: 'en' as const, label: 'Processing success rate' },
	{ language: 'de' as const, label: 'Erfolgsquote der Verarbeitung' }
])(
	'separates $language processing success from scoring correctness',
	async ({ language, label }) => {
		setLocale(language);
		showResults({
			runSummary: {
				...summary,
				completed: 4,
				succeeded: 3,
				failed: 1,
				full_receipt_correct_rate: 0.25
			}
		});
		const card = await screen.findByRole('region', {
			name: language === 'en' ? 'Model outcomes: model' : 'Modellergebnisse: model'
		});
		const success = within(card).getByText(label);
		expect(success.parentElement?.textContent).toContain('75.00%');
		expect(success.closest('details')).toBeNull();
		const correct = within(card).getByText(
			language === 'en' ? 'Fully correct receipts' : 'Vollständig korrekte Belege'
		);
		expect(correct.parentElement?.textContent).toContain('25.00%');
		expect(correct.closest('details')?.open).toBe(false);
	}
);

it('shows historical processing success independently of unavailable correctness scoring', async () => {
	showResults({
		version: '3',
		runSummary: { ...summary, succeeded: 8, failed: 2, full_receipt_correct_rate: null }
	});
	const card = await screen.findByRole('region', { name: 'Model outcomes: model' });
	expect(within(card).getByText('Processing success rate').parentElement?.textContent).toContain(
		'80.00%'
	);
	expect(within(card).getByText('Fully correct receipts').parentElement?.textContent).toContain(
		'Unavailable'
	);
});

it('shows processing success as unavailable before any terminal cases', async () => {
	showResults({
		runSummary: {
			...summary,
			completed: 0,
			succeeded: 0,
			failed: 0,
			full_receipt_correct_rate: null
		}
	});
	const card = await screen.findByRole('region', { name: 'Model outcomes: model' });
	expect(within(card).getByText('Processing success rate').parentElement?.textContent).toContain(
		'Unavailable'
	);
	expect(card.textContent).not.toContain('NaN');
	expect(card.textContent).not.toContain('Infinity');
});
