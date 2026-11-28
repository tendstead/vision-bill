<script lang="ts">
	import { goto, replaceState } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { onMount } from 'svelte';
	import { session } from '$lib/auth';
	import BenchmarkResultDetails from '$lib/ui/BenchmarkResultDetails.svelte';
	import BenchmarkScoreDots from '$lib/ui/BenchmarkScoreDots.svelte';
	import BenchmarkSummaryBars from '$lib/ui/BenchmarkSummaryBars.svelte';
	import BenchmarkDumbbell from '$lib/ui/BenchmarkDumbbell.svelte';
	import BenchmarkLatencyStrip from '$lib/ui/BenchmarkLatencyStrip.svelte';
	import type { IndividualResult } from '$lib/benchmark-experiments';
	import {
		buildSeries,
		candidateHint,
		candidateTooltip,
		cellFor,
		deltaTone,
		failureHelp,
		failureLabel,
		failureReason,
		formatMs,
		formatPercent,
		formatPoints,
		formatSignedMs,
		orderForCompare,
		reasonTitle,
		runName,
		shortId,
		truncateMiddle,
		type Candidate,
		type CompareRunInfo,
		type Comparison,
		type Issue
	} from '$lib/benchmark-compare';

	type Run = CompareRunInfo & {
		scoring_version: string;
		experiment?: Record<string, unknown> | null;
	};
	type Status = {
		run: Run;
		queued: number;
		running: number;
		retrying: number;
		waiting_for_model: number;
		terminal: number;
		issues: Issue[];
	};

	let runs = $state<Run[]>([]),
		main = $state<string | null>(null),
		others = $state<string[]>([]),
		candidates = $state<Record<string, Candidate>>({}),
		status = $state<Status | null>(null),
		comparison = $state<Comparison | null>(null),
		error = $state(''),
		loaded = $state(false);

	// Admin-only page: bounce non-admins home (mirrors the /settings guard).
	$effect(() => {
		if ($session !== 'loading' && $session !== null && !$session.is_admin) {
			void goto(resolve('/'));
		}
	});

	const tone = { good: 'text-green-700 dark:text-green-400', bad: 'text-error', neutral: '' };
	const series = $derived(comparison ? buildSeries(comparison) : []);
	const selection = $derived(main ? orderForCompare(runs, [...others, main]) : []);
	const failures = $derived(
		comparison
			? comparison.receipts.flatMap((receipt) =>
					receipt.cells
						.filter((c) => !c.success)
						.map((cell) => ({
							receiptId: receipt.receipt_id,
							cell,
							series: series.find((s) => s.runId === cell.run_id && s.modelId === cell.model_id)
						}))
				)
			: []
	);
	const fmt = (value: number | null | undefined, digits = 1) =>
		value == null ? '—' : value.toFixed(digits);

	async function getJson<T>(url: string, what: string): Promise<T> {
		const response = await fetch(url);
		if (!response.ok) {
			const body = (await response.json().catch(() => null)) as { detail?: unknown } | null;
			throw new Error(typeof body?.detail === 'string' ? body.detail : `Unable to load ${what}.`);
		}
		return (await response.json()) as T;
	}
	function syncUrl() {
		const query = main ? `?run=${main}${others.length ? `&with=${others.join(',')}` : ''}` : '';
		// resolve() is applied to the route; only the query string is appended.
		// eslint-disable-next-line svelte/no-navigation-without-resolve
		replaceState(`${resolve('/benchmarks/results')}${query}`, {});
	}
	async function loadCandidates() {
		if (!main) return;
		const list = await getJson<Candidate[]>(
			`/api/v1/benchmarks/${main}/candidates`,
			'comparable runs'
		);
		candidates = Object.fromEntries(list.map((c) => [c.run_id, c]));
		// A run that stopped being selectable (for example it was reset) leaves the selection.
		others = others.filter((id) => candidates[id]?.selectable);
	}
	async function loadMain() {
		if (!main) return;
		status = await getJson<Status>(`/api/v1/benchmarks/${main}`, 'the benchmark run');
		comparison = await getJson<Comparison>(
			`/api/v1/benchmarks/compare?runs=${selection.join(',')}`,
			'the comparison'
		);
		// New results arrived: keep the in-depth view current.
		if (status.terminal !== lastTerminal) {
			lastTerminal = status.terminal;
			if (detailsRun && loadedDetails) void loadDetails(detailsOffset);
		}
	}
	async function refreshRuns() {
		runs = await getJson<Run[]>('/api/v1/benchmarks?limit=200', 'benchmark runs');
	}
	async function reload(withCandidates = false) {
		try {
			if (withCandidates) await loadCandidates();
			await loadMain();
			error = '';
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Unable to load the results.';
		}
	}
	async function chooseMain(id: string) {
		main = id;
		others = [];
		details = [];
		detailsOffset = 0;
		detailsRun = id;
		syncUrl();
		await reload(true);
		void loadDetails(0);
	}
	async function toggle(id: string) {
		others = others.includes(id) ? others.filter((o) => o !== id) : [...others, id];
		if (detailsRun && !selection.includes(detailsRun)) detailsRun = main;
		syncUrl();
		await reload();
	}

	// In-depth results of one run.
	let lastTerminal = -1,
		loadedDetails = false;
	let details = $state<IndividualResult[]>([]),
		detailsOffset = $state(0),
		detailsRun = $state<string | null>(null),
		loadingDetails = $state(false);
	async function loadDetails(offset = 0) {
		if (!detailsRun) return;
		loadingDetails = true;
		try {
			details = await getJson<IndividualResult[]>(
				`/api/v1/benchmarks/${detailsRun}/results?limit=50&offset=${offset}`,
				'individual results'
			);
			detailsOffset = offset;
			loadedDetails = true;
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Unable to load individual results.';
		} finally {
			loadingDetails = false;
		}
	}

	// Rerun the main run with the same frozen inputs.
	type Change = { receipt_id: string; change: string; detail: string };
	let rerunLabel = $state(''),
		rerunning = $state(false),
		changes = $state<Change[]>([]),
		rerunNotice = $state('');
	async function rerun() {
		if (!status) return;
		rerunning = true;
		rerunNotice = '';
		changes = [];
		try {
			const response = await fetch(`/api/v1/benchmarks/${status.run.id}/rerun`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(rerunLabel.trim() ? { label: rerunLabel.trim() } : {})
			});
			const body = await response.json().catch(() => null);
			if (!response.ok)
				throw new Error(typeof body?.detail === 'string' ? body.detail : 'Unable to rerun.');
			changes = (body.changes ?? []) as Change[];
			rerunNotice = `Queued rerun #${body.id}.`;
			rerunLabel = '';
			await refreshRuns();
			await chooseMain(body.id as string);
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Unable to rerun.';
		} finally {
			rerunning = false;
		}
	}

	onMount(() => {
		void (async () => {
			try {
				await refreshRuns();
				const q = new URLSearchParams(window.location.search);
				const wanted = q.get('run');
				const start = runs.find((r) => r.id === wanted)?.id ?? runs[0]?.id ?? null;
				if (start) {
					main = start;
					others = (q.get('with') ?? '').split(',').filter(Boolean);
					detailsRun = start;
					await loadCandidates().catch(() => undefined);
					await loadMain();
					syncUrl();
					void loadDetails(0);
				}
			} catch (cause) {
				error = cause instanceof Error ? cause.message : 'Unable to load benchmark runs.';
			} finally {
				loaded = true;
			}
		})();
		const timer = window.setInterval(() => {
			void (async () => {
				try {
					const before = runs.map((r) => `${r.id}:${r.status}`).join();
					await refreshRuns();
					if (!main) {
						// The first run was created while this page was open.
						if (runs[0]) await chooseMain(runs[0].id);
						return;
					}
					const changed = before !== runs.map((r) => `${r.id}:${r.status}`).join();
					await reload(changed);
				} catch {
					/* keep the last values; the next tick retries */
				}
			})();
		}, 3000);
		return () => window.clearInterval(timer);
	});

	const show = (value: unknown) =>
		value === null || value === undefined
			? '—'
			: typeof value === 'object'
				? JSON.stringify(value)
				: String(value);
	const delta = (runId: string, model: string) =>
		comparison?.runs.find((r) => r.run.id === runId)?.deltas.find((d) => d.model_id === model);
	const runOption = (run: Run) =>
		`${runName(run)} · ${run.model_ids.join(', ')} · ${new Date(run.created_at).toLocaleString()} · ${run.status}`;
</script>

<svelte:head><title>Benchmarks · Vision Bill</title></svelte:head>
<section class="mx-auto max-w-6xl">
	<div class="flex flex-wrap items-start justify-between gap-3">
		<div>
			<h1 class="text-2xl font-semibold">Benchmarks</h1>
			<p class="mt-1 text-sm text-on-surface-variant">
				Local-model evaluation results; select further runs to compare.
			</p>
		</div>
		<a
			class="rounded-lg bg-error px-4 py-2.5 text-sm font-medium text-on-error"
			href={resolve('/benchmarks')}>Create benchmark</a
		>
	</div>
	{#if error}<p class="mt-4 text-sm text-error" role="alert">{error}</p>{/if}
	{#if loaded && runs.length === 0}<div
			class="mt-6 rounded-xl border border-dashed border-outline-variant p-8 text-center text-sm text-on-surface-variant"
		>
			No benchmark runs yet.
		</div>{/if}

	{#if main}
		<div class="mt-6 rounded-xl border border-outline-variant p-4" data-testid="run-picker">
			<label class="block max-w-2xl text-sm font-medium"
				>Run
				<select
					class="mt-1 block w-full rounded-lg border border-outline-variant bg-surface p-2"
					value={main}
					onchange={(event) => void chooseMain(event.currentTarget.value)}
					>{#each runs as run (run.id)}<option value={run.id}>{runOption(run)}</option
						>{/each}</select
				></label
			>
			{#if runs.length > 1}
				<fieldset class="mt-4">
					<legend class="text-sm font-medium">Compare with</legend>
					<p class="text-xs text-on-surface-variant">
						Only runs that can be compared with the selected run are available; hover a greyed-out
						run to see why not. The oldest selected run is the baseline.
					</p>
					<div class="mt-2 max-h-56 space-y-1 overflow-y-auto">
						{#each runs.filter((r) => r.id !== main) as run (run.id)}
							{@const c = candidates[run.id]}
							{@const blocked = !c || !c.selectable}
							<label
								class="flex items-start gap-2 text-sm {blocked ? 'opacity-60' : ''}"
								title={c && c.reasons.length ? candidateTooltip(c) : ''}
								data-blocked={blocked}
							>
								<input
									type="checkbox"
									disabled={blocked}
									checked={others.includes(run.id)}
									onchange={() => void toggle(run.id)}
								/>
								<span
									><span class="font-medium">{runName(run)}</span>
									<span class="text-on-surface-variant"
										>· {run.model_ids.join(', ')} · {run.receipt_ids.length} receipts ·
										{new Date(run.created_at).toLocaleString()}{run.parent_run_id
											? ` · rerun of #${shortId(run.parent_run_id)}`
											: ''}</span
									>
									{#if c && c.reasons.length}<span
											class="ml-1 rounded bg-surface-container px-1.5 py-0.5 text-xs {blocked
												? 'text-error'
												: 'text-on-surface-variant'}"
											>{blocked ? 'not comparable: ' : ''}{candidateHint(c)}</span
										>{/if}</span
								>
							</label>
						{/each}
					</div>
				</fieldset>
			{/if}
		</div>
	{/if}

	{#if status}
		<div class="mt-4 rounded-xl bg-surface-container p-4 text-sm">
			<p>
				Scoring version: {status.run.scoring_version}. {status.run.experiment
					? 'Extraction and classification are evaluated separately.'
					: 'Historical scoring; no stored individual outputs.'}
			</p>
			<div class="mt-3 flex flex-wrap items-end gap-3">
				<label class="text-xs"
					>Label for the rerun (optional)<input
						class="mt-1 block w-64 rounded-lg border border-outline-variant bg-surface p-2 text-sm"
						bind:value={rerunLabel}
						maxlength="200"
						placeholder="e.g. after model update, ngl 20"
					/></label
				>
				<button
					class="rounded-lg border border-outline-variant px-3 py-2 text-sm"
					disabled={rerunning || !status.run.experiment}
					title={status.run.experiment
						? 'Same receipts and expected values, new run'
						: 'Runs without stored inputs cannot be rerun'}
					onclick={rerun}>Rerun with the same inputs</button
				>
				<a
					class="text-sm underline"
					href={resolve('/benchmarks')}
					onclick={(event) => {
						event.preventDefault();
						if (!status) return;
						window.sessionStorage.setItem(
							'benchmark-reuse-receipts',
							status.run.receipt_ids.join(',')
						);
						void goto(resolve('/benchmarks'));
					}}>Reuse this receipt selection</a
				>
			</div>
			{#if rerunNotice}<p class="mt-2 text-xs" role="status">{rerunNotice}</p>{/if}
			{#if changes.length}<div class="mt-2 text-xs" data-testid="rerun-changes">
					<p class="font-medium">
						Receipts changed since the original run (the frozen values are still used):
					</p>
					<ul class="list-disc pl-5">
						{#each changes as c, i (`${c.receipt_id}:${c.change}:${i}`)}<li>
								<a class="font-mono underline" href={resolve(`/receipts/${c.receipt_id}`)}
									>{c.receipt_id.slice(0, 8)}</a
								>
								— {c.change.replaceAll('_', ' ')}: {c.detail}
							</li>{/each}
					</ul>
				</div>{/if}
			{#if status.run.experiment}<details class="mt-2">
					<summary class="cursor-pointer">Run settings</summary>
					<pre class="mt-2 overflow-x-auto whitespace-pre-wrap text-xs">{JSON.stringify(
							status.run.experiment,
							null,
							2
						)}</pre>
				</details>{/if}
		</div>
		<div class="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
			<div class="rounded-xl bg-surface-container p-4">
				<p class="text-xs text-on-surface-variant">Completed</p>
				<p class="text-xl font-semibold">
					{status.terminal} / {status.run.receipt_ids.length * status.run.model_ids.length}
				</p>
			</div>
			<div class="rounded-xl bg-surface-container p-4">
				<p class="text-xs text-on-surface-variant">Queued</p>
				<p class="text-xl font-semibold">{status.queued}</p>
			</div>
			<div class="rounded-xl bg-surface-container p-4">
				<p class="text-xs text-on-surface-variant">Running</p>
				<p class="text-xl font-semibold">{status.running}</p>
			</div>
			<div class="rounded-xl bg-surface-container p-4">
				<p class="text-xs text-on-surface-variant">Retrying</p>
				<p class="text-xl font-semibold">{status.retrying}</p>
			</div>
			<div class="rounded-xl bg-surface-container p-4">
				<p class="text-xs text-on-surface-variant">Waiting</p>
				<p class="text-xl font-semibold">{status.waiting_for_model}</p>
			</div>
		</div>
		{#if status.issues.length}
			<div
				class="mt-4 rounded-xl border border-amber-600 bg-amber-50 p-4 text-sm text-amber-950 dark:bg-amber-950 dark:text-amber-100"
				data-testid="run-issues"
			>
				<p class="font-medium">Not finished yet — and why</p>
				<ul class="mt-1 list-disc space-y-1 pl-5">
					{#each status.issues as issue (`${issue.model_id}:${issue.receipt_id}`)}
						<li>
							<strong>{issue.model_id}</strong> ·
							<a class="font-mono underline" href={resolve(`/receipts/${issue.receipt_id}`)}
								>{shortId(issue.receipt_id)}</a
							>
							· {issue.status.replaceAll('_', ' ')}, {issue.attempts} attempt(s){issue.retry_at
								? `, next try ${new Date(issue.retry_at).toLocaleTimeString()}`
								: ''}: {issue.reason}
						</li>
					{/each}
				</ul>
			</div>
		{/if}
	{/if}

	{#if comparison}
		{#if comparison.runs.length > 1}
			<div
				class="mt-5 rounded-xl border p-4 text-sm {comparison.comparable.comparable
					? 'border-green-700 bg-green-50 text-green-950 dark:bg-green-950 dark:text-green-100'
					: 'border-amber-600 bg-amber-50 text-amber-950 dark:bg-amber-950 dark:text-amber-100'}"
				data-testid="comparability"
			>
				{#if comparison.comparable.comparable}
					<p class="font-medium">Directly comparable</p>
					<p class="mt-1">
						Same receipts and expected values, model digests, prompt and schema, scoring version and
						council policy.
					</p>
				{:else}
					<p class="font-medium">Not directly comparable</p>
					<ul class="mt-1 list-disc space-y-1 pl-5">
						{#each comparison.comparable.reasons as reason (`${reason.code}:${reason.model_id ?? ''}`)}
							<li>
								<strong>{reasonTitle(reason.code)}</strong>{reason.model_id
									? ` (${reason.model_id})`
									: ''}: {reason.message}. Runs: {reason.run_ids
									.map((id) => `#${shortId(id)}`)
									.join(', ')}
							</li>
						{/each}
					</ul>
				{/if}
			</div>
		{/if}

		{#if comparison.varied.length}
			<div class="mt-4 overflow-x-auto rounded-xl border border-outline-variant">
				<table class="w-full text-left text-sm" data-testid="varied-settings">
					<caption class="bg-surface-container p-2 text-left text-xs text-on-surface-variant"
						>Settings that differ between the runs</caption
					>
					<thead class="text-xs text-on-surface-variant"
						><tr
							><th class="p-2">Setting</th>{#each comparison.runs as r (r.run.id)}<th class="p-2"
									>{runName(r.run)}</th
								>{/each}</tr
						></thead
					>
					<tbody>
						{#each comparison.varied as v (v.field)}
							<tr class="border-t border-outline-variant"
								><td class="p-2 font-medium">{v.field}</td
								>{#each comparison.runs as r (r.run.id)}<td class="p-2 font-mono text-xs"
										>{show(v.values[r.run.id])}</td
									>{/each}</tr
							>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}

		<div class="mt-5 overflow-x-auto rounded-xl border border-outline-variant">
			<table class="w-full text-left text-sm" data-testid="summary-table">
				<caption class="bg-surface-container p-2 text-left text-xs text-on-surface-variant"
					>Totals{comparison.runs.length > 1
						? '; differences are against the baseline run (score in percentage points)'
						: ''}</caption
				>
				<thead class="text-xs text-on-surface-variant"
					><tr
						><th class="p-2">Model / run</th><th class="p-2">Digest</th><th class="p-2"
							>Correctness</th
						>{#if comparison.runs.length > 1}<th class="p-2">Δ</th>{/if}<th class="p-2"
							>Confidence</th
						><th class="p-2">Latency</th>{#if comparison.runs.length > 1}<th class="p-2">Δ</th
							>{/if}<th class="p-2">Success / failed</th>{#if comparison.runs.length > 1}<th
								class="p-2">Δ</th
							>{/if}<th class="p-2">Council</th></tr
					></thead
				>
				<tbody>
					{#each series as s (s.key)}
						{@const run = comparison.runs.find((r) => r.run.id === s.runId)}
						{@const sum = run?.summaries.find((m) => m.model_id === s.modelId)}
						{@const d = delta(s.runId, s.modelId)}
						<tr class="border-t border-outline-variant">
							<td class="p-2 font-medium"
								><span
									class="mr-2 inline-block h-2.5 w-2.5 rounded-full"
									style="background:{s.color}"
								></span>{s.label}</td
							>
							<td class="p-2 font-mono text-xs"
								>{run?.model_digests[s.modelId]?.slice(0, 12) ?? '—'}</td
							>
							<td class="p-2">{formatPercent(sum?.average_score)}</td>
							{#if comparison.runs.length > 1}<td
									class="p-2 {tone[deltaTone(d?.average_score, true)]}"
									>{formatPoints(d?.average_score)}</td
								>{/if}
							<td class="p-2"
								>{sum?.average_confidence == null ? '—' : `${fmt(sum.average_confidence)}%`}</td
							>
							<td class="p-2">{formatMs(sum?.average_latency_ms)}</td>
							{#if comparison.runs.length > 1}<td
									class="p-2 {tone[deltaTone(d?.average_latency_ms, false)]}"
									>{formatSignedMs(d?.average_latency_ms)}</td
								>{/if}
							<td class="p-2">{sum ? `${sum.succeeded} / ${sum.failed}` : '—'}</td>
							{#if comparison.runs.length > 1}<td class="p-2 {tone[deltaTone(d?.failed, false)]}"
									>{d ? (d.failed > 0 ? `+${d.failed}` : d.failed) : '—'}</td
								>{/if}
							<td class="p-2">{sum?.council_findings ?? 0} findings</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>

		{#if comparison.receipts.some((r) => r.cells.length)}
			<div class="mt-5 space-y-5">
				<BenchmarkSummaryBars {comparison} {series} />
				<BenchmarkScoreDots {comparison} {series} />
				{#if comparison.runs.length > 1}<BenchmarkDumbbell {comparison} {series} />{/if}
				<BenchmarkLatencyStrip {comparison} {series} />
			</div>
		{:else}
			<p class="mt-5 text-sm text-on-surface-variant">No stored results yet to chart.</p>
		{/if}

		{#if failures.length}
			<div class="mt-6 overflow-x-auto rounded-xl border border-outline-variant">
				<table class="w-full text-left text-sm" data-testid="failure-table">
					<caption class="bg-surface-container p-2 text-left text-xs text-on-surface-variant"
						>Failed tasks — why the model did not deliver a correct answer</caption
					>
					<thead class="text-xs text-on-surface-variant"
						><tr
							><th class="p-2">Receipt</th><th class="p-2">Model / run</th><th class="p-2">Kind</th
							><th class="p-2">Reason</th><th class="p-2">Time</th><th class="p-2">Attempts</th></tr
						></thead
					>
					<tbody>
						{#each failures as f (`${f.cell.run_id}:${f.cell.model_id}:${f.receiptId}`)}
							<tr class="border-t border-outline-variant align-top">
								<td class="p-2"
									><a class="font-mono underline" href={resolve(`/receipts/${f.receiptId}`)}
										>{shortId(f.receiptId)}</a
									></td
								>
								<td class="p-2">{f.series?.label ?? f.cell.model_id}</td>
								<td class="p-2" title={failureHelp(f.cell.error_kind)}
									><span
										class="whitespace-nowrap rounded bg-surface-container px-1.5 py-0.5 text-xs text-error"
										>{failureLabel(f.cell.error_kind)}</span
									></td
								>
								<td class="p-2 text-xs">{failureReason(f.cell)}</td>
								<td class="whitespace-nowrap p-2">{formatMs(f.cell.latency_ms)}</td>
								<td class="p-2">{f.cell.attempts}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}

		<div class="mt-6 overflow-x-auto rounded-xl border border-outline-variant">
			<table class="w-full text-left text-sm" data-testid="receipt-table">
				<caption class="bg-surface-container p-2 text-left text-xs text-on-surface-variant"
					>Score per receipt (the same values as the charts)</caption
				>
				<thead class="text-xs text-on-surface-variant"
					><tr
						><th class="p-2">Receipt</th>{#each series as s (s.key)}<th
								class="max-w-40 break-words p-2 align-bottom"
								title={s.label}>{truncateMiddle(s.label, 36)}</th
							>{/each}</tr
					></thead
				>
				<tbody>
					{#each comparison.receipts as receipt (receipt.receipt_id)}
						<tr class="border-t border-outline-variant">
							<td class="p-2"
								><a class="font-mono underline" href={resolve(`/receipts/${receipt.receipt_id}`)}
									>{shortId(receipt.receipt_id)}</a
								></td
							>
							{#each series as s (s.key)}
								{@const cell = cellFor(receipt, s.runId, s.modelId)}
								<td class="p-2">
									{#if !cell}—{:else if !cell.success}<span
											class="text-error"
											title={failureReason(cell)}
											>{failureLabel(cell.error_kind).toLowerCase()}</span
										>{:else}{formatPercent(
											cell.score
										)}{#if cell.score_delta != null && Math.abs(cell.score_delta) > 1e-9}
											<span class="ml-1 text-xs {tone[deltaTone(cell.score_delta, true)]}"
												>{formatPoints(cell.score_delta)}</span
											>{/if}{/if}
								</td>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>

		{#if comparison.runs.some((r) => r.run.model_ids.length)}
			<div class="mt-8" data-testid="run-details">
				<h2 class="text-lg font-semibold">Run details</h2>
				<div class="mt-2 flex flex-wrap items-center gap-3">
					{#if selection.length > 1}
						<label class="text-sm"
							>Run
							<select
								class="ml-1 rounded-lg border border-outline-variant bg-surface p-1.5"
								value={detailsRun}
								onchange={(event) => {
									detailsRun = event.currentTarget.value;
									details = [];
									void loadDetails(0);
								}}
								>{#each comparison.runs as r (r.run.id)}<option value={r.run.id}
										>{runName(r.run)}</option
									>{/each}</select
							></label
						>
					{/if}
					<button
						class="rounded-lg border border-outline-variant px-3 py-2 text-sm"
						disabled={loadingDetails}
						onclick={() => loadDetails(detailsOffset)}>Load individual results</button
					>
					<button
						class="text-sm underline"
						disabled={loadingDetails || detailsOffset === 0}
						onclick={() => loadDetails(Math.max(0, detailsOffset - 50))}>Previous 50</button
					>
					<button
						class="text-sm underline"
						disabled={loadingDetails || details.length < 50}
						onclick={() => loadDetails(detailsOffset + 50)}>Next 50</button
					>
				</div>
				<BenchmarkResultDetails results={details} />
			</div>
		{/if}
	{/if}
</section>
