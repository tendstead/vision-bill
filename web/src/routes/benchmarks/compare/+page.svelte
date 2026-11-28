<script lang="ts">
	import { goto, replaceState } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { onMount } from 'svelte';
	import { session } from '$lib/auth';
	import {
		buildSeries,
		cellFor,
		defaultSelection,
		deltaTone,
		formatMs,
		formatPercent,
		formatPoints,
		formatSignedMs,
		reasonTitle,
		runName,
		shortId,
		type CompareRunInfo,
		type Comparison
	} from '$lib/benchmark-compare';

	let runs = $state<CompareRunInfo[]>([]),
		selected = $state<string[]>([]),
		comparison = $state<Comparison | null>(null),
		error = $state(''),
		loaded = $state(false);

	// Admin-only page: bounce non-admins home (mirrors the /settings guard).
	$effect(() => {
		if ($session !== 'loading' && $session !== null && !$session.is_admin) {
			void goto(resolve('/'));
		}
	});

	const series = $derived(comparison ? buildSeries(comparison) : []);
	const tone = { good: 'text-green-700 dark:text-green-400', bad: 'text-error', neutral: '' };

	async function loadRuns() {
		const response = await fetch('/api/v1/benchmarks?limit=200');
		if (!response.ok) throw new Error('Unable to load benchmark runs.');
		runs = (await response.json()) as CompareRunInfo[];
	}
	async function loadComparison() {
		if (selected.length < 2) {
			comparison = null;
			return;
		}
		const response = await fetch(`/api/v1/benchmarks/compare?runs=${selected.join(',')}`);
		if (!response.ok) {
			const body = (await response.json().catch(() => null)) as { detail?: unknown } | null;
			throw new Error(
				typeof body?.detail === 'string' ? body.detail : 'Unable to compare these runs.'
			);
		}
		comparison = (await response.json()) as Comparison;
	}
	function syncUrl() {
		const query = selected.length ? `?runs=${selected.join(',')}` : '';
		// resolve() is applied to the route; only the query string is appended.
		// eslint-disable-next-line svelte/no-navigation-without-resolve
		replaceState(`${resolve('/benchmarks/compare')}${query}`, {});
	}
	async function select(ids: string[]) {
		selected = ids;
		syncUrl();
		try {
			await loadComparison();
			error = '';
		} catch (cause) {
			comparison = null;
			error = cause instanceof Error ? cause.message : 'Unable to compare these runs.';
		}
	}
	function toggle(id: string) {
		const next = selected.includes(id) ? selected.filter((s) => s !== id) : [...selected, id];
		// The oldest selected run is the baseline.
		const age = (rid: string) => runs.findIndex((r) => r.id === rid);
		void select(next.sort((a, b) => age(b) - age(a)));
	}

	onMount(() => {
		void (async () => {
			try {
				await loadRuns();
				const fromUrl =
					new URLSearchParams(window.location.search).get('runs') ??
					window.sessionStorage.getItem('benchmark-compare-runs');
				window.sessionStorage.removeItem('benchmark-compare-runs');
				const ids = fromUrl ? fromUrl.split(',').filter(Boolean) : defaultSelection(runs);
				selected = ids;
				syncUrl();
				await loadComparison();
			} catch (cause) {
				error = cause instanceof Error ? cause.message : 'Unable to load the comparison.';
			} finally {
				loaded = true;
			}
		})();
		// Running benchmarks change; keep the numbers live.
		const timer = window.setInterval(() => {
			if (comparison?.runs.some((r) => r.run.status !== 'completed')) {
				void loadComparison().catch(() => undefined);
			}
		}, 3000);
		return () => window.clearInterval(timer);
	});

	const delta = (runId: string, model: string) =>
		comparison?.runs.find((r) => r.run.id === runId)?.deltas.find((d) => d.model_id === model);
	const show = (value: unknown) =>
		value === null || value === undefined
			? '—'
			: typeof value === 'object'
				? JSON.stringify(value)
				: String(value);
</script>

<svelte:head><title>Compare benchmarks · Vision Bill</title></svelte:head>
<section class="mx-auto max-w-6xl">
	<div class="flex flex-wrap items-start justify-between gap-3">
		<div>
			<h1 class="text-2xl font-semibold">Compare benchmark runs</h1>
			<p class="mt-1 text-sm text-on-surface-variant">
				Pick two or more runs. The oldest selected run is the baseline.
			</p>
		</div>
		<a
			class="rounded-lg border border-outline-variant px-4 py-2.5 text-sm"
			href={resolve('/benchmarks/results')}>Back to results</a
		>
	</div>
	{#if error}<p class="mt-4 text-sm text-error" role="alert">{error}</p>{/if}

	<fieldset class="mt-5 rounded-xl border border-outline-variant p-4">
		<legend class="px-1 text-sm font-medium">Runs</legend>
		{#if loaded && runs.length < 2}
			<p class="text-sm text-on-surface-variant">
				At least two benchmark runs are needed to compare.
			</p>
		{/if}
		<div class="max-h-56 space-y-1 overflow-y-auto">
			{#each runs as run (run.id)}
				<label class="flex items-start gap-2 text-sm">
					<input
						type="checkbox"
						checked={selected.includes(run.id)}
						onchange={() => toggle(run.id)}
					/>
					<span
						><span class="font-medium">{runName(run)}</span>
						<span class="text-on-surface-variant"
							>· {run.model_ids.join(', ')} · {run.receipt_ids.length} receipts · {run.status} ·
							{new Date(run.created_at).toLocaleString()}{run.parent_run_id
								? ` · rerun of #${shortId(run.parent_run_id)}`
								: ''}</span
						></span
					>
				</label>
			{/each}
		</div>
	</fieldset>

	{#if comparison}
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
								: ''}:
							{reason.message}. Runs: {reason.run_ids.map((id) => `#${shortId(id)}`).join(', ')}
						</li>
					{/each}
				</ul>
			{/if}
		</div>

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
					>Totals; differences are against the baseline run (score in percentage points)</caption
				>
				<thead class="text-xs text-on-surface-variant"
					><tr
						><th class="p-2">Series</th><th class="p-2">Digest</th><th class="p-2">Score</th><th
							class="p-2">Δ</th
						><th class="p-2">Latency</th><th class="p-2">Δ</th><th class="p-2">Failed</th><th
							class="p-2">Δ</th
						></tr
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
							<td class="p-2 {tone[deltaTone(d?.average_score, true)]}"
								>{formatPoints(d?.average_score)}</td
							>
							<td class="p-2">{formatMs(sum?.average_latency_ms)}</td>
							<td class="p-2 {tone[deltaTone(d?.average_latency_ms, false)]}"
								>{formatSignedMs(d?.average_latency_ms)}</td
							>
							<td class="p-2">{sum?.failed ?? '—'}</td>
							<td class="p-2 {tone[deltaTone(d?.failed, false)]}"
								>{d ? (d.failed > 0 ? `+${d.failed}` : d.failed) : '—'}</td
							>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>

		<div class="mt-6 overflow-x-auto rounded-xl border border-outline-variant">
			<table class="w-full text-left text-sm" data-testid="receipt-table">
				<caption class="bg-surface-container p-2 text-left text-xs text-on-surface-variant"
					>Score per receipt (the same values as the charts)</caption
				>
				<thead class="text-xs text-on-surface-variant"
					><tr
						><th class="p-2">Receipt</th>{#each series as s (s.key)}<th class="p-2">{s.label}</th
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
											title={cell.error ?? ''}>failed</span
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
	{:else if loaded && selected.length < 2 && runs.length >= 2}
		<p class="mt-5 text-sm text-on-surface-variant">Select at least two runs.</p>
	{/if}
</section>
