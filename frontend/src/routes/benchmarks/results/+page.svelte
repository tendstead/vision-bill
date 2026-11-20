<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { onMount, onDestroy } from 'svelte';
	import { session } from '$lib/auth';
	import { api } from '$lib/api/client';
	import { captureSessionActor, isCurrentSessionActor } from '$lib/session-lifetime';
	import type { BenchmarkRun, BenchmarkStatus } from '$lib/types';
	import BenchmarkResultDetails from '$lib/ui/BenchmarkResultDetails.svelte';
	import { t } from '$lib/i18n';
	import type { IndividualResult } from '$lib/benchmark-experiments';

	const actor = captureSessionActor();
	const currentView = () => alive && isCurrentSessionActor(actor);
	let runs = $state<BenchmarkRun[]>([]),
		selectedId = $state<string | null>(null),
		result = $state<BenchmarkStatus | null>(null),
		error = $state('');
	// Admin-only page: bounce non-admins home (mirrors the /settings guard).
	$effect(() => {
		if ($session !== 'loading' && $session !== null && !$session.is_admin) {
			void goto(resolve('/'));
		}
	});
	let alive = true;
	let refreshing = false;
	let refreshRequest = 0,
		detailsRequest = 0;
	onDestroy(() => {
		alive = false;
	});
	let details = $state<IndividualResult[]>([]);
	let detailsOffset = $state(0),
		loadingDetails = $state(false);
	async function loadDetails(offset = 0) {
		const runId = selectedId,
			request = ++detailsRequest;
		const current = () => currentView() && selectedId === runId && detailsRequest === request;
		if (!actor || !runId || !current()) return;
		loadingDetails = true;
		try {
			const rows = await api.getBenchmarkResults(runId, offset, actor);
			if (!current()) return;
			details = rows;
			detailsOffset = offset;
		} catch (cause) {
			if (current())
				error = cause instanceof Error ? cause.message : $t('benchmarkUi.detailsError');
		} finally {
			if (current()) loadingDetails = false;
		}
	}
	const fmt = (value: number | null, digits = 2) => (value === null ? '—' : value.toFixed(digits));
	async function refresh(selectionChanged = false) {
		if (refreshing && !selectionChanged) return;
		const request = ++refreshRequest;
		const current = () => currentView() && refreshRequest === request;
		if (!actor || !current()) return;
		refreshing = true;
		try {
			const available = await api.listBenchmarks(actor);
			if (!current()) return;
			runs = available;
			if (!selectedId || !runs.some((run) => run.id === selectedId)) {
				selectedId = runs[0]?.id ?? null;
				detailsRequest++;
				result = null;
				details = [];
				detailsOffset = 0;
				loadingDetails = false;
			}
			if (selectedId) {
				const next = await api.getBenchmarkStatus(selectedId, actor);
				if (!current()) return;
				result = next;
			}
			if (current()) error = '';
		} catch (cause) {
			if (current())
				error = cause instanceof Error ? cause.message : $t('benchmarkUi.resultsError');
		} finally {
			if (refreshRequest === request) refreshing = false;
		}
	}
	onMount(() => {
		void refresh();
		const timer = window.setInterval(() => void refresh(), 3000);
		return () => window.clearInterval(timer);
	});
</script>

<svelte:head><title>{$t('benchmarkUi.results')} · Vision Bill</title></svelte:head>
<section class="mx-auto max-w-6xl">
	<div class="flex flex-wrap items-start justify-between gap-3">
		<div>
			<h1 class="text-2xl font-semibold">{$t('benchmarkUi.results')}</h1>
			<p class="mt-1 text-sm text-on-surface-variant">{$t('benchmarkUi.resultsIntro')}</p>
		</div>
		<a
			class="rounded-lg bg-error-action px-4 py-2.5 text-sm font-medium text-on-error"
			href={resolve('/benchmarks')}>{$t('benchmarkUi.create')}</a
		>
	</div>
	{#if error}<p class="mt-4 text-sm text-error" role="alert">{error}</p>{/if}
	{#if runs.length === 0}<div
			class="mt-6 rounded-xl border border-dashed border-outline-variant p-8 text-center text-sm text-on-surface-variant"
		>
			{$t('benchmarkUi.noRuns')}
		</div>{:else}
		<label class="mt-6 block max-w-md text-sm font-medium"
			>{$t('benchmarkUi.run')}<select
				class="mt-1 block w-full rounded-lg border border-outline-variant bg-surface p-2"
				value={selectedId ?? ''}
				onchange={(event) => {
					selectedId = event.currentTarget.value;
					++detailsRequest;
					loadingDetails = false;
					result = null;
					error = '';
					details = [];
					detailsOffset = 0;
					void refresh(true);
				}}
				>{#each runs as run (run.id)}<option value={run.id}
						>#{run.id} — {$t(`benchmarkUi.status.${run.status}`, { default: run.status })}</option
					>{/each}</select
			></label
		>
		{#if result}
			<div class="mt-4 rounded-xl bg-surface-container p-4 text-sm">
				<p>
					{$t('benchmarkUi.scoringVersion')}
					{result.run.scoring_version}. {result.run.experiment
						? $t('benchmarkUi.separateScoring')
						: $t('benchmarkUi.historical')}
				</p>
				<a
					class="mt-2 inline-block underline"
					href={resolve('/benchmarks')}
					onclick={(event) => {
						event.preventDefault();
						if (!result || !currentView()) return;
						window.sessionStorage.setItem(
							'benchmark-reuse-receipts',
							result.run.receipt_ids.join(',')
						);
						void goto(resolve('/benchmarks'));
					}}>{$t('benchmarkUi.reuse')}</a
				>
				{#if result.run.experiment}<details class="mt-2">
						<summary class="cursor-pointer">{$t('benchmarkUi.settings')}</summary>
						<pre class="mt-2 overflow-x-auto whitespace-pre-wrap text-xs">{JSON.stringify(
								result.run.experiment,
								null,
								2
							)}</pre>
					</details>{/if}
			</div>
			<div class="mt-5 grid grid-cols-2 gap-2 sm:grid-cols-5">
				<div class="rounded-xl bg-surface-container p-4">
					<p class="text-xs text-on-surface-variant">{$t('benchmarkUi.completed')}</p>
					<p class="text-xl font-semibold">
						{result.terminal} / {result.run.receipt_ids.length * result.run.model_ids.length}
					</p>
				</div>
				<div class="rounded-xl bg-surface-container p-4">
					<p class="text-xs text-on-surface-variant">{$t('benchmarkUi.queued')}</p>
					<p class="text-xl font-semibold">{result.queued}</p>
				</div>
				<div class="rounded-xl bg-surface-container p-4">
					<p class="text-xs text-on-surface-variant">{$t('benchmarkUi.running')}</p>
					<p class="text-xl font-semibold">{result.running}</p>
				</div>
				<div class="rounded-xl bg-surface-container p-4">
					<p class="text-xs text-on-surface-variant">{$t('benchmarkUi.retrying')}</p>
					<p class="text-xl font-semibold">{result.retrying}</p>
				</div>
				<div class="rounded-xl bg-surface-container p-4">
					<p class="text-xs text-on-surface-variant">{$t('benchmarkUi.waiting')}</p>
					<p class="text-xl font-semibold">{result.waiting_for_model}</p>
				</div>
			</div>
			<div class="mt-6 space-y-3 lg:hidden">
				{#each result.summaries as summary (summary.model_id)}
					<section
						class="min-w-0 rounded-xl border border-outline-variant p-4 text-sm"
						aria-label={$t('benchmarkUi.modelOutcomes', { values: { model: summary.model_id } })}
					>
						<h2 class="break-words font-semibold">{summary.model_id}</h2>
						<dl class="mt-3 grid grid-cols-2 gap-3">
							<div>
								<dt class="text-xs text-on-surface-variant">
									{$t('benchmarkUi.processingSuccessRate')}
								</dt>
								<dd class="mt-1 font-medium">
									{@render rate(
										summary.completed > 0 ? summary.succeeded / summary.completed : null,
										summary.completed,
										'completedCases'
									)}
								</dd>
							</div>
							<div>
								<dt class="text-xs text-on-surface-variant">{$t('benchmarkResults.latency')}</dt>
								<dd class="mt-1 font-medium">
									{summary.average_latency_ms == null
										? '—'
										: `${fmt(summary.average_latency_ms, 0)} ms`}
								</dd>
							</div>
						</dl>
						<p class="mt-3">
							{$t('benchmarkUi.failedCases', { values: { count: summary.failed } })}
						</p>
						<details class="mt-3 border-t border-outline-variant pt-3">
							<summary class="cursor-pointer font-medium">{$t('benchmarkUi.moreMetrics')}</summary>
							<dl class="mt-3 space-y-3">
								<div>
									<dt>{$t('benchmarkResults.fullyCorrectReceipts')}</dt>
									<dd>
										{@render rate(
											summary.full_receipt_correct_rate,
											summary.completed,
											'completedCases'
										)}
									</dd>
								</div>
								<div>
									<dt>{$t('benchmarkResults.allCaseScore')}</dt>
									<dd>
										{@render rate(
											summary.all_case_average_score,
											summary.completed,
											'completedCases'
										)}
									</dd>
								</div>
								<div>
									<dt>{$t('benchmarkResults.successOnlyScore')}</dt>
									<dd>
										{@render rate(summary.average_score, summary.succeeded, 'successfulCases')}
									</dd>
								</div>
								<div>
									<dt>{$t('benchmarkResults.confidence')}</dt>
									<dd>
										{summary.average_confidence == null
											? '—'
											: `${fmt(summary.average_confidence)}%`}
									</dd>
								</div>
								<div>
									<dt>{$t('benchmarkResults.successFailed')}</dt>
									<dd>{summary.succeeded} / {summary.failed}</dd>
								</div>
								<div>
									<dt>{$t('benchmarkResults.council')}</dt>
									<dd>
										{$t('benchmarkResults.findings', {
											values: { count: summary.council_findings }
										})}
									</dd>
								</div>
								<div>
									<dt>{$t('benchmarkUi.modelDigest')}</dt>
									<dd class="break-all">{summary.model_digest ?? '—'}</dd>
								</div>
							</dl>
						</details>
					</section>
				{/each}
			</div>
			<div class="hidden lg:block mt-6 overflow-x-auto rounded-xl border border-outline-variant">
				<table class="w-full text-left text-sm">
					<thead class="bg-surface-container text-xs text-on-surface-variant">
						<tr>
							<th scope="col" class="p-3">{$t('benchmarkResults.model')}</th>
							<th scope="col" class="p-3">{$t('benchmarkResults.fullyCorrectReceipts')}</th>
							<th scope="col" class="p-3">{$t('benchmarkResults.allCaseScore')}</th>
							<th scope="col" class="p-3">{$t('benchmarkResults.successOnlyScore')}</th>
							<th scope="col" class="p-3">{$t('benchmarkResults.confidence')}</th>
							<th scope="col" class="p-3">{$t('benchmarkResults.latency')}</th>
							<th scope="col" class="p-3">{$t('benchmarkResults.successFailed')}</th>
							<th scope="col" class="p-3">{$t('benchmarkResults.council')}</th>
						</tr>
					</thead>
					<tbody>
						{#each result.summaries as summary (summary.model_id)}
							<tr class="border-t border-outline-variant">
								<td class="p-3 font-medium">
									{summary.model_id}<br />
									<span class="text-xs text-on-surface-variant">{summary.model_digest ?? '—'}</span>
								</td>
								<td class="p-3">
									{@render rate(
										summary.full_receipt_correct_rate,
										summary.completed,
										'completedCases'
									)}
								</td>
								<td class="p-3">
									{@render rate(
										summary.all_case_average_score,
										summary.completed,
										'completedCases'
									)}
								</td>
								<td class="p-3">
									{@render rate(summary.average_score, summary.succeeded, 'successfulCases')}
								</td>
								<td class="p-3">
									{summary.average_confidence == null ? '—' : `${fmt(summary.average_confidence)}%`}
								</td>
								<td class="p-3">
									{summary.average_latency_ms == null
										? '—'
										: `${fmt(summary.average_latency_ms, 0)} ms`}
								</td>
								<td class="p-3">{summary.succeeded} / {summary.failed}</td>
								<td class="p-3">
									{$t('benchmarkResults.findings', { values: { count: summary.council_findings } })}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			{#if result.run.experiment}
				<div class="mt-5 flex flex-wrap items-center gap-3">
					<button
						class="rounded-lg border border-outline-variant px-3 py-2 text-sm"
						disabled={loadingDetails}
						onclick={() => loadDetails(detailsOffset)}>{$t('benchmarkUi.loadDetails')}</button
					>
					<button
						class="text-sm underline"
						disabled={loadingDetails || detailsOffset === 0}
						onclick={() => loadDetails(Math.max(0, detailsOffset - 50))}
						>{$t('benchmarkUi.previous')}</button
					>
					<button
						class="text-sm underline"
						disabled={loadingDetails || details.length < 50}
						onclick={() => loadDetails(detailsOffset + 50)}>{$t('benchmarkUi.next')}</button
					>
				</div>
				<BenchmarkResultDetails results={details} />
			{/if}
		{/if}
	{/if}
</section>

{#snippet rate(
	value: number | null,
	count: number,
	denominator: 'completedCases' | 'successfulCases'
)}
	{#if value == null}
		{$t('benchmarkResults.unavailable')}
	{:else}
		{(value * 100).toFixed(2)}%
		<span class="block text-xs text-on-surface-variant">
			{$t(`benchmarkResults.${denominator}`, { values: { count } })}
		</span>
	{/if}
{/snippet}
