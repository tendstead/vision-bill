<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { onMount, onDestroy, untrack } from 'svelte';
	import { SvelteSet } from 'svelte/reactivity';
	import { session } from '$lib/auth';
	import { api, ApiError } from '$lib/api/client';
	import { captureSessionActor, isCurrentSessionActor } from '$lib/session-lifetime';
	import type { BenchmarkRun, BenchmarkStatus } from '$lib/types';
	import BenchmarkComparison from '$lib/ui/BenchmarkComparison.svelte';
	import BenchmarkResultDetails from '$lib/ui/BenchmarkResultDetails.svelte';
	import { t } from '$lib/i18n';
	import type { IndividualResult } from '$lib/benchmark-experiments';

	const actor = captureSessionActor();
	const currentView = () => alive && isCurrentSessionActor(actor);
	let runs = $state<BenchmarkRun[]>([]),
		selectedId = $state<string | null>(null),
		result = $state<BenchmarkStatus | null>(null),
		sibling = $state<BenchmarkStatus | null>(null),
		error = $state('');
	let requestedId: string | null = null;
	let loadingIndex = $state(true);
	let historyOffset = $state(0);
	let hasOlder = $state(false);
	let loadingOlder = $state(false);
	let historyRequest = 0;
	async function loadOlder() {
		if (!actor || !currentView() || loadingOlder) return;
		const request = ++historyRequest;
		const current = () => currentView() && historyRequest === request;
		loadingOlder = true;
		try {
			const page = await api.listBenchmarks(actor, { offset: historyOffset });
			if (!current()) return;
			const known = new Set(runs.map((run) => run.id));
			runs = [...runs, ...page.filter((run) => !known.has(run.id))];
			historyOffset += page.length;
			hasOlder = page.length === 50;
			error = '';
		} catch (cause) {
			if (current())
				error = cause instanceof Error ? cause.message : $t('benchmarkUi.resultsError');
		} finally {
			if (current()) loadingOlder = false;
		}
	}
	const choices = $derived.by(() => {
		const seen = new SvelteSet<string>();
		return runs
			.filter((run) => {
				if (!run.comparison_id) return true;
				if (seen.has(run.comparison_id)) return false;
				seen.add(run.comparison_id);
				return true;
			})
			.map((run) =>
				run.comparison_id && run.comparison_id === result?.run.comparison_id ? result.run : run
			);
	});
	function pipeline(run: BenchmarkRun) {
		const options = run.experiment?.options as { pipeline?: string } | undefined;
		const identity = run.experiment?.identity as
			{ ocr_settings?: { enabled?: boolean } } | undefined;
		switch (run.comparison_arm ?? options?.pipeline) {
			case 'direct_vision':
				return $t('benchmarkUi.direct');
			case 'ocr':
				return $t('benchmarkUi.ocr');
			case 'upload': {
				const enabled = identity?.ocr_settings?.enabled;
				const resolved =
					enabled == null ? '' : ' · ' + $t(enabled ? 'benchmarkUi.ocr' : 'benchmarkUi.direct');
				return $t('benchmarkUi.upload') + resolved;
			}
			default:
				return $t('benchmarkResults.unavailable');
		}
	}
	function resetDetails() {
		++detailsRequest;
		detailsError = '';
		detailsCompleted = null;
		loadingDetails = false;
		details = [];
		detailsOffset = 0;
	}
	function select(id: string) {
		requestedId = id;
		selectedId = id;
		result = sibling = null;
		error = '';
		resetDetails();
		void goto(resolve(`/benchmarks/results?run=${encodeURIComponent(id)}`), {
			replaceState: true,
			noScroll: true,
			keepFocus: true
		});
		void refresh(false, true);
	}
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
	let detailsError = $state('');
	let detailsCompleted = $state<number | null>(null);
	$effect(() => {
		if (
			result &&
			!sibling &&
			!loadingDetails &&
			detailsCompleted !== null &&
			detailsCompleted !== result.terminal
		) {
			untrack(() => void loadDetails(detailsOffset));
		}
	});
	let detailsOffset = $state(0),
		loadingDetails = $state(false);
	async function loadDetails(offset = 0) {
		const runId = selectedId,
			request = ++detailsRequest;
		const current = () => currentView() && selectedId === runId && detailsRequest === request;
		if (!actor || !runId || !current() || !result) return;
		const terminal = result.terminal;
		loadingDetails = true;
		try {
			const rows = await api.getBenchmarkResults(runId, offset, actor);
			if (!current()) return;
			detailsError = '';
			details = rows;
			detailsCompleted = terminal;
			detailsOffset = offset;
		} catch (cause) {
			if (current()) {
				details = [];
				detailsCompleted = null;
				detailsOffset = offset;
				detailsError = cause instanceof Error ? cause.message : $t('benchmarkUi.detailsError');
			}
		} finally {
			if (current()) loadingDetails = false;
		}
	}
	const fmt = (value: number | null, digits = 2) =>
		value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits);
	async function refresh(index = true, selectionChanged = false) {
		if (refreshing && !selectionChanged) return;
		const request = ++refreshRequest;
		const current = () => currentView() && refreshRequest === request;
		if (!actor || !current()) return;
		refreshing = true;
		try {
			if (index) {
				++historyRequest;
				loadingOlder = false;
				const available = await api.listBenchmarks(actor);
				if (!current()) return;
				runs = available;
				historyOffset = available.length;
				hasOlder = available.length === 50;
				const id =
					requestedId ??
					(runs.some((r) => r.id === selectedId) ? selectedId : (runs[0]?.id ?? null));
				if (id !== selectedId) {
					selectedId = id;
					result = sibling = null;
					resetDetails();
				}
			}
			if (!selectedId) {
				error = '';
				return;
			}
			const statusFor = (id: string, cached: BenchmarkStatus | null) =>
				!index && cached?.run.id === id && ['completed', 'failed'].includes(cached.run.status)
					? cached
					: api.getBenchmarkStatus(id, actor);
			const next = await statusFor(selectedId, result);
			if (!current()) return;
			let partner: BenchmarkStatus | null = null;
			const group = next.run.comparison_id;
			if (group) {
				const opposite = next.run.comparison_arm === 'ocr' ? 'direct_vision' : 'ocr';
				const isPartner = (r: BenchmarkRun) =>
					r.comparison_id === group && r.comparison_arm === opposite && r.id !== next.run.id;
				let other = runs.find(isPartner);
				if (!other) {
					const groupRuns = await api.listBenchmarks(actor, { comparison_id: group });
					if (!current()) return;
					other = groupRuns.find(isPartner);
				}
				if (!other) throw new Error($t('benchmarkResults.missingArm'));
				partner = await statusFor(other.id, sibling);
				if (!current()) return;
				if (!isPartner(partner.run)) throw new Error($t('benchmarkResults.missingArm'));
			}
			if (!runs.some((r) => r.id === next.run.id)) runs = [next.run, ...runs];
			runs = runs.map((r) => (r.id === next.run.id ? next.run : r));
			if (partner) {
				const otherRun = partner.run;
				if (!runs.some((r) => r.id === otherRun.id)) runs = [...runs, otherRun];
			}
			result = next;
			sibling = partner;
			error = '';
		} catch (cause) {
			if (current()) {
				const transient =
					cause instanceof ApiError &&
					(cause.status >= 500 || [0, 408, 429].includes(cause.status));
				if (!transient || result?.run.id !== selectedId) {
					result = sibling = null;
					resetDetails();
				}
				error = cause instanceof Error ? cause.message : $t('benchmarkUi.resultsError');
			}
		} finally {
			if (refreshRequest === request) {
				refreshing = false;
				loadingIndex = false;
			}
		}
	}
	onMount(() => {
		requestedId = new URL(window.location.href).searchParams.get('run');
		void refresh();
		const timer = window.setInterval(() => {
			if (
				result &&
				[result, sibling].some((arm) => arm && !['completed', 'failed'].includes(arm.run.status))
			)
				void refresh(false);
		}, 3000);
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
	<button class="mt-4 text-sm underline" onclick={() => refresh()}
		>{$t('benchmarkResults.refresh')}</button
	>
	{#if loadingIndex}<p class="mt-4 text-sm" role="status">{$t('common.loading')}</p>
	{:else if runs.length === 0 && !error}<div
			class="mt-6 rounded-xl border border-dashed border-outline-variant p-8 text-center text-sm text-on-surface-variant"
		>
			{$t('benchmarkUi.noRuns')}
		</div>{:else if runs.length > 0}
		<label class="mt-6 block max-w-md text-sm font-medium"
			>{$t('benchmarkUi.run')}<select
				class="mt-1 block w-full rounded-lg border border-outline-variant bg-surface p-2"
				value={selectedId ?? ''}
				onchange={(event) => select(event.currentTarget.value)}
				>{#each choices as run (run.id)}<option value={run.id}
						>{run.comparison_id ? $t('benchmarkResults.comparison') : pipeline(run)} · #{run.id} — {$t(
							`benchmarkUi.status.${run.status}`,
							{ default: run.status }
						)}</option
					>{/each}</select
			></label
		>
		{#if hasOlder}<button class="mt-2 text-sm underline" disabled={loadingOlder} onclick={loadOlder}
				>{$t('benchmarkResults.older')}</button
			>{/if}
		{#if result}
			<div class="mt-4 rounded-xl bg-surface-container p-4 text-sm">
				{#if !result.run.comparison_id}<p class="mb-1 font-semibold">
						{$t('benchmarkUi.pipeline')} · {pipeline(result.run)}
					</p>{/if}
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
							result.run.receipt_ids.filter((id) => !result?.run.case_ids?.includes(id)).join(',')
						);
						window.sessionStorage.setItem(
							'benchmark-reuse-cases',
							(result.run.case_ids ?? []).join(',')
						);
						void goto(resolve('/benchmarks'));
					}}>{$t('benchmarkUi.reuse')}</a
				>
				{#if result.run.experiment}<details class="mt-2">
						<summary class="cursor-pointer">{$t('benchmarkUi.settings')}</summary>
						{#each sibling ? [result, sibling] : [result] as arm (arm.run.id)}
							{#if sibling}<p class="mt-2">{pipeline(arm.run)} · #{arm.run.id}</p>{/if}
							<pre class="mt-2 overflow-x-auto whitespace-pre-wrap text-xs">{JSON.stringify(
									arm.run.experiment,
									null,
									2
								)}</pre>
						{/each}
					</details>{/if}
			</div>
			{#if sibling && actor}
				{#key result.run.comparison_id}<BenchmarkComparison
						{actor}
						direct={result.run.comparison_arm === 'direct_vision' ? result : sibling}
						ocr={result.run.comparison_arm === 'ocr' ? result : sibling}
					/>{/key}
			{:else}
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
								<summary class="cursor-pointer font-medium">{$t('benchmarkUi.moreMetrics')}</summary
								>
								<dl class="mt-3 space-y-3">
									<div>
										<dt>
											{$t(
												result.run.scoring_version === '5'
													? 'benchmarkResults.confirmedCorrectReceipts'
													: 'benchmarkResults.fullyCorrectReceipts'
											)}
										</dt>
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
								<th scope="col" class="p-3"
									>{$t(
										result.run.scoring_version === '5'
											? 'benchmarkResults.confirmedCorrectReceipts'
											: 'benchmarkResults.fullyCorrectReceipts'
									)}</th
								>
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
										<span class="text-xs text-on-surface-variant"
											>{summary.model_digest ?? '—'}</span
										>
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
										{summary.average_confidence == null
											? '—'
											: `${fmt(summary.average_confidence)}%`}
									</td>
									<td class="p-3">
										{summary.average_latency_ms == null
											? '—'
											: `${fmt(summary.average_latency_ms, 0)} ms`}
									</td>
									<td class="p-3">{summary.succeeded} / {summary.failed}</td>
									<td class="p-3">
										{$t('benchmarkResults.findings', {
											values: { count: summary.council_findings }
										})}
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
					{#if detailsError}<p class="mt-3 text-sm text-error" role="alert">{detailsError}</p>{/if}
					<BenchmarkResultDetails results={details} names={result.run.case_names} />
				{/if}
			{/if}
		{/if}
	{/if}
</section>

{#snippet rate(
	value: number | null,
	count: number,
	denominator: 'completedCases' | 'successfulCases'
)}
	{#if value == null || !Number.isFinite(value)}
		{$t('benchmarkResults.unavailable')}
	{:else}
		{(value * 100).toFixed(2)}%
		<span class="block text-xs text-on-surface-variant">
			{$t(`benchmarkResults.${denominator}`, { values: { count } })}
		</span>
	{/if}
{/snippet}
