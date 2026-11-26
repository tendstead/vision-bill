<script lang="ts">
	import { onDestroy } from 'svelte';
	import { t, locale } from '$lib/i18n';
	import { api } from '$lib/api/client';
	import type { BenchmarkStatus, BenchmarkSummary } from '$lib/types';
	import type { IndividualResult } from '$lib/benchmark-experiments';
	import { isCurrentSessionActor, type SessionActor } from '$lib/session-lifetime';
	import BenchmarkResultDetails from './BenchmarkResultDetails.svelte';

	let {
		direct,
		ocr,
		actor
	}: { direct: BenchmarkStatus; ocr: BenchmarkStatus; actor: SessionActor } = $props();
	const models = $derived([...new Set([...direct.run.model_ids, ...ocr.run.model_ids])]);
	const cases = $derived([...new Set([...direct.run.receipt_ids, ...ocr.run.receipt_ids])]);
	const comparable = $derived(
		!!direct.run.dataset_fingerprint &&
			direct.run.dataset_fingerprint === ocr.run.dataset_fingerprint &&
			direct.run.scoring_version === ocr.run.scoring_version &&
			direct.run.receipt_ids.length === cases.length &&
			ocr.run.receipt_ids.length === cases.length
	);
	const percent = $derived(
		new Intl.NumberFormat($locale ?? 'en', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
	);
	function score(value: number | null | undefined) {
		return value != null && Number.isFinite(value)
			? percent.format(value * 100) + '%'
			: $t('benchmarkResults.unavailable');
	}
	function duration(value: unknown) {
		return typeof value === 'number' && Number.isFinite(value) ? Math.round(value) + ' ms' : '—';
	}
	function delta(left: number | null | undefined, right: number | null | undefined) {
		if (
			!comparable ||
			left == null ||
			right == null ||
			!Number.isFinite(left) ||
			!Number.isFinite(right)
		)
			return '—';
		const points = (right - left) * 100;
		return $t('benchmarkResults.points', {
			values: { value: (points > 0 ? '+' : '') + percent.format(points) }
		});
	}
	function finalScore(arm: BenchmarkStatus, summary?: BenchmarkSummary) {
		return summary &&
			arm.run.model_ids.includes(summary.model_id) &&
			arm.run.receipt_ids.length > 0 &&
			summary.completed === arm.run.receipt_ids.length &&
			arm.run.status !== 'failed'
			? summary.all_case_average_score
			: null;
	}
	let selectedModel = $state('');
	const model = $derived(selectedModel || models[0]);
	let offset = $state(0);
	const completed = $derived(
		[direct, ocr]
			.map((arm) => arm.summaries.find((summary) => summary.model_id === model)?.completed ?? 0)
			.join('/')
	);
	let page = $state<{
		ids: string[];
		completed: string;
		direct: Map<string, IndividualResult>;
		ocr: Map<string, IndividualResult>;
	} | null>(null);
	let inspected = $state('');
	let loading = $state(false);
	let error = $state('');
	let generation = 0;
	onDestroy(() => {
		generation++;
	});
	$effect(() => {
		if (page && page.completed !== completed) void load(offset);
	});
	async function load(start: number, chosenModel = model) {
		const snapshot = completed;
		const request = ++generation;
		const current = () => request === generation && isCurrentSessionActor(actor);
		if (!current()) return;
		error = '';
		loading = true;
		inspected = '';
		offset = start;
		page = null;
		const ids = cases.slice(start, start + 50);
		async function read(arm: BenchmarkStatus) {
			const results = arm.run.model_ids.includes(chosenModel)
				? await api.getBenchmarkResults(arm.run.id, 0, actor, { model: chosenModel, cases: ids })
				: [];
			return new Map(
				results
					.filter((result) => result.model_id === chosenModel)
					.map((result) => [result.receipt_id, result])
			);
		}
		try {
			const [left, right] = await Promise.all([read(direct), read(ocr)]);
			if (current()) {
				page = { ids, direct: left, ocr: right, completed: snapshot };
			}
		} catch (cause) {
			if (current())
				error = cause instanceof Error ? cause.message : $t('benchmarkUi.detailsError');
		} finally {
			if (current()) loading = false;
		}
	}
</script>

<div class="mt-6 overflow-x-auto rounded-xl border border-outline-variant">
	<table class="w-full text-left text-sm">
		<caption class="p-3 text-left font-semibold">{$t('benchmarkResults.comparison')}</caption>
		<thead class="bg-surface-container text-on-surface-variant"
			><tr>
				<th class="p-3" scope="col">{$t('benchmarkResults.model')}</th>
				{#each [direct, ocr] as arm (arm.run.id)}
					<th class="p-3" scope="col"
						>{$t(arm.run.comparison_arm === 'ocr' ? 'benchmarkUi.ocr' : 'benchmarkUi.direct')}
						<span class="block text-xs font-normal"
							>{$t(`benchmarkUi.status.${arm.run.status}`, { default: arm.run.status })}</span
						></th
					>
				{/each}
				<th class="p-3" scope="col">{$t('benchmarkResults.delta')}</th>
			</tr></thead
		>
		<tbody>
			{#each models as id (id)}
				{@const left = direct.summaries.find((s) => s.model_id === id)}
				{@const right = ocr.summaries.find((s) => s.model_id === id)}
				<tr class="border-t border-outline-variant align-top">
					<th scope="row" class="break-words p-3">{id}</th>
					<td class="p-3">{@render outcome(direct, id, left)}</td>
					<td class="p-3">{@render outcome(ocr, id, right)}</td>
					<td class="p-3 font-medium">{delta(finalScore(direct, left), finalScore(ocr, right))}</td>
				</tr>
			{/each}
		</tbody>
	</table>
</div>
<p class="mt-2 text-sm text-on-surface-variant">{$t('benchmarkResults.comparisonDenominator')}</p>
<div class="mt-6 flex flex-wrap items-end gap-3">
	<label class="text-sm"
		>{$t('benchmarkResults.caseModel')}
		<select
			value={model}
			class="block rounded border border-outline-variant bg-surface p-2"
			onchange={(event) => {
				selectedModel = event.currentTarget.value;
				void load(0, selectedModel);
			}}
		>
			{#each models as id (id)}<option value={id}>{id}</option>{/each}
		</select>
	</label>
	<button
		class="rounded-lg border border-outline-variant px-3 py-2 text-sm"
		disabled={loading}
		onclick={() => load(offset)}>{$t('benchmarkUi.loadDetails')}</button
	>
	{#if page}
		<button class="text-sm underline" disabled={offset === 0} onclick={() => load(offset - 50)}
			>{$t('benchmarkUi.previous')}</button
		>
		<button
			class="text-sm underline"
			disabled={offset + 50 >= cases.length}
			onclick={() => load(offset + 50)}>{$t('benchmarkUi.next')}</button
		>
	{/if}
</div>
{#if error}<p class="mt-3 text-sm text-error" role="alert">{error}</p>{/if}
{#if loading}<p role="status" class="mt-3 text-sm">{$t('common.loading')}</p>{/if}
{#if page}
	<div class="mt-4 overflow-x-auto rounded-xl border border-outline-variant">
		<table class="w-full text-left text-sm">
			<caption class="p-3 text-left font-semibold">{$t('benchmarkResults.caseComparison')}</caption>
			<thead class="bg-surface-container"
				><tr>
					<th class="p-3" scope="col">{$t('benchmarkResults.case')}</th>
					<th class="p-3" scope="col">{$t('benchmarkUi.direct')}</th>
					<th class="p-3" scope="col">{$t('benchmarkUi.ocr')}</th>
					<th class="p-3" scope="col">{$t('benchmarkResults.delta')}</th>
				</tr></thead
			>
			<tbody>
				{#each page.ids as id (id)}
					{@const left = page.direct.get(id)}
					{@const right = page.ocr.get(id)}
					<tr class="border-t border-outline-variant align-top">
						<th class="p-3" scope="row"
							><button
								class="text-left underline"
								aria-expanded={inspected === id}
								onclick={() => {
									inspected = inspected === id ? '' : id;
								}}
							>
								{model} · {direct.run.case_names?.[id] ?? ocr.run.case_names?.[id] ?? id}
							</button></th
						>
						<td class="p-3">{@render caseScore(direct, left)}</td>
						<td class="p-3">{@render caseScore(ocr, right)}</td>
						<td class="p-3">{delta(left?.scores?.normalized, right?.scores?.normalized)}</td>
					</tr>
					{#if inspected === id}
						<tr
							><td colspan="4" class="p-3"
								><div class="grid gap-4 lg:grid-cols-2">
									{@render evidence(left, 'benchmarkUi.direct')}
									{@render evidence(right, 'benchmarkUi.ocr')}
								</div></td
							></tr
						>
					{/if}
				{/each}
			</tbody>
		</table>
	</div>
{/if}

{#snippet outcome(arm: BenchmarkStatus, id: string, summary: BenchmarkSummary | undefined)}
	{#if !arm.run.model_ids.includes(id)}
		{$t('benchmarkResults.notApplicable')}
	{:else}
		<strong>{score(summary?.all_case_average_score)}</strong>
		{#if arm.run.status !== 'failed' && (summary?.completed ?? 0) < arm.run.receipt_ids.length}
			<span class="text-xs">{$t('benchmarkResults.provisional')}</span>
		{/if}
		<span class="block text-xs"
			>{$t('benchmarkResults.progress', {
				values: { done: summary?.completed ?? 0, total: arm.run.receipt_ids.length }
			})}</span
		>
		<span class="block text-xs"
			>{$t('benchmarkUi.failedCases', { values: { count: summary?.failed ?? 0 } })}</span
		>
		<span class="block text-xs"
			>{$t('benchmarkResults.averageTime')} · {duration(summary?.average_latency_ms)}</span
		>
	{/if}
{/snippet}

{#snippet caseScore(arm: BenchmarkStatus, detail: IndividualResult | undefined)}
	{#if !arm.run.model_ids.includes(model)}{$t('benchmarkResults.notApplicable')}
	{:else if detail}
		<strong>{score(detail.scores?.normalized)}</strong>
		{#if !detail.success}<span class="block text-error"
				>{$t('benchmarkResults.failed')} · {detail.error === 'ocr_gpu_out_of_memory'
					? $t('settings.dedicatedOcr.reasons.ocr_gpu_out_of_memory')
					: detail.error}</span
			>{/if}
		{#if detail.version === '4' || detail.version === '5'}
			<span class="block text-xs"
				>{$t('benchmarkResults.exactAmounts')} · {detail.exact_amounts == null
					? '—'
					: $t(detail.exact_amounts ? 'benchmarkResults.yes' : 'benchmarkResults.no')}</span
			>
			<span class="block text-xs">{$t('benchmarkResults.itemExtra')} · {detail.item_fp ?? '—'}</span
			>
			<span class="block text-xs"
				>{$t('benchmarkResults.itemMissing')} · {detail.item_fn ?? '—'}</span
			>
		{/if}
		<span class="block text-xs"
			>{$t('benchmarkResults.totalTime')} · {duration(detail.metadata?.elapsed_ms)}</span
		>
		{#if arm.run.comparison_arm === 'ocr'}<span class="block text-xs"
				>{$t('benchmarkResults.ocrTime')} · {duration(
					detail.metadata?.telemetry?.stage_ms?.ocr
				)}</span
			>{/if}
		<span class="block text-xs"
			>{$t('benchmarkResults.llmTime')} · {duration(detail.metadata?.llm_ms)}</span
		>
	{:else}
		{$t(
			['completed', 'failed'].includes(arm.run.status)
				? 'benchmarkResults.unavailable'
				: 'benchmarkResults.pending'
		)}
	{/if}
{/snippet}

{#snippet evidence(detail: IndividualResult | undefined, label: string)}
	{#if detail}<section class="min-w-0">
			<h3>{$t(label)}</h3>
			<BenchmarkResultDetails results={[detail]} names={ocr.run.case_names} />
		</section>{/if}
{/snippet}
