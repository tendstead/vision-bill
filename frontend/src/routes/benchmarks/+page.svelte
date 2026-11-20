<script lang="ts">
	import { t } from '$lib/i18n';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { onDestroy, onMount } from 'svelte';
	import { api, ApiError } from '$lib/api/client';
	import { session } from '$lib/auth';
	import { captureSessionActor, isCurrentSessionActor } from '$lib/session-lifetime';
	import type { BenchmarkCreate, BenchmarkReferencePreview, ModelInfo } from '$lib/types';
	import ConfirmDialog from '$lib/ui/ConfirmDialog.svelte';
	import BenchmarkExperimentControls from '$lib/ui/BenchmarkExperimentControls.svelte';
	import { defaultExperiment, experimentRequest } from '$lib/benchmark-experiments';

	const actor = captureSessionActor();
	let alive = true;
	const current = () => alive && isCurrentSessionActor(actor);
	onDestroy(() => {
		alive = false;
	});

	let references = $state<BenchmarkReferencePreview[]>([]);
	let referenceSelection = $state('');
	let qualify = $state(false);
	let previewing = $state(false);
	$effect(() => {
		if (receiptIds !== referenceSelection) {
			references = [];
			qualify = false;
		}
	});
	function selectedReceiptIds(value: string) {
		const ids = value
			.split(',')
			.map((id) => id.trim().toLowerCase())
			.filter(Boolean);
		return [...new Set(ids)];
	}
	async function previewReferences() {
		if (!actor || !current() || previewing) return;
		previewing = true;
		qualify = false;
		references = [];
		error = '';
		const selection = receiptIds;
		try {
			const ids = selectedReceiptIds(selection);
			if (!ids.length) throw new Error($t('benchmarkUi.explicitRequired'));
			const previews = await Promise.all(ids.map((id) => api.previewBenchmarkReference(id, actor)));
			if (current() && receiptIds === selection) {
				referenceSelection = selection;
				references = previews;
			}
		} catch (cause) {
			if (current() && receiptIds === selection)
				error = cause instanceof Error ? cause.message : $t('benchmarkUi.previewError');
		} finally {
			if (current()) previewing = false;
		}
	}
	let experiment = $state(defaultExperiment());
	let capabilities = $state({ ocr_available: false, ocr_enabled: false, thinking_control: false });
	let loaded = $state(false);
	let capability: 'text' | 'vision' = $derived(
		experiment.pipeline === 'ocr' || (experiment.pipeline === 'upload' && capabilities.ocr_enabled)
			? 'text'
			: 'vision'
	);
	$effect(() => {
		if (!actor || !loaded || !current()) return;
		let active = true;
		models = [];
		selected = [];
		void api
			.listModels(capability, actor)
			.then((available) => {
				if (active && current()) {
					models = available;
					selected = available.map((model) => model.id);
				}
			})
			.catch(() => {
				if (active && current()) error = $t('benchmarkUi.modelsError');
			});
		return () => {
			active = false;
		};
	});
	let models = $state<ModelInfo[]>([]),
		selected = $state<string[]>([]),
		receiptIds = $state(''),
		category = $state(''),
		maxConfidence = $state<number | undefined>(),
		limit = $state<number | undefined>(),
		timeout = $state(600),
		confirm = $state(false),
		creating = $state(false),
		error = $state('');
	// Admin-only page: bounce non-admins home (mirrors the /settings guard).
	$effect(() => {
		if ($session !== 'loading' && $session !== null && !$session.is_admin) {
			void goto(resolve('/'));
		}
	});
	async function initialize() {
		if (!actor || !current()) return;
		receiptIds = window.sessionStorage.getItem('benchmark-reuse-receipts') ?? '';
		window.sessionStorage.removeItem('benchmark-reuse-receipts');
		const reuse = new URLSearchParams(window.location.search).get('reuse');
		if (reuse) {
			const previous = await api.getBenchmarkStatus(reuse, actor).catch(() => null);
			if (!current()) return;
			if (previous) receiptIds = previous.run.receipt_ids.join(',');
		}
		const available = await api.getBenchmarkCapabilities(actor).catch(() => null);
		if (!current()) return;
		if (available) capabilities = available;
		loaded = true;
	}
	onMount(() => {
		void initialize();
	});
	function toggle(id: string) {
		selected = selected.includes(id) ? selected.filter((model) => model !== id) : [...selected, id];
	}
	async function create() {
		if (!actor || !current() || creating) return;
		creating = true;
		confirm = false;
		error = '';
		const ids = selectedReceiptIds(receiptIds);
		const body: BenchmarkCreate = {
			model_ids: selected,
			experiment: experimentRequest(experiment),
			request_timeout_seconds: timeout,
			council_policy: 'all',
			apply_council_flags: false
		};
		if (ids.length) body.receipt_ids = ids;
		if (category) body.category = category;
		if (maxConfidence !== undefined) body.max_source_confidence = maxConfidence;
		if (limit !== undefined) body.limit = limit;
		try {
			if (qualify) {
				if (!references.length || receiptIds !== referenceSelection)
					throw new Error($t('benchmarkUi.stalePreview'));
				for (const reference of references) {
					await api.qualifyBenchmarkReference(
						reference.receipt_identity,
						{
							expected_review_revision: reference.review_revision,
							expected_document_sha256: reference.document_sha256
						},
						actor
					);
					if (!current()) return;
				}
			}
			await api.createBenchmark(body, actor);
			if (current()) await goto(resolve('/benchmarks/results'));
		} catch (cause) {
			if (current()) {
				if (
					cause instanceof ApiError &&
					cause.status === 503 &&
					cause.detail === 'ocr_gpu_out_of_memory'
				) {
					error = $t('settings.dedicatedOcr.reasons.ocr_gpu_out_of_memory');
				} else {
					error = cause instanceof Error ? cause.message : $t('benchmarkUi.createError');
				}
			}
		} finally {
			if (current()) {
				creating = false;
				confirm = false;
			}
		}
	}
</script>

<svelte:head><title>{$t('benchmarkUi.create')} · Vision Bill</title></svelte:head>
<section class="mx-auto max-w-3xl">
	<h1 class="text-2xl font-semibold">{$t('benchmarkUi.create')}</h1>
	<p class="mt-1 text-sm text-on-surface-variant">
		{$t('benchmarkUi.intro')}
	</p>
	{#if error}<p class="mt-4 text-sm text-error" role="alert">{error}</p>{/if}
	{#if creating}<p class="mt-4 text-sm text-on-surface-variant" role="status">
			{$t('benchmarkUi.starting')}
		</p>{/if}
	<form
		class="mt-6 space-y-5"
		onsubmit={(event) => {
			event.preventDefault();
			if (!creating) confirm = true;
		}}
	>
		<fieldset class="rounded-xl border border-outline-variant p-4">
			<legend class="px-1 font-medium"
				>{$t(
					capability === 'text' ? 'diagnostics.formatterModels' : 'diagnostics.visionModels'
				)}</legend
			>{#each models as model (model.id)}<label class="mr-4 inline-flex items-center gap-2 py-1"
					><input
						type="checkbox"
						checked={selected.includes(model.id)}
						onchange={() => toggle(model.id)}
					/>{model.id}</label
				>{/each}
		</fieldset>
		<fieldset class="rounded-xl border border-outline-variant p-4">
			<legend class="px-1 font-medium">{$t('benchmarkUi.selection')}</legend><label
				class="block text-sm"
				>{$t('benchmarkUi.ids')}
				<input
					class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
					bind:value={receiptIds}
					placeholder={$t('benchmarkUi.idPlaceholder')}
				/></label
			>
			<p class="mt-1 text-xs text-on-surface-variant">
				{$t('benchmarkUi.referenceHint')}
			</p>
			<button
				type="button"
				class="mt-3 rounded-lg border p-2"
				disabled={previewing}
				onclick={previewReferences}>{$t('benchmarkUi.preview')}</button
			>
			{#if references.length}
				{#each references as reference (reference.receipt_identity)}
					<details class="mt-3 rounded border p-2">
						<summary
							>{reference.receipt_identity} · {$t('benchmarkUi.review')}
							{reference.review_revision}</summary
						>
						<a href={resolve('/receipts/[id]', { id: reference.receipt_identity })}
							>{$t('benchmarkUi.inspect')}</a
						>
						<p>{$t('benchmarkUi.scored')} {reference.scored_fields.join(', ')}</p>
						<pre class="overflow-auto text-xs">{JSON.stringify(
								{
									expected: reference.expected,
									sources: reference.sources,
									findings: reference.findings
								},
								null,
								2
							)}</pre>
					</details>
				{/each}
				<label class="mt-3 block"
					><input type="checkbox" bind:checked={qualify} /> {$t('benchmarkUi.consent')}</label
				>
			{/if}
			<div class="mt-3 grid gap-3 sm:grid-cols-3">
				<label class="text-sm"
					>{$t('benchmarkUi.category')}<input
						class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
						bind:value={category}
					/></label
				><label class="text-sm"
					>{$t('benchmarkUi.confidence')}<input
						class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
						type="number"
						min="0"
						max="100"
						bind:value={maxConfidence}
					/></label
				><label class="text-sm"
					>{$t('benchmarkUi.limit')}<input
						class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
						type="number"
						min="1"
						bind:value={limit}
					/></label
				>
			</div>
		</fieldset>
		<BenchmarkExperimentControls
			bind:settings={experiment}
			ocrAvailable={capabilities.ocr_available}
			thinkingControl={capabilities.thinking_control}
		/>
		<fieldset class="rounded-xl border border-outline-variant p-4">
			<legend class="px-1 font-medium">{$t('benchmarkUi.execution')}</legend>
			<label class="text-sm"
				>{$t('benchmarkUi.timeout')}
				<input
					class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
					type="number"
					min="1"
					max="600"
					bind:value={timeout}
				/>
			</label>
		</fieldset>
		<p class="rounded-lg bg-error-container p-3 text-sm text-on-error-container">
			<strong>{$t('benchmarkUi.before')}</strong>
			{$t('benchmarkUi.warning')}
		</p>
		<button
			class="rounded-lg bg-error-action px-4 py-2.5 text-sm font-medium text-on-error disabled:opacity-50"
			disabled={selected.length === 0 || creating}>{$t('benchmarkUi.create')}</button
		>
	</form>
	<ConfirmDialog
		open={confirm}
		title={$t('benchmarkUi.confirm')}
		body={$t('benchmarkUi.warning')}
		confirmLabel={$t('benchmarkUi.start')}
		busy={creating}
		onConfirm={create}
		onCancel={() => (confirm = false)}
	>
		<div class="mt-3 space-y-2 break-words text-sm">
			<p>{$t('benchmarkUi.modelsSummary', { values: { models: selected.join(', ') } })}</p>
			<p>
				{selectedReceiptIds(receiptIds).length
					? $t('benchmarkUi.receiptsSummary', {
							values: { count: selectedReceiptIds(receiptIds).length }
						})
					: $t('benchmarkUi.filtersSummary')}
			</p>
			{#if !selectedReceiptIds(receiptIds).length}
				{#if category}<p>{$t('benchmarkUi.category')}: {category}</p>{/if}
				{#if maxConfidence !== undefined}<p>{$t('benchmarkUi.confidence')}: {maxConfidence}</p>{/if}
				{#if limit !== undefined}<p>{$t('benchmarkUi.limit')}: {limit}</p>{/if}
			{/if}
		</div>
	</ConfirmDialog>
</section>
