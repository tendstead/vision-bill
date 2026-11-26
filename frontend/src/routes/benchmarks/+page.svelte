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
	import BenchmarkReceiptPicker from '$lib/ui/BenchmarkReceiptPicker.svelte';
	import { caseApi, type BenchmarkCaseSummary } from '$lib/api/benchmark-cases';
	import {
		defaultExperiment,
		experimentRequest,
		selectedReceiptIds
	} from '$lib/benchmark-experiments';

	const actor = captureSessionActor();
	let alive = true;
	const current = () => alive && isCurrentSessionActor(actor);
	onDestroy(() => {
		alive = false;
	});

	let receiptIds = $state(''),
		category = $state(''),
		maxConfidence = $state<number | undefined>(),
		limit = $state<number | undefined>(),
		timeout = $state(600),
		confirm = $state(false),
		creating = $state(false),
		error = $state('');

	let createdRunId = $state<string | null>(null);
	let references = $state<BenchmarkReferencePreview[]>([]);
	let referenceErrors = $state<{ id: string; message: string }[]>([]);
	let referenceSelection = $state('');
	let qualify = $state(false);
	let previewing = $state(false);
	$effect(() => {
		if (receiptIds !== referenceSelection) {
			references = [];
			referenceErrors = [];
			qualify = false;
		}
	});
	async function previewReferences(onlyId?: string) {
		if (!actor || !current() || previewing) return;
		const selection = receiptIds;
		const ids = onlyId ? [onlyId] : selectedReceiptIds(selection);
		if (!ids.length) return;
		previewing = true;
		qualify = false;
		error = '';
		try {
			const previews = await Promise.allSettled(
				ids.map((id) => api.previewBenchmarkReference(id, actor))
			);
			if (!current() || selection !== receiptIds) return;
			referenceSelection = selection;
			const fresh = references.filter((item) => !ids.includes(item.receipt_identity));
			const problems = referenceErrors.filter((item) => !ids.includes(item.id));
			for (const [index, preview] of previews.entries()) {
				if (preview.status === 'fulfilled') fresh.push(preview.value);
				else
					problems.push({
						id: ids[index],
						message:
							preview.reason instanceof Error
								? preview.reason.message
								: $t('benchmarkUi.previewError')
					});
			}
			references = fresh;
			referenceErrors = problems;
		} finally {
			if (current()) previewing = false;
		}
	}
	let experiment = $state(defaultExperiment());
	let capabilities = $state({ ocr_available: false, ocr_enabled: false, thinking_control: false });
	const modes = ['direct_vision', 'ocr', 'compare'] as const;
	let visionModels = $state<ModelInfo[]>([]);
	let textModels = $state<ModelInfo[]>([]);
	let modelErrors = $state({ vision: false, text: false });
	let loadingModels = $state(true);
	let discoveryGeneration = 0;
	let capabilityError = $state(false);
	let chosenModels = $state<string[]>([]);
	let models = $derived(experiment.pipeline === 'direct_vision' ? visionModels : textModels);
	let visionIds = $derived(new Set(visionModels.map((item) => item.id)));
	let selected = $derived(chosenModels.filter((id) => models.some((model) => model.id === id)));
	let unavailableSelections = $derived(chosenModels.filter((id) => !selected.includes(id)));
	let modelError = $derived(
		(experiment.pipeline !== 'ocr' && modelErrors.vision) ||
			(experiment.pipeline !== 'direct_vision' && modelErrors.text)
	);
	let cases = $state<BenchmarkCaseSummary[]>([]);
	let caseIds = $state<string[]>([]);
	let caseListError = $state(false);
	let loadingCases = $state(true);
	let filterSelection = $state(false);
	let validCases = $derived(
		caseIds.every((id) => cases.some((item) => item.id === id && item.confirmed))
	);
	let needsQualification = $derived(references.filter((item) => !item.currently_qualified));
	let receiptCount = $derived(selectedReceiptIds(receiptIds).length);
	let hasReceipts = $derived(receiptCount > 0);
	$effect(() => {
		if (hasReceipts || caseIds.length > 0) filterSelection = false;
	});
	let pairedModels = $derived(selected.filter((id) => visionIds.has(id)));
	let selectedInputCount = $derived(receiptCount + caseIds.length);
	let jobsPerInput = $derived(
		selected.length + (experiment.pipeline === 'compare' ? pairedModels.length : 0)
	);
	let canCreate = $derived(
		selected.length > 0 &&
			!loadingModels &&
			!modelError &&
			unavailableSelections.length === 0 &&
			(experiment.pipeline === 'direct_vision' || capabilities.ocr_available) &&
			(experiment.pipeline !== 'compare' || pairedModels.length > 0) &&
			validCases &&
			(hasReceipts || caseIds.length > 0 || filterSelection) &&
			(!needsQualification.length || qualify) &&
			referenceErrors.length === 0 &&
			!previewing &&
			!createdRunId &&
			!creating
	);
	// Admin-only page: bounce non-admins home (mirrors the /settings guard).
	$effect(() => {
		if ($session !== 'loading' && $session !== null && !$session.is_admin) {
			void goto(resolve('/'));
		}
	});
	async function initialize() {
		if (!actor || !current()) return;
		const params = new URLSearchParams(window.location.search);
		receiptIds =
			params.get('receipt') ?? window.sessionStorage.getItem('benchmark-reuse-receipts') ?? '';
		caseIds = selectedReceiptIds(
			params.get('case') ?? window.sessionStorage.getItem('benchmark-reuse-cases') ?? ''
		);
		window.sessionStorage.removeItem('benchmark-reuse-receipts');
		window.sessionStorage.removeItem('benchmark-reuse-cases');
		const reuse = params.get('reuse');
		if (reuse) {
			const previous = await api.getBenchmarkStatus(reuse, actor).catch(() => null);
			if (!current()) return;
			if (previous) {
				caseIds = previous.run.case_ids ?? [];
				receiptIds = previous.run.receipt_ids.filter((id) => !caseIds.includes(id)).join(',');
			} else error = $t('benchmarkForm.reuseError');
		}
		void loadCases();
		await loadModels();
	}
	async function loadModels() {
		if (!actor || !current() || creating || createdRunId) return;
		const generation = ++discoveryGeneration;
		loadingModels = true;
		confirm = false;
		const available = await api.getBenchmarkCapabilities(actor).catch(() => null);
		if (!current() || generation !== discoveryGeneration) return;
		capabilities = available ?? {
			ocr_available: false,
			ocr_enabled: false,
			thinking_control: false
		};
		capabilityError = !available;
		const [vision, text] = await Promise.allSettled([
			api.listModels('vision', actor),
			capabilities.ocr_available ? api.listModels('text', actor) : Promise.resolve([])
		]);
		if (!current() || generation !== discoveryGeneration) return;
		visionModels = vision.status === 'fulfilled' ? vision.value : [];
		textModels = text.status === 'fulfilled' ? text.value : [];
		modelErrors = { vision: vision.status === 'rejected', text: text.status === 'rejected' };
		loadingModels = false;
	}
	async function loadCases() {
		if (!actor || !current()) return;
		loadingCases = true;
		caseListError = false;
		try {
			const items = await caseApi.list(actor);
			if (current()) cases = items;
		} catch {
			if (current()) caseListError = true;
		} finally {
			if (current()) loadingCases = false;
		}
	}
	onMount(() => {
		void initialize();
		window.addEventListener('focus', loadModels);
		return () => window.removeEventListener('focus', loadModels);
	});
	function toggle(id: string) {
		chosenModels = chosenModels.includes(id)
			? chosenModels.filter((model) => model !== id)
			: [...chosenModels, id];
	}
	async function create() {
		if (!actor || !current() || !canCreate) return;
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
		if (caseIds.length) body.case_ids = [...caseIds];
		if (experiment.pipeline === 'compare') body.comparison = true;
		if (!ids.length && !caseIds.length && filterSelection) {
			if (category) body.category = category;
			if (maxConfidence !== undefined) body.max_source_confidence = maxConfidence;
			if (limit !== undefined) body.limit = limit;
		}
		try {
			if (qualify) {
				if (!references.length || receiptIds !== referenceSelection)
					throw new Error($t('benchmarkUi.stalePreview'));
				for (const reference of needsQualification) {
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
			const run = await api.createBenchmark(body, actor);
			if (!current()) return;
			createdRunId = run.id;
			await goto(resolve(`/benchmarks/results?run=${encodeURIComponent(run.id)}`));
		} catch (cause) {
			if (current() && !createdRunId) {
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
		{$t('benchmarkForm.intro')}
	</p>
	{#if error}<p class="mt-4 text-sm text-error" role="alert">{error}</p>{/if}
	{#if createdRunId}
		<p class="mt-4 text-sm" role="status">
			{$t('benchmarkForm.created')}
			<a
				class="text-primary underline"
				href={resolve(`/benchmarks/results?run=${encodeURIComponent(createdRunId)}`)}
				>{$t('benchmarkForm.openRun')}</a
			>
		</p>
	{:else if creating}<p class="mt-4 text-sm text-on-surface-variant" role="status">
			{$t('benchmarkUi.starting')}
		</p>{/if}
	<form
		class="mt-6 space-y-5"
		onsubmit={(event) => {
			event.preventDefault();
			if (canCreate) confirm = true;
		}}
	>
		<fieldset class="rounded-xl border border-outline-variant p-4">
			<legend class="px-1 font-medium">{$t('benchmarkUi.pipeline')}</legend>
			<div class="grid gap-3 sm:grid-cols-3">
				{#each modes as mode (mode)}
					<label
						class="flex items-start gap-2 rounded-lg border border-outline-variant p-3 text-sm has-checked:border-primary"
					>
						<input
							class="mt-1"
							type="radio"
							name="mode"
							value={mode}
							bind:group={experiment.pipeline}
							disabled={mode !== 'direct_vision' && !capabilities.ocr_available}
							aria-label={$t('benchmarkForm.' + mode)}
						/>
						<span
							><strong>{$t('benchmarkForm.' + mode)}</strong><small
								class="mt-1 block text-on-surface-variant"
								>{$t('benchmarkForm.' + mode + 'Hint')}</small
							></span
						>
					</label>
				{/each}
			</div>
			{#if capabilityError}<p role="alert" class="mt-2 text-sm text-error">
					{$t('benchmarkForm.capabilitiesError')}
				</p>
			{:else if !capabilities.ocr_available && !loadingModels}<p
					class="mt-2 text-sm text-on-surface-variant"
				>
					{$t('benchmarkForm.noOcr')}
				</p>{/if}
		</fieldset>
		<fieldset class="rounded-xl border border-outline-variant p-4">
			<legend class="px-1 font-medium"
				>{$t(
					experiment.pipeline === 'ocr'
						? 'diagnostics.formatterModels'
						: experiment.pipeline === 'compare'
							? 'benchmarkForm.compareModels'
							: 'diagnostics.visionModels'
				)}</legend
			>
			<button
				type="button"
				class="mb-2 block text-sm text-primary underline"
				disabled={loadingModels || creating}
				onclick={loadModels}>{$t('benchmarkForm.refreshModels')}</button
			>
			{#if loadingModels}<p class="text-sm">{$t('common.loading')}</p>
			{:else if modelError}<p class="text-sm">
					{$t('benchmarkUi.modelsError')}
				</p>
			{:else if !models.length}<p class="text-sm">{$t('benchmarkForm.noModels')}</p>
			{:else}
				{#each models as model, index (model.id)}
					<label class="mr-4 inline-flex items-center gap-2 py-1"
						><input
							type="checkbox"
							aria-label={model.id}
							aria-describedby={experiment.pipeline === 'compare'
								? `model-capability-${index}`
								: undefined}
							checked={selected.includes(model.id)}
							onchange={() => toggle(model.id)}
						/>{model.id}
						{#if experiment.pipeline === 'compare'}<small
								id={`model-capability-${index}`}
								class="text-on-surface-variant"
								>{$t(
									visionIds.has(model.id) ? 'benchmarkForm.bothPaths' : 'benchmarkForm.ocrOnly'
								)}</small
							>{/if}</label
					>
				{/each}
				{#if experiment.pipeline === 'compare' && !pairedModels.length}<p class="mt-2 text-sm">
						{$t('benchmarkForm.visionRequired')}
					</p>{/if}
			{/if}
			{#if unavailableSelections.length && !loadingModels && !modelError}
				<p class="mt-2 text-sm text-on-surface-variant">
					{$t('benchmarkForm.unavailableModels')}
				</p>
				{#each unavailableSelections as id (id)}
					<button
						type="button"
						class="mr-3 text-sm text-primary underline"
						onclick={() => toggle(id)}
						>{$t('benchmarkForm.remove', { values: { name: id } })}</button
					>
				{/each}
			{/if}
		</fieldset>
		<fieldset class="rounded-xl border border-outline-variant p-4">
			<legend class="px-1 font-medium">{$t('benchmarkForm.cases')}</legend>
			<a class="text-sm text-primary underline" href={resolve('/benchmarks/cases')}
				>{$t('benchmarkForm.manageCases')}</a
			>
			{#if loadingCases}<p class="mt-2 text-sm">{$t('common.loading')}</p>
			{:else if caseListError}<p class="mt-2 text-sm">
					{$t('benchmarkForm.caseError')}
					<button class="text-primary underline" type="button" onclick={loadCases}
						>{$t('benchmarkForm.retry')}</button
					>
				</p>
			{:else}
				{#each cases as item (item.id)}
					<label class="mt-2 flex items-center gap-2 text-sm">
						<input
							type="checkbox"
							checked={caseIds.includes(item.id)}
							onchange={() =>
								(caseIds = caseIds.includes(item.id)
									? caseIds.filter((id) => id !== item.id)
									: [...caseIds, item.id])}
							disabled={!item.confirmed && !caseIds.includes(item.id)}
						/>
						{item.name} · {$t('benchmarkForm.photos', { values: { count: item.photo_count } })} · {$t(
							item.confirmed ? 'benchmarkForm.confirmed' : 'benchmarkForm.draft'
						)}
					</label>
				{:else}<p class="mt-2 text-sm text-on-surface-variant">
						{$t('benchmarkForm.noCases')}
					</p>{/each}
			{/if}
			{#if !loadingCases}
				{#each caseIds.filter((id) => !cases.some((item) => item.id === id)) as id (id)}
					<button
						class="mt-2 block text-sm text-primary underline"
						type="button"
						onclick={() => (caseIds = caseIds.filter((selected) => selected !== id))}
						>{$t('benchmarkForm.removeMissingCase', { values: { id } })}</button
					>
				{/each}
			{/if}
		</fieldset>
		<fieldset class="rounded-xl border border-outline-variant p-4">
			<legend class="px-1 font-medium">{$t('benchmarkForm.receipts')}</legend>
			<BenchmarkReceiptPicker
				bind:receiptIds
				onPreview={previewReferences}
				busy={previewing || creating}
			/>

			<p class="mt-1 text-xs text-on-surface-variant">
				{$t('benchmarkForm.referenceHint')}
			</p>
			<button
				type="button"
				class="mt-3 rounded-lg border p-2"
				disabled={previewing || !hasReceipts || creating}
				onclick={() => previewReferences()}>{$t('benchmarkUi.preview')}</button
			>
			{#each referenceErrors as issue (issue.id)}
				<p role="alert" class="mt-2 break-words text-sm text-error">{issue.id} · {issue.message}</p>
			{/each}
			{#if references.length}
				{#each references as reference (reference.receipt_identity)}
					<details class="mt-3 rounded border p-2">
						<summary
							>{reference.receipt_identity} · {$t('benchmarkUi.review')}
							{reference.review_revision} · {$t(
								reference.currently_qualified
									? 'benchmarkForm.qualified'
									: 'benchmarkForm.qualifyRequired'
							)}</summary
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
				{#if needsQualification.length}<label class="mt-3 block"
						><input type="checkbox" bind:checked={qualify} /> {$t('benchmarkUi.consent')}</label
					>{/if}
			{/if}
			<details class="mt-4">
				<summary class="cursor-pointer text-sm font-medium">{$t('benchmarkForm.advanced')}</summary>
				<label class="mt-3 block text-sm"
					>{$t('benchmarkUi.ids')}
					<input
						class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
						bind:value={receiptIds}
						placeholder={$t('benchmarkUi.idPlaceholder')}
					/>
				</label>
				<label class="mt-3 flex items-center gap-2 text-sm"
					><input
						type="checkbox"
						bind:checked={filterSelection}
						disabled={hasReceipts || caseIds.length > 0}
					/>{$t('benchmarkForm.useFilters')}</label
				>
				<fieldset disabled={!filterSelection || hasReceipts || caseIds.length > 0}>
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
			</details>
		</fieldset>
		<BenchmarkExperimentControls
			bind:settings={experiment}
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
			disabled={!canCreate}>{$t('benchmarkUi.create')}</button
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
			<p>{$t('benchmarkForm.' + experiment.pipeline)}</p>
			<p>{$t('benchmarkUi.modelsSummary', { values: { models: selected.join(', ') } })}</p>
			{#if hasReceipts}
				<p>{$t('benchmarkUi.receiptsSummary', { values: { count: receiptCount } })}</p>
			{:else if !caseIds.length}
				<p>{$t('benchmarkUi.filtersSummary')}</p>
			{/if}
			{#if caseIds.length}<p>
					{$t('benchmarkForm.caseSummary', {
						values: {
							names: cases
								.filter((item) => caseIds.includes(item.id))
								.map((item) => item.name)
								.join(', ')
						}
					})}
				</p>{/if}
			{#if experiment.pipeline === 'compare'}
				<p>{$t('benchmarkForm.compareHint')}</p>
				{#if selectedInputCount}<p>
						{$t('benchmarkForm.totalJobs', {
							values: { count: jobsPerInput * selectedInputCount }
						})}
					</p>
				{:else}<p>{$t('benchmarkForm.jobsPerInput', { values: { count: jobsPerInput } })}</p>{/if}
			{/if}
			{#if !hasReceipts && !caseIds.length}
				{#if category}<p>{$t('benchmarkUi.category')}: {category}</p>{/if}
				{#if maxConfidence !== undefined}<p>{$t('benchmarkUi.confidence')}: {maxConfidence}</p>{/if}
				{#if limit !== undefined}<p>{$t('benchmarkUi.limit')}: {limit}</p>{/if}
			{/if}
		</div>
	</ConfirmDialog>
</section>
