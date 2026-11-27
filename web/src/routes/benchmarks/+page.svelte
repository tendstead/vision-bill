<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { onMount } from 'svelte';
	import { session } from '$lib/auth';
	import BenchmarkExperimentControls from '$lib/ui/BenchmarkExperimentControls.svelte';
	import { defaultExperiment, experimentRequest } from '$lib/benchmark-experiments';

	let experiment = $state(defaultExperiment());
	let capabilities = $state({ ocr_available: false, thinking_control: false });
	type Model = { id: string };
	let models = $state<Model[]>([]),
		selected = $state<string[]>([]),
		receiptIds = $state(''),
		category = $state(''),
		maxConfidence = $state<number | undefined>(),
		limit = $state<number | undefined>(),
		timeout = $state(300),
		policy = $state('all'),
		absolute = $state<number | undefined>(),
		relative = $state<number | undefined>(),
		applyFlags = $state(false),
		confirm = $state(false),
		creating = $state(false),
		error = $state('');
	// Admin-only page: bounce non-admins home (mirrors the /settings guard).
	$effect(() => {
		if ($session !== 'loading' && $session !== null && !$session.is_admin) {
			void goto(resolve('/'));
		}
	});
	onMount(async () => {
		receiptIds = window.sessionStorage.getItem('benchmark-reuse-receipts') ?? '';
		window.sessionStorage.removeItem('benchmark-reuse-receipts');
		const reuse = new URLSearchParams(window.location.search).get('reuse');
		if (reuse) {
			const previous = await fetch(`/api/v1/benchmarks/${encodeURIComponent(reuse)}`);
			if (previous.ok) {
				const data = await previous.json();
				receiptIds = data.run.receipt_ids.join(',');
			}
		}
		const available = await fetch('/api/v1/benchmarks/capabilities').catch(() => null);
		if (available?.ok) capabilities = await available.json();
		try {
			const response = await fetch('/api/v1/llm/models');
			if (!response.ok) throw new Error();
			models = (await response.json()) as Model[];
			selected = models.map((model) => model.id);
		} catch {
			error = 'Unable to load local vision models.';
		}
	});
	function toggle(id: string) {
		selected = selected.includes(id) ? selected.filter((model) => model !== id) : [...selected, id];
	}
	async function create() {
		creating = true;
		error = '';
		const ids = receiptIds
			.split(',')
			.map((id) => id.trim())
			.filter(Boolean);
		const body: Record<string, unknown> = {
			model_ids: selected,
			experiment: experimentRequest(experiment),
			request_timeout_seconds: timeout,
			council_policy: policy,
			apply_council_flags: applyFlags
		};
		if (ids.length) body.receipt_ids = ids;
		if (category) body.category = category;
		if (maxConfidence !== undefined) body.max_source_confidence = maxConfidence;
		if (limit !== undefined) body.limit = limit;
		if (absolute !== undefined) body.council_absolute_threshold = absolute;
		if (relative !== undefined) body.council_relative_threshold = relative;
		try {
			const response = await fetch('/api/v1/benchmarks', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(body)
			});
			if (!response.ok) throw new Error(await response.text());
			await goto(resolve('/benchmarks/results'));
		} catch (cause) {
			error = cause instanceof Error ? cause.message : 'Unable to create benchmark.';
		} finally {
			creating = false;
			confirm = false;
		}
	}
</script>

<svelte:head><title>Create benchmark · Vision Bill</title></svelte:head>
<section class="mx-auto max-w-3xl">
	<h1 class="text-2xl font-semibold">Create benchmark</h1>
	<p class="mt-1 text-sm text-on-surface-variant">
		Compare local vision models using verified receipts. Normal receipt data is not changed.
	</p>
	{#if error}<p class="mt-4 text-sm text-error" role="alert">{error}</p>{/if}
	<form
		class="mt-6 space-y-5"
		onsubmit={(event) => {
			event.preventDefault();
			confirm = true;
		}}
	>
		<fieldset class="rounded-xl border border-outline-variant p-4">
			<legend class="px-1 font-medium">Vision models</legend
			>{#each models as model (model.id)}<label class="mr-4 inline-flex items-center gap-2 py-1"
					><input
						type="checkbox"
						checked={selected.includes(model.id)}
						onchange={() => toggle(model.id)}
					/>{model.id}</label
				>{/each}
		</fieldset>
		<fieldset class="rounded-xl border border-outline-variant p-4">
			<legend class="px-1 font-medium">Verified receipt selection</legend><label
				class="block text-sm"
				>Explicit receipt IDs <input
					class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
					bind:value={receiptIds}
					placeholder="Comma-separated receipt UUIDs"
				/></label
			>
			<p class="mt-1 text-xs text-on-surface-variant">Explicit IDs override filters.</p>
			<div class="mt-3 grid gap-3 sm:grid-cols-3">
				<label class="text-sm"
					>Category<input
						class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
						bind:value={category}
					/></label
				><label class="text-sm"
					>Maximum source confidence<input
						class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
						type="number"
						min="0"
						max="100"
						bind:value={maxConfidence}
					/></label
				><label class="text-sm"
					>Maximum receipts<input
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
			<legend class="px-1 font-medium">Execution and council</legend>
			<div class="grid gap-3 sm:grid-cols-2">
				<label class="text-sm"
					>Request timeout (seconds)<input
						class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
						type="number"
						min="1"
						bind:value={timeout}
					/></label
				><label class="text-sm"
					>Council policy<select
						class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
						bind:value={policy}
						><option value="all">All differences (€0.01)</option><option value="material"
							>Material (€1 and 2%)</option
						><option value="custom">Custom threshold</option></select
					></label
				><label class="text-sm"
					>Custom absolute €<input
						class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
						type="number"
						min="0"
						step="0.01"
						bind:value={absolute}
					/></label
				><label class="text-sm"
					>Custom relative<input
						class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
						type="number"
						min="0"
						step="0.001"
						bind:value={relative}
					/></label
				>
			</div>
			<label class="mt-3 inline-flex items-center gap-2 text-sm"
				><input type="checkbox" bind:checked={applyFlags} /> Apply council flags</label
			>
		</fieldset>
		<p class="rounded-lg bg-error-container p-3 text-sm text-on-error-container">
			<strong>Before creating:</strong> this can take a long time, consume substantial energy, and make
			the local LLM unavailable for normal receipt extraction while it runs.
		</p>
		<button
			class="rounded-lg bg-error px-4 py-2.5 text-sm font-medium text-on-error disabled:opacity-50"
			disabled={selected.length === 0 || creating}>Create benchmark</button
		>
	</form>
	{#if confirm}<div class="fixed inset-0 z-40 grid place-items-center bg-black/50 p-4">
			<div class="max-w-md rounded-xl bg-surface p-5 shadow-elevation-4">
				<h2 class="text-lg font-semibold">Create benchmark?</h2>
				<p class="mt-2 text-sm text-on-surface-variant">
					This can take a long time, consume substantial energy, and make the local LLM unavailable
					for normal receipt extraction while it runs.
				</p>
				<div class="mt-5 flex justify-end gap-2">
					<button class="rounded-lg px-4 py-2" onclick={() => (confirm = false)}>Cancel</button
					><button
						class="rounded-lg bg-error px-4 py-2 text-on-error"
						disabled={creating}
						onclick={create}>Create benchmark</button
					>
				</div>
			</div>
		</div>{/if}
</section>
