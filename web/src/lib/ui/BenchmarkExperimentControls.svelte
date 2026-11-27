<script lang="ts">
	import type { ExperimentSettings } from '$lib/benchmark-experiments';
	let {
		settings = $bindable(),
		ocrAvailable = false,
		thinkingControl = false
	}: { settings: ExperimentSettings; ocrAvailable?: boolean; thinkingControl?: boolean } = $props();
</script>

<fieldset class="rounded-xl border border-outline-variant p-4">
	<legend class="px-1 font-medium">Extraction experiment</legend>
	<div class="grid gap-3 sm:grid-cols-2">
		<label class="text-sm"
			>Pipeline
			<select
				class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
				bind:value={settings.pipeline}
			>
				<option value="upload">Normal upload pipeline</option>
				<option value="direct_vision">Direct vision model</option>
				<option value="ocr" disabled={!ocrAvailable}>OCR followed by model</option>
			</select>
		</label>
		<label class="text-sm"
			>Temperature
			<input
				class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
				type="number"
				min="0"
				max="2"
				step="0.1"
				placeholder="Current provider setting"
				bind:value={settings.temperature}
			/>
		</label>
		<label class="text-sm"
			>Thinking
			<select
				class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
				bind:value={settings.thinking}
				disabled={!thinkingControl}
			>
				<option value="default">Server default</option>
				<option value="off">Disabled</option>
				<option value="on">Enabled</option>
			</select>
		</label>
	</div>
	<label class="mt-3 flex items-center gap-2 text-sm"
		><input type="checkbox" bind:checked={settings.includeClassification} /> Include categories and tags</label
	>
	<label class="mt-3 flex items-center gap-2 text-sm"
		><input type="checkbox" bind:checked={settings.allowFallback} /> Allow direct vision fallback if OCR
		fails</label
	>
	<p class="mt-2 text-xs text-on-surface-variant">
		These settings apply only to this run. Extraction and classification are scored separately. OCR
		failures stop the experiment unless fallback is selected.
	</p>
	{#if !ocrAvailable}<p class="mt-2 text-xs text-on-surface-variant">
			The OCR extension is not configured. Normal upload currently uses direct vision.
		</p>{/if}
	{#if !thinkingControl}<p class="mt-2 text-xs text-on-surface-variant">
			Thinking overrides are available with Ollama. This provider retains its existing reasoning
			settings.
		</p>{/if}
</fieldset>
