<script lang="ts">
	import { t } from '$lib/i18n';
	import type { ExperimentSettings } from '$lib/benchmark-experiments';
	let {
		settings = $bindable(),
		thinkingControl = false
	}: { settings: ExperimentSettings; thinkingControl?: boolean } = $props();
</script>

<fieldset class="rounded-xl border border-outline-variant p-4">
	<legend class="px-1 font-medium">{$t('benchmarkUi.experiment')}</legend>
	<div class="grid gap-3 sm:grid-cols-2">
		<label class="text-sm"
			>{$t('benchmarkUi.temperature')}
			<input
				class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
				type="number"
				min="0"
				max="2"
				step="any"
				placeholder={$t('benchmarkUi.providerSetting')}
				bind:value={settings.temperature}
			/>
		</label>
		<label class="text-sm"
			>{$t('benchmarkUi.thinking')}
			<select
				class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
				bind:value={settings.thinking}
				disabled={!thinkingControl}
			>
				<option value="default">{$t('benchmarkUi.default')}</option>
				<option value="off">{$t('benchmarkUi.off')}</option>
				<option value="on">{$t('benchmarkUi.on')}</option>
			</select>
		</label>
	</div>
	<label class="mt-3 flex items-center gap-2 text-sm"
		><input type="checkbox" bind:checked={settings.includeClassification} />
		{$t('benchmarkUi.classification')}</label
	>
	<p class="mt-2 text-xs text-on-surface-variant">
		{$t('benchmarkForm.settingsHint')}
	</p>
	{#if !thinkingControl}<p class="mt-2 text-xs text-on-surface-variant">
			{$t('benchmarkUi.thinkingHint')}
		</p>{/if}
</fieldset>
