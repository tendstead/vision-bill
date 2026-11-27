<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { createQuery } from '@tanstack/svelte-query';
	import { session } from '$lib/auth';
	import { api, ApiError } from '$lib/api/client';
	import { t } from '$lib/i18n';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	import type { SettingsUpdate } from '$lib/types';

	const settings = createQuery(
		() => ({
			queryKey: queryKeys.settings(),
			queryFn: () => api.getSettings()
		}),
		() => queryClient
	);

	let form = $state<SettingsUpdate>({
		llm: { provider: 'ollama', host: '', model_name: '', temperature: 0 },
		allow_registration: true
	});
	let initialized = $state(false);
	let saving = $state(false);
	let saved = $state(false);
	let error = $state('');

	$effect(() => {
		if (!initialized && settings.data) {
			form = {
				llm: { ...settings.data.llm },
				allow_registration: settings.data.allow_registration
			};
			initialized = true;
		}
	});

	$effect(() => {
		if ($session !== 'loading' && $session !== null && !$session.is_admin) {
			void goto(resolve('/'));
		}
	});

	function source(field: string): string {
		return settings.data?.sources[field] ?? 'default';
	}

	function isLocked(field: string): boolean {
		return source(field) === 'environment';
	}

	function sourceLabel(field: string): string {
		return $t(`settings.source${source(field)[0].toUpperCase()}${source(field).slice(1)}`);
	}

	async function save() {
		if (saving) return;
		saving = true;
		saved = false;
		error = '';
		try {
			const result = await api.updateSettings(form);
			form = { llm: { ...result.llm }, allow_registration: result.allow_registration };
			queryClient.setQueryData(queryKeys.settings(), result);
			await queryClient.invalidateQueries({ queryKey: queryKeys.uiConfig() });
			saved = true;
		} catch (cause) {
			error = cause instanceof ApiError ? cause.detail : $t('settings.saveError');
		} finally {
			saving = false;
		}
	}
</script>

<svelte:head>
	<title>{$t('pages.settings.title')} – {$t('app.name')}</title>
</svelte:head>

<section>
	<h1 class="text-2xl font-semibold">{$t('pages.settings.title')}</h1>
	<p class="mt-2 text-on-surface-variant">{$t('settings.description')}</p>

	{#if settings.isLoading}
		<p class="mt-6 text-sm text-on-surface-variant">{$t('common.loading')}</p>
	{:else if settings.error}
		<p class="mt-6 text-sm text-error" role="alert">{$t('settings.loadError')}</p>
	{:else if settings.data}
		<form
			class="mt-6 max-w-2xl space-y-6"
			onsubmit={(event) => {
				event.preventDefault();
				void save();
			}}
		>
			<fieldset class="rounded-xl border border-outline-variant bg-surface-container-low p-5">
				<legend class="px-1 font-semibold">{$t('settings.llm')}</legend>
				<div class="mt-2 space-y-4">
					<label class="block text-sm">
						<span class="flex items-center justify-between gap-2"
							><span>{$t('settings.provider')}</span><span class="text-xs text-on-surface-variant"
								>{sourceLabel('llm.provider')}</span
							></span
						>
						<select
							class="mt-1 block w-full rounded-lg border border-outline-variant bg-surface p-2.5"
							bind:value={form.llm.provider}
							disabled={isLocked('llm.provider')}
						>
							<option value="ollama">Ollama</option><option value="anthropic">Anthropic</option
							><option value="openai">OpenAI</option>
						</select>
					</label>
					<label class="block text-sm">
						<span class="flex items-center justify-between gap-2"
							><span>{$t('settings.host')}</span><span class="text-xs text-on-surface-variant"
								>{sourceLabel('llm.host')}</span
							></span
						>
						<input
							class="mt-1 block w-full rounded-lg border border-outline-variant bg-surface p-2.5"
							bind:value={form.llm.host}
							disabled={isLocked('llm.host')}
						/>
					</label>
					<label class="block text-sm">
						<span class="flex items-center justify-between gap-2"
							><span>{$t('settings.model')}</span><span class="text-xs text-on-surface-variant"
								>{sourceLabel('llm.model_name')}</span
							></span
						>
						<input
							class="mt-1 block w-full rounded-lg border border-outline-variant bg-surface p-2.5"
							bind:value={form.llm.model_name}
							disabled={isLocked('llm.model_name')}
						/>
					</label>
					<label class="block text-sm">
						<span class="flex items-center justify-between gap-2"
							><span>{$t('settings.temperature')}</span><span
								class="text-xs text-on-surface-variant">{sourceLabel('llm.temperature')}</span
							></span
						>
						<input
							class="mt-1 block w-full rounded-lg border border-outline-variant bg-surface p-2.5"
							type="number"
							min="0"
							max="2"
							step="0.1"
							bind:value={form.llm.temperature}
							disabled={isLocked('llm.temperature')}
						/>
					</label>
				</div>
				<p class="mt-4 rounded-lg bg-warning-container p-3 text-sm text-on-warning-container">
					{$t('settings.providerWarning')}
				</p>
			</fieldset>

			<fieldset class="rounded-xl border border-outline-variant bg-surface-container-low p-5">
				<legend class="px-1 font-semibold">{$t('settings.registration')}</legend>
				<label class="mt-2 flex items-center justify-between gap-3 text-sm">
					<span
						><span class="block">{$t('settings.allowRegistration')}</span><span
							class="text-xs text-on-surface-variant">{sourceLabel('auth.allow_registration')}</span
						></span
					>
					<input
						type="checkbox"
						class="size-4"
						bind:checked={form.allow_registration}
						disabled={isLocked('auth.allow_registration')}
					/>
				</label>
			</fieldset>

			{#if settings.data.restart_required}
				<p class="rounded-lg bg-warning-container p-3 text-sm text-on-warning-container">
					{$t('settings.restartRequired')}
				</p>
			{/if}
			{#if error}<p class="text-sm text-error" role="alert">{error}</p>{/if}
			{#if saved}<p class="text-sm text-primary" role="status">{$t('settings.saved')}</p>{/if}
			<button
				type="submit"
				class="rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-on-primary disabled:opacity-50"
				disabled={saving}
			>
				{saving ? $t('common.loading') : $t('common.save')}
			</button>
		</form>
	{/if}
</section>
