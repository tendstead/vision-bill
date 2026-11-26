<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { createQuery } from '@tanstack/svelte-query';
	import { session } from '$lib/auth';
	import { sessionViewGuard } from '$lib/session-view';
	import { api, ApiError } from '$lib/api/client';
	import { t } from '$lib/i18n';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	import type { LlmAuthenticationUpdate, LlmConnectionResult, SettingsUpdate } from '$lib/types';
	import AppearanceSelect from '$lib/ui/AppearanceSelect.svelte';
	import DedicatedOcr from './DedicatedOcr.svelte';

	const isCurrentView = sessionViewGuard();

	const settings = createQuery(
		() => ({
			queryKey: queryKeys.settings(),
			queryFn: () => api.getSettings()
		}),
		() => queryClient
	);

	const uiConfig = createQuery(
		() => ({ queryKey: queryKeys.uiConfig(), queryFn: () => api.getUiConfig() }),
		() => queryClient
	);
	let form = $state<SettingsUpdate>({
		llm: { provider: 'ollama', host: '', model_name: '', temperature: 0 },
		allow_registration: true
	});
	let baseline = $state<SettingsUpdate | null>(null);
	let initialized = $state(false);
	let saving = $state(false);
	let saved = $state(false);
	let error = $state('');
	let authMode = $state<'keep' | LlmAuthenticationUpdate['mode']>('keep');
	let bearerKey = $state('');
	let discoveredModels = $state<string[]>([]);
	let manualModel = $state(false);
	let connectionRevision = 0;
	let testingConnection = $state(false);
	let connectionResult = $state<
		LlmConnectionResult['status'] | 'request_failed' | 'settings_changed' | null
	>(null);
	const connectionEdited = $derived(
		authMode !== 'keep' ||
			form.llm.host !== baseline?.llm.host ||
			form.llm.provider !== baseline?.llm.provider
	);

	$effect(() => {
		if (!initialized && settings.data) {
			form = {
				llm: { ...settings.data.llm },
				allow_registration: settings.data.allow_registration
			};
			baseline = structuredClone($state.snapshot(form));
			initialized = true;
		}
	});

	$effect(() => {
		if ($session !== 'loading' && $session !== null && !$session.is_admin) {
			void goto(resolve('/'));
		}
	});

	$effect(() => {
		if ($session === null || !isCurrentView()) resetEndpoint();
		if (authMode !== 'bearer') bearerKey = '';
	});

	function clearDiscovery() {
		connectionRevision++;
		connectionResult = null;
		discoveredModels = [];
		manualModel = false;
	}

	function resetEndpoint() {
		clearDiscovery();
		authMode = 'keep';
		bearerKey = '';
	}

	function source(field: string): string {
		return settings.data?.sources[field] ?? 'default';
	}

	function sourceLabel(value: string): string {
		return $t(`settings.source${value[0].toUpperCase()}${value.slice(1)}`);
	}

	async function testConnection() {
		if (!isCurrentView() || connectionEdited || testingConnection || saving) return;
		clearDiscovery();
		const revision = connectionRevision;
		testingConnection = true;
		try {
			const result = await api.testLlmConnection(form.llm.host, form.llm.provider);
			if (isCurrentView() && revision === connectionRevision) {
				connectionResult = result.status;
				if (result.status === 'connected') discoveredModels = result.models;
			}
		} catch (cause) {
			if (isCurrentView() && revision === connectionRevision)
				connectionResult =
					cause instanceof ApiError && cause.status === 409 ? 'settings_changed' : 'request_failed';
		} finally {
			if (isCurrentView()) testingConnection = false;
		}
	}

	async function save() {
		if (!isCurrentView() || saving || testingConnection) return;
		saving = true;
		saved = false;
		error = '';
		try {
			const changed_paths = ['provider', 'host', 'model_name', 'temperature']
				.filter(
					(key) =>
						form.llm[key as keyof typeof form.llm] !== baseline?.llm[key as keyof typeof form.llm]
				)
				.map((key) => `llm.${key}`);
			if (form.allow_registration !== baseline?.allow_registration)
				changed_paths.push('auth.allow_registration');
			let llm_authentication: LlmAuthenticationUpdate | undefined;
			if (authMode === 'none') llm_authentication = { mode: 'none' };
			if (authMode === 'bearer') llm_authentication = { mode: 'bearer', api_key: bearerKey };
			const result = await api.updateSettings({ ...form, changed_paths, llm_authentication });
			if (!isCurrentView()) return;
			resetEndpoint();
			form = { llm: { ...result.llm }, allow_registration: result.allow_registration };
			baseline = structuredClone($state.snapshot(form));
			queryClient.setQueryData(queryKeys.settings(), result);
			await Promise.all([
				queryClient.invalidateQueries({ queryKey: queryKeys.uiConfig() }),
				queryClient.invalidateQueries({ queryKey: queryKeys.dedicatedOcr() })
			]);
			if (isCurrentView()) saved = true;
		} catch (cause) {
			if (!isCurrentView()) return;
			error = cause instanceof ApiError ? cause.detail : $t('settings.saveError');
		} finally {
			if (isCurrentView()) saving = false;
		}
	}
</script>

<svelte:head>
	<title>{$t('pages.settings.title')} – {$t('app.name')}</title>
</svelte:head>

<section>
	<h1 class="text-2xl font-semibold">{$t('pages.settings.title')}</h1>
	<p class="mt-2 text-on-surface-variant">{$t('settings.description')}</p>
	<div class="mt-6 max-w-2xl rounded-xl border border-outline-variant bg-surface-container-low p-5">
		<AppearanceSelect />
	</div>

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
								>{sourceLabel(source('llm.provider'))}</span
							></span
						>
						<select
							class="mt-1 block w-full rounded-lg border border-outline-variant bg-surface p-2.5"
							onchange={resetEndpoint}
							bind:value={form.llm.provider}
						>
							<option value="ollama">Ollama</option><option value="openai">OpenAI</option>
						</select>
					</label>
					<label class="block text-sm">
						<span class="flex items-center justify-between gap-2"
							><span>{$t('settings.host')}</span><span class="text-xs text-on-surface-variant"
								>{sourceLabel(source('llm.host'))}</span
							></span
						>
						<input
							class="mt-1 block w-full rounded-lg border border-outline-variant bg-surface p-2.5"
							oninput={resetEndpoint}
							bind:value={form.llm.host}
						/>
					</label>
					{#if settings.data.llm_authentication}
						<div class="space-y-2">
							<label class="block text-sm">
								{$t('settings.authentication.label')}
								<select
									class="mt-1 block w-full rounded-lg border border-outline-variant bg-surface p-2.5"
									onchange={clearDiscovery}
									bind:value={authMode}
									disabled={saving}
								>
									<option value="keep">{$t('settings.authentication.keep')}</option>
									<option value="none">{$t('settings.authentication.none')}</option>
									<option value="bearer">{$t('settings.authentication.bearer')}</option>
								</select>
							</label>
							<p class="text-xs text-on-surface-variant">
								{$t(`settings.authentication.status.${settings.data.llm_authentication.status}`)}
								<span class="ml-1">· {sourceLabel(settings.data.llm_authentication.source)}</span>
							</p>
							{#if authMode === 'bearer'}
								<label class="block text-sm">
									{$t('settings.authentication.newKey')}
									<input
										class="mt-1 block w-full rounded-lg border border-outline-variant bg-surface p-2.5"
										type="password"
										autocomplete="new-password"
										spellcheck="false"
										maxlength="8192"
										required
										bind:value={bearerKey}
										disabled={saving}
									/>
								</label>
							{/if}
							<button
								type="button"
								class="rounded-lg border border-outline px-3 py-2 text-sm disabled:opacity-50"
								onclick={testConnection}
								disabled={connectionEdited || testingConnection || saving}
							>
								{$t(testingConnection ? 'common.loading' : 'settings.authentication.test')}
							</button>
							{#if connectionEdited}
								<p class="text-xs text-on-surface-variant">
									{$t('settings.authentication.saveFirst')}
								</p>
							{:else if connectionResult}
								<p
									class="text-sm {connectionResult === 'connected' ? 'text-primary' : 'text-error'}"
									role="status"
								>
									{$t(
										connectionResult === 'connected' && !discoveredModels.length
											? 'settings.authentication.emptyModels'
											: `settings.authentication.result.${connectionResult}`
									)}
								</p>
							{/if}
						</div>
					{/if}
					<label class="block text-sm">
						<span class="flex items-center justify-between gap-2"
							><span>{$t('settings.model')}</span><span class="text-xs text-on-surface-variant"
								>{sourceLabel(source('llm.model_name'))}</span
							></span
						>
						{#if discoveredModels.length && !manualModel && !connectionEdited}
							<select
								class="mt-1 block w-full rounded-lg border border-outline-variant bg-surface p-2.5"
								bind:value={form.llm.model_name}
							>
								{#if !discoveredModels.includes(form.llm.model_name)}
									<option value={form.llm.model_name}
										>{form.llm.model_name || $t('settings.authentication.chooseModel')}</option
									>
								{/if}
								{#each discoveredModels as id (id)}
									<option value={id}>{id}</option>
								{/each}
							</select>
						{:else}
							<input
								class="mt-1 block w-full rounded-lg border border-outline-variant bg-surface p-2.5"
								bind:value={form.llm.model_name}
							/>
						{/if}
					</label>
					{#if discoveredModels.length && !connectionEdited}
						<button
							type="button"
							class="text-sm text-primary underline"
							onclick={() => (manualModel = !manualModel)}
						>
							{$t(
								manualModel ? 'settings.authentication.fromList' : 'settings.authentication.manual'
							)}
						</button>
					{/if}
					<p class="text-xs text-on-surface-variant">
						{$t('settings.formatterLimit')}: {uiConfig.data?.formatter_max_output_tokens ??
							$t('settings.unknown')}
					</p>
					{#if uiConfig.data?.ocr_enabled}
						<p class="text-xs text-on-surface-variant">{$t('settings.ocrLimitUnknown')}</p>
					{/if}
					<label class="block text-sm">
						<span class="flex items-center justify-between gap-2"
							><span>{$t('settings.temperature')}</span><span
								class="text-xs text-on-surface-variant"
								>{sourceLabel(source('llm.temperature'))}</span
							></span
						>
						<input
							class="mt-1 block w-full rounded-lg border border-outline-variant bg-surface p-2.5"
							type="number"
							min="0"
							max="2"
							step="0.1"
							bind:value={form.llm.temperature}
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
							class="text-xs text-on-surface-variant"
							>{sourceLabel(source('auth.allow_registration'))}</span
						></span
					>
					<input
						type="checkbox"
						class="size-4"
						bind:checked={form.allow_registration}
						disabled={source('auth.allow_registration') === 'environment'}
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
				class="rounded-lg bg-action px-4 py-2.5 text-sm font-medium text-on-primary disabled:opacity-50"
				disabled={saving || testingConnection}
			>
				{saving ? $t('common.loading') : $t('common.save')}
			</button>
		</form>
	{/if}

	{#if $session && $session !== 'loading' && $session.is_admin}
		<DedicatedOcr />
	{/if}
</section>
