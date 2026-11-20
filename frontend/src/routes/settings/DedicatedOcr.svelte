<script lang="ts">
	import { resolve } from '$app/paths';
	import { createQuery } from '@tanstack/svelte-query';
	import { api, ApiError } from '$lib/api/client';
	import { t } from '$lib/i18n';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	import { captureSessionActor } from '$lib/session-lifetime';
	import { sessionViewGuard } from '$lib/session-view';

	const actor = captureSessionActor();
	const isCurrentView = sessionViewGuard();
	const queryKey = [...queryKeys.dedicatedOcr(), actor?.generation];
	const status = createQuery(
		() => ({
			queryKey,
			queryFn: () => api.getDedicatedOcr(actor),
			staleTime: 0,
			refetchOnWindowFocus: true,
			refetchInterval: (query) => {
				const state = query.state.data?.state;
				return state === 'requested' || state === 'installing' ? 2000 : false;
			}
		}),
		() => queryClient
	);

	const reasons = new Set([
		'gpu_profile_required',
		'model_request_unsettled',
		'model_handoff_model_not_adopted',
		'model_handoff_foreign_residency',
		'model_handoff_residency_unknown',
		'model_handoff_busy',
		'model_release_unconfirmed',
		'ocr_setup_pending',
		'ocr_activation_state_invalid',
		'ocr_config_managed_externally',
		'llm_lifecycle_required',
		'llm_quiescence_required',
		'llm_request_recovery_required',
		'ocr_handoff_confirmation_required',
		'ocr_runtime_unavailable',
		'ocr_gpu_out_of_memory',
		'ocr_download_failed',
		'native_model_download_failed',
		'native_model_integrity_failed',
		'native_device_identity_unavailable',
		'native_package_mismatch',
		'native_installation_write_failed',
		'native_activation_inputs_invalid',
		'native_qualification_failed',
		'native_serving_startup_failed',
		'native_activation_recovery_required',
		'native_installation_busy'
	]);
	const installed = $derived(status.data?.installed ?? false);
	const activationState = $derived(status.data?.state);
	const selected = $derived(status.data?.selected ?? activationState === 'active');
	const sharedGpu = $derived(status.data?.shared_gpu ?? true);
	const reason = $derived(status.data?.reason);
	const runtimeUnavailable = $derived(reason === 'ocr_runtime_unavailable');
	const message = $derived(
		reason && reasons.has(reason)
			? `settings.dedicatedOcr.reasons.${reason}`
			: installed && activationState === 'disabled'
				? 'settings.dedicatedOcr.installedOff'
				: `settings.dedicatedOcr.states.${activationState}`
	);
	const canRequest = $derived(
		!runtimeUnavailable &&
			(activationState === 'disabled' ||
				activationState === 'failed' ||
				activationState === 'action_required')
	);
	let confirmHandoff = $state(false);
	let submitting = $state(false);
	let requestFailed = $state(false);
	let requestReason = $state<string | null>(null);
	let topologyDraft = $state<boolean | null>(null);
	const busy = $derived(
		submitting || activationState === 'requested' || activationState === 'installing'
	);
	const topologyShared = $derived(topologyDraft ?? sharedGpu);
	const topologyChanged = $derived(topologyShared !== sharedGpu);
	const canChooseTopology = $derived(
		!!activationState && activationState !== 'unavailable' && !status.isError && !busy
	);
	const needsTopologyHandoff = $derived(topologyChanged && topologyShared && installed);
	const needsSetupHandoff = $derived(
		!topologyChanged && sharedGpu && canRequest && !selected && !installed
	);
	const canSaveTopology = $derived(
		canChooseTopology && topologyChanged && (!needsTopologyHandoff || confirmHandoff)
	);
	const canToggle = $derived(
		!busy &&
			!status.isError &&
			!topologyChanged &&
			(selected || (canRequest && (!needsSetupHandoff || confirmHandoff)))
	);

	$effect(() => {
		if ((activationState === 'active' || activationState === 'disabled') && isCurrentView()) {
			void queryClient.invalidateQueries({ queryKey: queryKeys.uiConfig() });
			void queryClient.invalidateQueries({ queryKey: ['models'] });
		}
	});

	const canResumeDirect = $derived(
		activationState === 'failed' &&
			!busy &&
			!topologyChanged &&
			!status.isError &&
			reason !== 'native_activation_recovery_required' &&
			reason !== 'native_installation_busy'
	);

	async function changeDedicatedOcr(targetSharedGpu: boolean | null = null, resumeDirect = false) {
		const previous = status.data;
		if (
			!previous ||
			!isCurrentView() ||
			!(resumeDirect ? canResumeDirect : targetSharedGpu === null ? canToggle : canSaveTopology)
		)
			return;
		submitting = true;
		requestFailed = false;
		requestReason = null;
		const confirmed = confirmHandoff;
		confirmHandoff = false;
		try {
			const result =
				targetSharedGpu === null
					? await api.requestDedicatedOcr(
							actor,
							resumeDirect ? false : confirmed,
							resumeDirect ? false : !selected
						)
					: await api.setDedicatedOcrGpuTopology(actor, targetSharedGpu, confirmed);
			if (!isCurrentView()) return;
			await queryClient.cancelQueries({ queryKey });
			if (isCurrentView()) queryClient.setQueryData(queryKey, result);
		} catch (error) {
			if (!isCurrentView()) return;
			const refreshed = await status.refetch();
			if (isCurrentView()) {
				requestReason =
					error instanceof ApiError &&
					error.status === 409 &&
					reasons.has(error.detail) &&
					error.detail !== refreshed.data?.reason
						? error.detail
						: null;
				requestFailed =
					!requestReason &&
					(refreshed.isError ||
						(targetSharedGpu === null
							? refreshed.data?.state === previous.state &&
								refreshed.data.reason === previous.reason
							: (refreshed.data?.shared_gpu ?? true) !== targetSharedGpu));
			}
		} finally {
			if (isCurrentView()) {
				if (targetSharedGpu !== null) topologyDraft = null;
				submitting = false;
			}
		}
	}
</script>

{#snippet handoffConfirmation()}
	<label class="flex items-start gap-2 text-sm">
		<input
			type="checkbox"
			class="mt-0.5 size-4 shrink-0"
			bind:checked={confirmHandoff}
			disabled={busy}
		/>
		<span>{$t('settings.dedicatedOcr.confirmHandoff')}</span>
	</label>
{/snippet}

<section
	class="mt-6 max-w-2xl space-y-3 rounded-xl border border-outline-variant bg-surface-container-low p-5"
	aria-labelledby="dedicated-ocr-title"
>
	<h2 class="font-semibold" id="dedicated-ocr-title">{$t('settings.dedicatedOcr.title')}</h2>
	{#if activationState}
		<p class="text-sm text-on-surface-variant" role="status">{$t(message)}</p>
	{:else if status.isPending}
		<p class="text-sm text-on-surface-variant" role="status">{$t('common.loading')}</p>
	{/if}
	{#if status.isError}
		<p class="text-sm text-error" role="alert">{$t('settings.dedicatedOcr.loadError')}</p>
	{/if}
	{#if status.isError || runtimeUnavailable}
		<button
			type="button"
			class="rounded-lg border border-outline-variant px-4 py-2.5 text-sm font-medium disabled:opacity-50"
			disabled={status.isFetching || submitting}
			onclick={() => void status.refetch()}
		>
			{$t('settings.dedicatedOcr.retryStatus')}
		</button>
	{/if}
	{#if requestReason}
		<p class="text-sm text-error" role="alert">
			{$t(`settings.dedicatedOcr.reasons.${requestReason}`)}
		</p>
	{:else if requestFailed}
		<p class="text-sm text-error" role="alert">{$t('settings.dedicatedOcr.requestError')}</p>
	{/if}
	{#if activationState && activationState !== 'unavailable'}
		<div class="space-y-2 border-b border-outline-variant pb-3">
			<label class="flex items-start gap-2 text-sm font-medium">
				<input
					type="checkbox"
					class="mt-0.5 size-4 shrink-0"
					checked={topologyShared}
					disabled={!canChooseTopology}
					aria-describedby="dedicated-ocr-gpu-description"
					onchange={(event) => {
						const checked = event.currentTarget.checked;
						topologyDraft = checked === sharedGpu ? null : checked;
						confirmHandoff = false;
					}}
				/>
				<span>{$t('settings.dedicatedOcr.sharedGpu')}</span>
			</label>
			<p id="dedicated-ocr-gpu-description" class="text-sm text-on-surface-variant">
				{$t('settings.dedicatedOcr.sharedGpuHelp')}
			</p>
			{#if needsTopologyHandoff}
				{@render handoffConfirmation()}
			{/if}
			<button
				type="button"
				class="rounded-lg border border-outline-variant px-4 py-2.5 text-sm font-medium disabled:opacity-50"
				disabled={!canSaveTopology}
				onclick={() => changeDedicatedOcr(topologyShared)}
			>
				{$t('settings.dedicatedOcr.saveGpuTopology')}
			</button>
		</div>
	{/if}
	<button
		type="button"
		role="switch"
		aria-label={$t('settings.dedicatedOcr.title')}
		aria-checked={selected}
		aria-busy={busy}
		class="inline-flex items-center gap-3 rounded-full border border-outline-variant px-4 py-2.5 text-sm font-medium disabled:opacity-50"
		disabled={!canToggle}
		onclick={() => changeDedicatedOcr()}
	>
		<span
			aria-hidden="true"
			class:bg-action={selected}
			class="flex h-6 w-11 items-center rounded-full bg-outline-variant p-1"
		>
			<span
				class:translate-x-5={selected}
				class="size-4 rounded-full bg-surface transition-transform"
			></span>
		</span>
		{$t('settings.dedicatedOcr.title')}
	</button>
	{#if canResumeDirect}
		<button
			type="button"
			class="rounded-lg border border-outline-variant px-4 py-2.5 text-sm font-medium"
			onclick={() => changeDedicatedOcr(null, true)}
		>
			{$t('settings.dedicatedOcr.resumeDirect')}
		</button>
	{/if}
	{#if needsSetupHandoff && !status.isError}
		{@render handoffConfirmation()}
	{/if}
	<p class="text-sm text-on-surface-variant">{$t('settings.dedicatedOcr.prerequisites')}</p>
	<a class="text-sm underline" href={resolve('/settings/ocr-guide')}>
		{$t('settings.dedicatedOcr.setupGuide')}
	</a>
</section>
