<script lang="ts">
	import { captureSessionActor, isCurrentSessionActor } from '$lib/session-lifetime';
	import { diagnosticLabel } from '$lib/diagnostics';
	import { createQuery } from '@tanstack/svelte-query';
	import { resolve } from '$app/paths';
	import { api, ApiError } from '$lib/api/client';
	import { requestKey } from '$lib/api/upload';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	import { t } from '$lib/i18n';
	import type { SubmissionStatus } from '$lib/types';

	let { id, editingDraft = false }: { id: string; editingDraft?: boolean } = $props();
	const actor = captureSessionActor();
	let retrying = $state(false);
	let retryError = $state('');
	let retryRequest: {
		id: string;
		expected_generation: number;
		request_key: string;
		policy: 'preserve' | 'current';
	} | null = null;
	let activeView: { id: string } | null = null;
	let activeRetry: { id: string; generation: number; view: object } | null = null;

	$effect(() => {
		const view = { id };
		activeView = view;
		activeRetry = null;
		retryRequest = null;
		retrying = false;
		retryError = '';
		// Also runs on unmount: an old request no longer owns cache or UI effects.
		return () => {
			if (activeView === view) {
				activeView = null;
				activeRetry = null;
			}
		};
	});
	const job = createQuery(
		() => ({
			queryKey: queryKeys.image(id),
			queryFn: () => api.getImage(id),
			retry: false,
			staleTime: 0,
			refetchInterval: (query) => (query.state.data?.terminal ? false : 2000)
		}),
		() => queryClient
	);

	const policyHeld = $derived(
		(job.data?.status === 'pending' || job.data?.status === 'processing') &&
			job.data.wait_reason === 'execution_policy_changed' &&
			!job.data.receipt_id
	);

	async function retry(policy: 'preserve' | 'current' = 'preserve') {
		if (
			!isCurrentSessionActor(actor) ||
			!activeView ||
			activeView.id !== id ||
			retrying ||
			!job.data ||
			!(policy === 'current' ? policyHeld : job.data.terminal) ||
			job.data.status === 'analyzed'
		)
			return;
		const request = { id, generation: job.data.generation, view: activeView };
		const key = queryKeys.image(request.id);
		const isCurrent = () =>
			isCurrentSessionActor(actor) &&
			activeView === request.view &&
			activeRetry === request &&
			id === request.id;
		activeRetry = request;
		retrying = true;
		retryError = '';
		try {
			if (
				retryRequest?.id !== request.id ||
				retryRequest.expected_generation !== request.generation ||
				retryRequest.policy !== policy
			) {
				retryRequest = {
					id: request.id,
					expected_generation: request.generation,
					request_key: requestKey(),
					policy
				};
			}
			const result = await api.retrySubmission(request.id, {
				expected_generation: retryRequest.expected_generation,
				request_key: retryRequest.request_key,
				...(retryRequest.policy === 'current' ? { policy: retryRequest.policy } : {})
			});
			if (!isCurrent()) return;
			const cached = queryClient.getQueryData<SubmissionStatus>(key);
			// A GET may already know a later generation or later state within this generation.
			// A delayed retry response must never replace either one.
			if (!cached || cached.generation < result.generation) {
				await queryClient.cancelQueries({ queryKey: key, exact: true });
				if (!isCurrent()) return;
				queryClient.setQueryData<SubmissionStatus>(key, (latest) =>
					latest && latest.generation >= result.generation ? latest : result
				);
			}
			if (!isCurrent()) return;
			retryRequest = null;
			void queryClient.invalidateQueries({ queryKey: key, exact: true }, { cancelRefetch: false });
			// Refresh list/count observers without cancelling a different job's GET.
			void queryClient.invalidateQueries({
				queryKey: ['images'],
				predicate: (query) => typeof query.queryKey[1] !== 'string'
			});
		} catch (error) {
			if (!isCurrent()) return;
			retryError = error instanceof ApiError ? `${error.status}: ${error.detail}` : String(error);
			if (error instanceof ApiError && error.status === 409) {
				void queryClient.refetchQueries({ queryKey: key, exact: true });
			}
		} finally {
			if (isCurrent()) {
				activeRetry = null;
				retrying = false;
			}
		}
	}
</script>

<div class="my-4 rounded-xl border border-outline-variant bg-surface-container-low p-4">
	{#if job.error}
		<p role="alert">{$t('submission.loadError')} {job.error.message}</p>
		<button type="button" class="mt-3 text-primary underline" onclick={() => job.refetch()}
			>{$t('submission.refresh')}</button
		>
	{:else if !job.data}
		<p role="status">{$t('common.loading')}</p>
	{:else}
		<p class="text-lg font-medium" role="status" aria-live="polite">
			{#if job.data.observations_available && !job.data.terminal}
				{$t(
					job.data.status === 'processing'
						? 'submission.finishingSaved'
						: 'submission.savedObservations'
				)}
			{:else}
				{$t(`submission.${job.data.status}`)}
			{/if}
		</p>
		<p class="mt-2 text-sm">
			{$t('submission.attempt', {
				values: { count: job.data.attempt_count, max: job.data.max_attempts }
			})}
		</p>
		{#if !job.data.terminal}
			<p class="mt-2 text-sm text-on-surface-variant">{$t('submission.returnLater')}</p>
			{#if job.data.retry_at}
				<p class="mt-2 text-sm">
					{$t('submission.retryAt', {
						values: { time: new Date(job.data.retry_at).toLocaleString() }
					})}
				</p>
			{/if}
			{#if job.data.wait_reason}<p class="mt-2 text-sm">
					{$t('submission.waitReason')}: {diagnosticLabel($t, 'reasons', job.data.wait_reason)} ({job
						.data.wait_reason})
					{#if job.data.status === 'pending' && job.data.wait_reason !== 'retry_backoff'}{$t(
							'diagnostics.readinessWait'
						)}{/if}
				</p>{/if}
		{/if}
		{#if job.data.result_code && !(editingDraft && job.data.result_code === 'needs_review')}<p
				class="mt-2 text-sm"
			>
				{$t('submission.resultCode')}: {diagnosticLabel($t, 'reasons', job.data.result_code)} ({job
					.data.result_code})
			</p>{/if}
		{#if job.data.error && job.data.error !== job.data.result_code}<p
				class="mt-2 break-words text-sm text-error"
			>
				{job.data.error}
			</p>{/if}
		<div class="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2">
			{#if job.data.receipt_id}
				<a
					class="inline-block rounded-lg bg-action px-4 py-2 text-on-primary"
					href={resolve(`/receipts/${job.data.receipt_id}`)}>{$t('submission.openReceipt')}</a
				>
			{/if}
			{#if !editingDraft && job.data.draft_available && !job.data.observations_available}
				<a
					class="inline-block rounded-lg bg-action px-4 py-2 text-on-primary"
					href={resolve(`/upload?job=${id}&draft=1`)}>{$t('submission.openDraft')}</a
				>
			{/if}
			{#if job.data.terminal && job.data.status !== 'analyzed'}
				<button
					type="button"
					class="rounded-lg border border-outline px-3 py-2 text-on-surface hover:bg-surface-container-high disabled:opacity-50"
					disabled={retrying}
					onclick={() => retry()}>{$t('submission.retry')}</button
				>
			{/if}
			{#if policyHeld}
				<button
					type="button"
					class="rounded-lg bg-action px-4 py-2 text-on-primary disabled:opacity-50"
					disabled={retrying}
					onclick={() => retry('current')}>{$t('submission.retryCurrentPolicy')}</button
				>
			{/if}
			<a class="inline-block text-sm text-primary underline" href={resolve('/upload')}
				>{$t('submission.newUpload')}</a
			>
		</div>
		{#if !editingDraft && job.data.source_views?.length}
			<ul class="mt-4 flex flex-wrap gap-3">
				{#each job.data.source_views as view (view.view_id)}
					{@const href = api.sourceViewUrl(view.url, actor)}
					{#if href}
						<li>
							<!-- Authenticated artifact URLs are supplied by the API. -->
							<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -->
							<a class="text-primary underline" {href} target="_blank" rel="noreferrer"
								>{$t('submission.photo', { values: { number: view.photo_index + 1 } })} ({$t(
									view.coordinate_space === 'working'
										? 'receiptWorkspace.working'
										: 'review.original'
								)})</a
							>
						</li>
					{/if}
				{/each}
			</ul>
		{/if}
		<details class="mt-2 text-xs text-on-surface-variant">
			<summary class="cursor-pointer">{$t('submission.reference')}</summary>
			<p class="mt-1 break-all">{id}</p>
		</details>
	{/if}
	{#if retryError}<p class="mt-3 text-sm text-error" role="alert">{retryError}</p>{/if}
	{#if !job.data || job.error}<a
			class="mt-4 inline-block text-primary underline"
			href={resolve('/upload')}>{$t('submission.newUpload')}</a
		>{/if}
</div>
