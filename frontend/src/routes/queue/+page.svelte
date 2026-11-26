<script lang="ts">
	import { diagnosticLabel, hasPreviousAttemptResult } from '$lib/diagnostics';
	import { createQuery } from '@tanstack/svelte-query';
	import { sessionViewGuard } from '$lib/session-view';
	import { resolve } from '$app/paths';
	import { t, translate } from '$lib/i18n';
	import { api, ApiError } from '$lib/api/client';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	import { formatRelativeTime } from '$lib/ui/time';
	import { snackbar } from '$lib/ui/snackbar.svelte';
	import ConfirmDialog from '$lib/ui/ConfirmDialog.svelte';
	import Icon from '$lib/ui/Icon.svelte';
	import ReceiptImage from '$lib/ui/ReceiptImage.svelte';
	import type { ImageRow, ImageStatus } from '$lib/types';

	const isCurrentView = sessionViewGuard();

	const statuses: ImageStatus[] = [
		'pending',
		'processing',
		'failed',
		'timed_out',
		'unreadable',
		'needs_review'
	];

	const list = createQuery(
		() => ({
			queryKey: queryKeys.images({ status: statuses }),
			queryFn: () => api.listImages({ status: statuses, limit: 100 }),
			refetchInterval: 15_000
		}),
		() => queryClient
	);

	let analyzing = $state(false);
	let confirmDelete: ImageRow | null = $state(null);
	let deleting = $state(false);
	let currentDelete = $derived(
		list.data?.find((image) => image.id === confirmDelete?.id) ?? confirmDelete
	);
	let analyzeError = $state('');
	let deleteError = $state('');

	async function analyzeNow() {
		if (!isCurrentView() || analyzing) {
			return;
		}
		analyzing = true;
		analyzeError = '';
		try {
			const response = await api.analyzePending();
			if (!isCurrentView()) return;
			if (response.results.length > 0) {
				const failed = response.results.filter((r) => r.status === 'failed').length;
				snackbar.notify(
					failed > 0 ? 'error' : 'success',
					translate('queue.analyzed', { values: { count: response.results.length } })
				);
			}
			await queryClient.invalidateQueries({ queryKey: ['images'] });
			if (!isCurrentView()) return;
			await queryClient.invalidateQueries({ queryKey: ['receipts'] });
		} catch (error) {
			if (!isCurrentView()) return;
			analyzeError =
				error instanceof ApiError && error.status === 503
					? translate('upload.errorDb')
					: translate('queue.analyzeFailed');
		} finally {
			if (isCurrentView()) analyzing = false;
		}
	}

	async function removeImage() {
		if (
			!isCurrentView() ||
			!confirmDelete ||
			deleting ||
			currentDelete?.receipt_id ||
			currentDelete?.status === 'processing'
		) {
			return;
		}
		deleting = true;
		deleteError = '';
		try {
			await api.deleteImage(confirmDelete.id);
			if (!isCurrentView()) return;
			snackbar.notify('success', translate('queue.deleted'));
			confirmDelete = null;
			await queryClient.invalidateQueries({ queryKey: ['images'] });
		} catch (error) {
			if (!isCurrentView()) return;
			confirmDelete = null;
			const conflict = error instanceof ApiError && error.status === 409;
			deleteError = translate(conflict ? 'queue.deleteConflict' : 'queue.deleteFailed');
			if (conflict) await queryClient.invalidateQueries({ queryKey: ['images'] });
		} finally {
			if (isCurrentView()) deleting = false;
		}
	}
</script>

<svelte:head>
	<title>{$t('pages.queue.title')} – {$t('app.name')}</title>
</svelte:head>

<section class="mx-auto max-w-3xl">
	<div class="flex flex-wrap items-center justify-between gap-3">
		<h1 class="text-2xl font-semibold">{$t('pages.queue.title')}</h1>
		<button
			type="button"
			class="flex items-center gap-2 rounded-lg bg-action px-4 py-2.5 text-sm font-medium text-on-primary hover:opacity-90 disabled:opacity-50"
			disabled={analyzing || list.data?.length === 0}
			onclick={analyzeNow}
		>
			<Icon icon="refresh" />
			{#if analyzing}
				{$t('queue.analyzing')}
			{:else}
				{$t('queue.analyzeNow')}
			{/if}
		</button>
	</div>

	{#if deleteError}<p class="mt-3 text-sm text-error" role="alert">{deleteError}</p>{/if}

	{#if analyzeError}
		<p class="mt-3 flex items-center gap-2 text-sm text-error" role="alert">
			<Icon icon="alert" />
			{analyzeError}
		</p>
	{/if}

	{#if list.isLoading}
		<p class="mt-6 text-sm text-on-surface-variant">{$t('common.loading')}</p>
	{:else if list.error}
		<div
			class="mt-6 rounded-xl border border-outline-variant bg-surface-container-low p-4 text-sm text-on-surface-variant"
		>
			{$t('queue.loadError')}
		</div>
	{:else if list.data?.length === 0}
		<div class="mt-6 rounded-xl border border-dashed border-outline-variant p-8 text-center">
			<p class="text-sm font-medium">{$t('queue.emptyTitle')}</p>
			<p class="mt-1 text-sm text-on-surface-variant">{$t('queue.emptyBody')}</p>
		</div>
	{:else}
		<p class="mt-4 text-sm text-on-surface-variant">
			{$t('queue.shownJobs', { values: { count: list.data?.length ?? 0 } })}
		</p>
		<ul class="mt-4 space-y-3">
			{#each list.data as image (image.id)}
				<li
					class="flex items-center gap-3 rounded-xl border border-outline-variant bg-surface-container-low p-3"
				>
					<ReceiptImage
						imageId={image.id}
						thumbnailPath={image.thumbnail_path ?? null}
						alt=""
						previewLabel={`${$t('receipt.viewFullImage')}: ${image.original_filename ?? image.id}`}
						class="size-14 shrink-0 rounded-lg object-cover"
					/>
					<div class="min-w-0 flex-1">
						<p class="break-words text-sm font-medium">
							<a class="text-primary underline" href={resolve(`/upload?job=${image.id}`)}
								>{image.original_filename ?? `#${image.id}`}</a
							>
						</p>
						<p class="text-sm text-on-surface-variant">
							{#if image.created_at}
								{formatRelativeTime(image.created_at)} ·
							{/if}
							{#if image.observations_available && (image.status === 'pending' || image.status === 'processing')}
								<span class="text-warning"
									>{$t(
										image.status === 'processing'
											? 'submission.finishingSaved'
											: 'submission.savedObservations'
									)}</span
								>
							{:else if image.status === 'pending'}
								<span class="text-warning">{$t('queue.pending')}</span>
							{:else if image.status === 'failed'}
								<span class="text-error">{$t('queue.failed')}</span>
							{:else}
								<span>{$t(`submission.${image.status}`)}</span>
							{/if}
						</p>
						<p class="mt-1 text-sm text-on-surface-variant">
							{$t('submission.attempt', {
								values: { count: image.attempt_count, max: image.max_attempts }
							})}
						</p>
						{#if (image.status === 'pending' || image.status === 'processing') && image.retry_at}
							<p class="mt-1 text-sm text-on-surface-variant">
								{$t('submission.retryAt', {
									values: { time: new Date(image.retry_at).toLocaleString() }
								})}
							</p>
						{/if}
						{#if (image.status === 'pending' || image.status === 'processing') && image.wait_reason}<p
								class="mt-1 text-sm text-on-surface-variant"
							>
								{$t('submission.waitReason')}: {diagnosticLabel($t, 'reasons', image.wait_reason)}
								{#if image.status === 'pending' && image.wait_reason !== 'retry_backoff'}{$t(
										'diagnostics.readinessWait'
									)}{/if}
							</p>{/if}
						{#if image.result_code}
							<p class="mt-2 break-words text-sm text-on-surface-variant">
								{#if hasPreviousAttemptResult(image)}{`${$t('submission.previousAttempt')}: `}{/if}{diagnosticLabel(
									$t,
									'reasons',
									image.result_code
								)}
							</p>
						{/if}
						{#if image.error && image.error !== image.result_code}
							<p class="mt-1 break-words text-sm text-error">
								{image.error}
							</p>
						{/if}
						<a
							class="mt-3 inline-flex rounded-lg border border-outline px-3 py-2 text-sm font-medium text-primary hover:bg-surface-container-high"
							href={image.receipt_id
								? resolve(`/receipts/${image.receipt_id}`)
								: resolve(`/upload?job=${image.id}${image.draft_available ? '&draft=1' : ''}`)}
						>
							{$t(
								image.receipt_id
									? 'submission.openReceipt'
									: image.draft_available
										? 'submission.openDraft'
										: 'queue.openJob'
							)}
						</a>
					</div>
					<button
						type="button"
						class="shrink-0 rounded-lg p-2 text-on-surface-variant hover:bg-error-container hover:text-on-error-container"
						aria-label={$t('common.remove')}
						title={$t('common.remove')}
						onclick={() => {
							deleteError = '';
							confirmDelete = image;
						}}
						disabled={deleting || !!image.receipt_id || image.status === 'processing'}
					>
						<Icon icon="trash" />
					</button>
				</li>
			{/each}
		</ul>
	{/if}
</section>

<ConfirmDialog
	open={confirmDelete !== null}
	title={$t('queue.deleteTitle')}
	body={confirmDelete
		? translate('queue.deleteBody', {
				values: { file: confirmDelete.original_filename ?? `#${confirmDelete.id}` }
			})
		: ''}
	confirmLabel={$t('common.delete')}
	busy={deleting}
	confirmDisabled={!!currentDelete?.receipt_id || currentDelete?.status === 'processing'}
	onConfirm={removeImage}
	onCancel={() => (confirmDelete = null)}
/>
