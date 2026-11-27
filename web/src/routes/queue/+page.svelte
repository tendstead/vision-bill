<script lang="ts">
	import { createQuery } from '@tanstack/svelte-query';
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

	const statuses: ImageStatus[] = ['pending', 'processing', 'failed', 'timed_out', 'unreadable'];

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
	let analyzeError = $state('');

	async function analyzeNow() {
		if (analyzing) {
			return;
		}
		analyzing = true;
		analyzeError = '';
		try {
			const response = await api.analyzePending();
			if (response.results.length > 0) {
				const failed = response.results.filter((r) => r.status === 'failed').length;
				snackbar.notify(
					failed > 0 ? 'error' : 'success',
					translate('queue.analyzed', { values: { count: response.results.length } })
				);
			}
			await queryClient.invalidateQueries({ queryKey: ['images'] });
			await queryClient.invalidateQueries({ queryKey: ['receipts'] });
		} catch (error) {
			analyzeError =
				error instanceof ApiError && error.status === 503
					? translate('upload.errorDb')
					: translate('queue.analyzeFailed');
		} finally {
			analyzing = false;
		}
	}

	async function removeImage() {
		if (!confirmDelete || deleting) {
			return;
		}
		deleting = true;
		try {
			await api.deleteImage(confirmDelete.id);
			snackbar.notify('success', translate('queue.deleted'));
			confirmDelete = null;
			await queryClient.invalidateQueries({ queryKey: ['images'] });
		} catch (error) {
			snackbar.notify(
				'error',
				error instanceof ApiError && error.status === 409
					? translate('queue.deleteConflict')
					: translate('queue.deleteFailed')
			);
		} finally {
			deleting = false;
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
			class="flex items-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-on-primary hover:opacity-90 disabled:opacity-50"
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
		<ul class="mt-4 space-y-3">
			{#each list.data as image (image.id)}
				<li
					class="flex items-center gap-3 rounded-xl border border-outline-variant bg-surface-container-low p-3"
				>
					<ReceiptImage
						imageId={image.id}
						thumbnailPath={image.thumbnail_path ?? null}
						alt=""
						class="size-14 shrink-0 rounded-lg object-cover"
					/>
					<div class="min-w-0 flex-1">
						<p class="truncate text-sm font-medium">
							<a class="text-primary underline" href={resolve(`/upload?job=${image.id}`)}
								>{image.original_filename ?? `#${image.id}`}</a
							>
						</p>
						<p class="text-xs text-on-surface-variant">
							{#if image.created_at}
								{formatRelativeTime(image.created_at)} ·
							{/if}
							{#if image.status === 'pending'}
								<span class="text-warning">{$t('queue.pending')}</span>
							{:else if image.status === 'failed'}
								<span class="text-error">{$t('queue.failed')}</span>
							{:else}
								<span>{$t(`submission.${image.status}`)}</span>
							{/if}
						</p>
						{#if image.error}
							<p class="mt-1 truncate text-xs text-error" title={image.error}>
								{image.error}
							</p>
						{/if}
					</div>
					<button
						type="button"
						class="shrink-0 rounded-lg p-2 text-on-surface-variant hover:bg-error-container hover:text-on-error-container"
						aria-label={$t('common.remove')}
						title={$t('common.remove')}
						onclick={() => (confirmDelete = image)}
						disabled={image.status === 'processing'}
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
	onConfirm={removeImage}
	onCancel={() => (confirmDelete = null)}
/>
