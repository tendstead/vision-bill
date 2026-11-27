<script lang="ts">
	import { createQuery } from '@tanstack/svelte-query';
	import { resolve } from '$app/paths';
	import { api } from '$lib/api/client';
	import { queryClient } from '$lib/query/client';
	import { t } from '$lib/i18n';

	let { id }: { id: string } = $props();
	const job = createQuery(
		() => ({
			queryKey: ['image', id],
			queryFn: () => api.getImage(id),
			retry: false,
			refetchInterval: (query) => {
				const status = query.state.data?.status;
				return !status || status === 'pending' || status === 'processing' ? 2000 : false;
			}
		}),
		() => queryClient
	);
</script>

<div class="mt-4 rounded-xl border border-outline-variant bg-surface-container-low p-5">
	{#if job.error}
		<p role="alert">{$t('submission.loadError')}</p>
		<button type="button" class="mt-3 text-primary underline" onclick={() => job.refetch()}
			>{$t('submission.refresh')}</button
		>
	{:else if !job.data}
		<p role="status">{$t('common.loading')}</p>
	{:else}
		<p class="text-lg font-medium" role="status" aria-live="polite">
			{$t(`submission.${job.data.status}`)}
		</p>
		{#if job.data.status === 'pending' || job.data.status === 'processing'}
			<p class="mt-2 text-sm text-on-surface-variant">{$t('submission.returnLater')}</p>
		{:else if job.data.receipt_id}
			<a
				class="mt-4 inline-block rounded-lg bg-primary px-4 py-2 text-on-primary"
				href={resolve(`/receipts/${job.data.receipt_id}`)}>{$t('submission.openReceipt')}</a
			>
		{:else}
			<p class="mt-2 text-sm text-error">{job.data.error}</p>
			<p class="mt-2 text-sm">{$t('submission.tryAgain')}</p>
		{/if}
		<p class="mt-4 break-all text-xs text-on-surface-variant">{$t('submission.reference')}: {id}</p>
	{/if}
	<a class="mt-4 inline-block text-primary underline" href={resolve('/upload')}
		>{$t('submission.newUpload')}</a
	>
</div>
