<script lang="ts">
	import { createQuery } from '@tanstack/svelte-query';
	import { resolve } from '$app/paths';
	import { t, translate } from '$lib/i18n';
	import { api, ApiError } from '$lib/api/client';
	import { fetchAllReceipts } from '$lib/api/receipts';
	import { queryKeys } from '$lib/query/keys';
	import { queryClient } from '$lib/query/client';
	import { formatMoney } from '$lib/ui/money';
	import { formatRelativeTime } from '$lib/ui/time';
	import { goto } from '$app/navigation';
	import { snackbar } from '$lib/ui/snackbar.svelte';
	import ConfirmDialog from '$lib/ui/ConfirmDialog.svelte';
	import Icon from '$lib/ui/Icon.svelte';
	import ReceiptImage from '$lib/ui/ReceiptImage.svelte';
	import type { ReceiptRow, ReceiptStatus } from '$lib/types';

	type StatusFilter = '' | ReceiptStatus;

	let search = $state('');
	let statusFilter = $state<StatusFilter>('');
	let debounced = $state('');
	let timer: ReturnType<typeof setTimeout> | undefined;
	let confirmDelete: ReceiptRow | null = $state(null);
	let deleting = $state(false);

	function onSearchInput(event: Event) {
		search = (event.currentTarget as HTMLInputElement).value;
		if (timer) clearTimeout(timer);
		timer = setTimeout(() => (debounced = search), 300);
	}

	async function removeReceipt() {
		if (!confirmDelete || deleting) {
			return;
		}
		deleting = true;
		try {
			await api.deleteReceipt(confirmDelete.id);
			snackbar.notify('success', translate('receipt.deleted'));
			confirmDelete = null;
			await queryClient.invalidateQueries({ queryKey: ['receipts'] });
			await queryClient.invalidateQueries({ queryKey: ['images'] });
		} catch (error) {
			snackbar.notify(
				'error',
				error instanceof ApiError && error.status === 409
					? translate('receipt.deleteBlocked')
					: translate('receipt.deleteFailed')
			);
		} finally {
			deleting = false;
		}
	}

	const list = createQuery(
		() => {
			const filters = {
				search: debounced.trim() || undefined,
				status: statusFilter ? [statusFilter] : undefined
			};
			return { queryKey: queryKeys.receipts(filters), queryFn: () => fetchAllReceipts(filters) };
		},
		() => queryClient
	);

	const FILTERS: [StatusFilter, string][] = [
		['', 'receipts.filterAll'],
		['unverified', 'receipt.unverified'],
		['verified', 'receipt.verified']
	];
</script>

<svelte:head>
	<title>{$t('pages.receipts.title')} – {$t('app.name')}</title>
</svelte:head>

<section class="mx-auto max-w-3xl">
	<h1 class="text-2xl font-semibold">{$t('pages.receipts.title')}</h1>

	<div class="mt-4 flex flex-col gap-3">
		<input
			type="search"
			aria-label={$t('receipts.searchPlaceholder')}
			placeholder={$t('receipts.searchPlaceholder')}
			class="flex-1 rounded-lg border border-outline-variant bg-surface-container-lowest px-3 py-2.5 text-sm"
			value={search}
			oninput={onSearchInput}
		/>
		<div class="flex flex-wrap gap-2" role="group" aria-label={$t('receipts.filter')}>
			{#each FILTERS as [value, labelKey] (value)}
				<button
					type="button"
					aria-pressed={statusFilter === value}
					onclick={() => (statusFilter = value)}
					class="rounded-full border px-3 py-1.5 text-sm font-medium transition-colors
						{value === ''
						? statusFilter === value
							? 'border-outline bg-surface-container-highest text-on-surface'
							: 'border-outline-variant text-on-surface-variant hover:bg-surface-container'
						: value === 'verified'
							? statusFilter === value
								? 'border-primary bg-primary-container text-on-primary-container'
								: 'border-primary/50 text-primary hover:bg-primary-container'
							: statusFilter === value
								? 'border-error bg-error-container text-on-error-container'
								: 'border-error/50 text-error hover:bg-error-container'}"
				>
					{$t(labelKey)}
				</button>
			{/each}
		</div>
	</div>

	{#if list.isLoading}
		<p class="mt-6 text-sm text-on-surface-variant">{$t('common.loading')}</p>
	{:else if list.error}
		<div
			class="mt-6 rounded-xl border border-outline-variant bg-surface-container-low p-4 text-sm text-on-surface-variant"
		>
			{$t('common.error')}
		</div>
	{:else if list.data?.length === 0}
		<div class="mt-6 rounded-xl border border-dashed border-outline-variant p-8 text-center">
			<p class="text-sm text-on-surface-variant">
				{debounced.trim() || statusFilter ? $t('receipts.noResults') : $t('receipts.empty')}
			</p>
		</div>
	{:else}
		<ul class="mt-4 space-y-3">
			{#each list.data as receipt (receipt.id)}
				<li class="flex items-center gap-3">
					<div
						class="flex min-w-0 flex-1 cursor-pointer items-center gap-3 rounded-xl border border-outline-variant bg-surface-container-low p-4 transition-colors hover:border-primary/50"
						role="link"
						tabindex="0"
						onclick={() => goto(resolve(`/receipts/${receipt.id}`))}
						onkeydown={(e) => {
							if (e.key === 'Enter' || e.key === ' ') {
								e.preventDefault();
								goto(resolve(`/receipts/${receipt.id}`));
							}
						}}
					>
						<ReceiptImage
							imageId={receipt.image_id}
							thumbnailPath={receipt.thumbnail_path ?? null}
							alt={receipt.merchant_name || $t('receipts.unknownVendor')}
							class="size-12 shrink-0 rounded-lg object-cover"
						/>
						<div class="min-w-0 flex-1">
							<p class="truncate text-sm font-medium">
								{receipt.merchant_name || $t('receipts.unknownVendor')}
							</p>
							<p class="text-xs text-on-surface-variant">
								{formatMoney(receipt.total, receipt.currency)}
								{#if receipt.date}
									· {receipt.date}
								{/if}
								{#if receipt.created_at}
									· {formatRelativeTime(receipt.created_at)}
								{/if}
							</p>
						</div>
						<span
							class="shrink-0 rounded-full px-2.5 py-1 text-xs font-medium
								{receipt.status === 'verified'
								? 'bg-primary-container text-on-primary-container'
								: 'bg-surface-container-highest text-on-surface-variant'}"
						>
							{$t(receipt.status === 'verified' ? 'receipt.verified' : 'receipt.unverified')}
						</span>
					</div>
					<button
						type="button"
						class="shrink-0 rounded-lg p-2 text-on-surface-variant hover:bg-error-container hover:text-on-error-container"
						aria-label={$t('common.remove')}
						title={$t('common.remove')}
						onclick={() => (confirmDelete = receipt)}
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
	title={$t('receipt.deleteTitle')}
	body={confirmDelete
		? translate('receipt.deleteBody', {
				values: { merchant: confirmDelete.merchant_name || $t('receipts.unknownVendor') }
			})
		: ''}
	confirmLabel={$t('common.delete')}
	busy={deleting}
	onConfirm={removeReceipt}
	onCancel={() => (confirmDelete = null)}
/>
