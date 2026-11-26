<script lang="ts">
	import { onDestroy } from 'svelte';
	import { createQuery } from '@tanstack/svelte-query';
	import { resolve } from '$app/paths';
	import { t } from '$lib/i18n';
	import { api } from '$lib/api/client';
	import { queryKeys } from '$lib/query/keys';
	import { queryClient } from '$lib/query/client';
	import { formatMoney } from '$lib/ui/money';
	import Icon from '$lib/ui/Icon.svelte';

	let search = $state('');
	let debounced = $state('');
	let timer: ReturnType<typeof setTimeout> | undefined;

	function onSearchInput(event: Event) {
		search = (event.currentTarget as HTMLInputElement).value;
		if (timer) clearTimeout(timer);
		timer = setTimeout(() => (debounced = search), 300);
	}

	onDestroy(() => clearTimeout(timer));

	let term = $derived(debounced.trim());

	const query = createQuery(
		() => ({
			queryKey: queryKeys.search(term),
			queryFn: () => api.searchProducts(term),
			enabled: term.length > 0
		}),
		() => queryClient
	);

	let purchases = $derived(query.data?.purchases ?? []);
	let currency = $derived(query.data?.currency ?? '');
</script>

<svelte:head>
	<title>{$t('pages.search.title')} – {$t('app.name')}</title>
</svelte:head>

<section class="mx-auto max-w-3xl">
	<h1 class="text-2xl font-semibold">{$t('pages.search.title')}</h1>

	<div class="mt-4 flex flex-col gap-3">
		<input
			type="search"
			aria-label={$t('search.placeholder')}
			placeholder={$t('search.placeholder')}
			class="flex-1 rounded-lg border border-outline-variant bg-surface-container-lowest px-3 py-2.5 text-sm"
			value={search}
			oninput={onSearchInput}
		/>
	</div>

	<p
		class="mt-4 text-sm text-on-surface-variant"
		role="status"
		aria-live="polite"
		aria-atomic="true"
	>
		{#if term && !query.isFetching && !query.error}{$t('search.resultCount', {
				values: { count: purchases.length }
			})}{/if}
	</p>

	{#if term === ''}
		<div class="mt-6 rounded-xl border border-dashed border-outline-variant p-8 text-center">
			<div
				class="mx-auto mb-3 flex size-12 items-center justify-center rounded-lg bg-surface-container"
			>
				<Icon icon="search" />
			</div>
			<p class="text-sm font-medium">{$t('search.emptyTitle')}</p>
			<p class="mt-1 text-sm text-on-surface-variant">{$t('search.emptyBody')}</p>
		</div>
	{:else if query.isLoading}
		<p class="mt-6 text-sm text-on-surface-variant">{$t('common.loading')}</p>
	{:else if query.error}
		<div
			class="mt-6 rounded-xl border border-outline-variant bg-surface-container-low p-4 text-sm text-on-surface-variant"
		>
			{$t('common.error')}
		</div>
	{:else if purchases.length === 0}
		<div class="mt-6 rounded-xl border border-dashed border-outline-variant p-8 text-center">
			<p class="text-sm text-on-surface-variant">
				{$t('search.noResults', { values: { query: term } })}
			</p>
		</div>
	{:else}
		{#if currency}
			<div class="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
				{#if query.data && query.data.latest_price !== null}
					<div class="rounded-xl border border-outline-variant bg-surface-container-low p-4">
						<p class="text-sm text-on-surface-variant">{$t('search.latestPrice')}</p>
						<p class="mt-1 text-xl font-semibold">
							{formatMoney(query.data.latest_price, currency)}
						</p>
					</div>
				{/if}
				{#if query.data && query.data.cheapest_price !== null}
					<div class="rounded-xl border border-outline-variant bg-surface-container-low p-4">
						<p class="text-sm text-on-surface-variant">{$t('search.cheapestPrice')}</p>
						<p class="mt-1 text-xl font-semibold">
							{formatMoney(query.data.cheapest_price, currency)}
						</p>
					</div>
				{/if}
				{#if query.data && query.data.average_price !== null}
					<div class="rounded-xl border border-outline-variant bg-surface-container-low p-4">
						<p class="text-sm text-on-surface-variant">{$t('search.averagePrice')}</p>
						<p class="mt-1 text-xl font-semibold">
							{formatMoney(query.data.average_price, currency)}
						</p>
					</div>
				{/if}
			</div>
		{/if}

		<ul class="mt-4 space-y-3" aria-label={$t('search.purchases')}>
			{#each purchases as purchase, i (i + ':' + purchase.receipt_id)}
				<li class="rounded-xl border border-outline-variant bg-surface-container-low p-4">
					<div class="flex flex-wrap items-start justify-between gap-3">
						<div class="min-w-0 flex-1 basis-40">
							<a
								href={resolve(`/receipts/${purchase.receipt_id}`)}
								class="break-words font-medium text-primary hover:underline"
								>{purchase.description}</a
							>
							<p class="mt-1 break-words text-sm text-on-surface-variant">
								{purchase.merchant_display_name ||
									purchase.merchant_name ||
									$t('receipts.unknownVendor')}
							</p>
							<p class="mt-1 text-sm text-on-surface-variant">{purchase.date}</p>
						</div>
						<dl class="grid max-w-full gap-2 text-right text-sm">
							<div>
								<dt class="text-on-surface-variant">{$t('search.tableUnitPrice')}</dt>
								<dd class="break-words text-lg font-semibold tabular-nums">
									{formatMoney(purchase.unit_price, purchase.currency) || '—'}
								</dd>
							</div>
							<div>
								<dt class="inline text-on-surface-variant">{$t('search.tableQty')}:</dt>
								<dd class="inline tabular-nums">{purchase.quantity ?? '—'}</dd>
							</div>
						</dl>
					</div>
				</li>
			{/each}
		</ul>
	{/if}
</section>
