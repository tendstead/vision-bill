<script lang="ts">
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

		<div class="mt-6 overflow-x-auto rounded-xl border border-outline-variant">
			<table class="w-full text-left text-sm">
				<thead class="bg-surface-container text-xs text-on-surface-variant">
					<tr>
						<th class="p-3">{$t('search.tableDate')}</th>
						<th class="p-3">{$t('search.tableStore')}</th>
						<th class="p-3">{$t('search.tableItem')}</th>
						<th class="p-3">{$t('search.tableQty')}</th>
						<th class="p-3">{$t('search.tableUnitPrice')}</th>
					</tr>
				</thead>
				<tbody>
					{#each purchases as purchase, i (i + ':' + purchase.receipt_id)}
						<tr class="border-t border-outline-variant">
							<td class="whitespace-nowrap p-3">{purchase.date}</td>
							<td class="p-3">
								{purchase.merchant_name || $t('receipts.unknownVendor')}
							</td>
							<td class="min-w-40 p-3">
								<a
									href={resolve(`/receipts/${purchase.receipt_id}`)}
									class="font-medium text-primary hover:underline"
								>
									{purchase.description}
								</a>
							</td>
							<td class="p-3">{purchase.quantity}</td>
							<td class="whitespace-nowrap p-3">
								{formatMoney(purchase.unit_price, purchase.currency)}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</section>
