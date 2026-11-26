<script lang="ts">
	import { createQuery } from '@tanstack/svelte-query';
	import { sessionViewGuard } from '$lib/session-view';
	import { resolve } from '$app/paths';
	import { t, translate } from '$lib/i18n';
	import { api } from '$lib/api/client';
	import { weeklySeries, type WeeklyPoint } from '$lib/receipt-stats';
	import { queryKeys } from '$lib/query/keys';
	import { queryClient } from '$lib/query/client';
	import { colorToStyle } from '$lib/collection-colors';
	import { formatMoney } from '$lib/ui/money';
	import { snackbar } from '$lib/ui/snackbar.svelte';
	import CollectionPicker from '$lib/ui/CollectionPicker.svelte';

	const isCurrentView = sessionViewGuard();

	const review = createQuery(
		() => ({
			queryKey: queryKeys.receipts({ status: ['unverified'], limit: 5 }),
			queryFn: () => api.listReceipts({ status: ['unverified'], limit: 5 })
		}),
		() => queryClient
	);

	const draftReview = createQuery(
		() => ({
			queryKey: queryKeys.images({ status: ['needs_review'], limit: 5 }),
			queryFn: () => api.listImages({ status: ['needs_review'], limit: 5 })
		}),
		() => queryClient
	);
	let reviewDrafts = $derived(
		(draftReview.data ?? []).filter((image) => !image.receipt_id && image.draft_available)
	);

	const statistics = createQuery(
		() => ({
			queryKey: queryKeys.statistics(12),
			queryFn: () => api.getStatistics(12)
		}),
		() => queryClient
	);

	let stats = $derived(statistics.data ?? null);
	let weeklyCharts = $derived.by(() =>
		(stats?.currencies ?? []).map((currency) => ({
			currency: currency.currency,
			points: weeklySeries(stats?.weekly_spending ?? [], currency.currency)
		}))
	);
	let merchantGroups = $derived.by(() =>
		(stats?.currencies ?? []).map((currency) => ({
			currency: currency.currency,
			merchants: (stats?.merchants ?? [])
				.filter((merchant) => merchant.currency === currency.currency)
				.slice(0, 5)
		}))
	);

	function maximum(points: WeeklyPoint[]): number {
		return Math.max(...points.map((point) => point.total), 1);
	}

	function twelveWeekTotal(points: WeeklyPoint[]): number {
		return points.reduce((total, point) => total + point.total, 0);
	}

	function weekLabel(value: string): string {
		return value.slice(5);
	}

	const active = createQuery(
		() => ({
			queryKey: queryKeys.collectionsActive(),
			queryFn: () => api.getActiveCollection()
		}),
		() => queryClient
	);
	let activeCollection = $derived(active.data ?? null);

	const collections = createQuery(
		() => ({ queryKey: queryKeys.collections(), queryFn: () => api.listCollections() }),
		() => queryClient
	);

	let stopping = $state(false);
	let activating = $state(false);

	async function invalidateCapture(): Promise<void> {
		if (!isCurrentView()) return;
		await queryClient.invalidateQueries({ queryKey: queryKeys.collectionsActive() });
		if (!isCurrentView()) return;
		await queryClient.invalidateQueries({ queryKey: queryKeys.collections() });
	}

	async function stopCapturing(): Promise<void> {
		if (!isCurrentView() || stopping || !activeCollection) return;
		stopping = true;
		try {
			await api.deactivateCollection(activeCollection.id);
			await invalidateCapture();
		} catch {
			if (isCurrentView()) snackbar.notify('error', translate('collections.deactivateFailed'));
		} finally {
			if (isCurrentView()) stopping = false;
		}
	}

	async function activate(collectionId: string): Promise<void> {
		if (!isCurrentView() || activating || !collectionId) return;
		activating = true;
		try {
			await api.activateCollection(collectionId);
			await invalidateCapture();
		} catch {
			if (isCurrentView()) snackbar.notify('error', translate('collections.activateFailed'));
		} finally {
			if (isCurrentView()) activating = false;
		}
	}
</script>

<svelte:head>
	<title>{$t('pages.dashboard.title')} – {$t('app.name')}</title>
</svelte:head>

<section class="mx-auto max-w-5xl">
	<div class="flex flex-wrap items-end justify-between gap-3">
		<div>
			<h1 class="text-2xl font-semibold">{$t('pages.dashboard.title')}</h1>
			<p class="mt-1 text-sm text-on-surface-variant">{$t('dashboard.verifiedOnly')}</p>
		</div>
		<a
			href={resolve('/statistics')}
			class="rounded-lg px-4 py-2 text-sm font-medium text-primary hover:bg-primary-container"
		>
			{$t('dashboard.openStatistics')}
		</a>
	</div>

	<div class="mt-6 rounded-xl border border-primary/50 bg-surface-container-low p-5">
		<h2 class="text-lg font-semibold">{$t('dashboard.openReview')}</h2>
		<div class="mt-4">
			<h3 class="text-sm font-medium">{$t('nav.receipts')}</h3>
			{#if review.isLoading}<p class="mt-2 text-sm">{$t('common.loading')}</p>{/if}
			{#if review.error}<p class="mt-2 text-sm text-error">{$t('common.error')}</p>{/if}
			{#if review.data?.length}<ul class="mt-1 divide-y divide-outline-variant">
					{#each review.data ?? [] as receipt (receipt.id)}
						<li>
							<a
								href={resolve(`/receipts/${receipt.id}`)}
								class="flex flex-wrap items-center justify-between gap-2 py-3 text-sm text-primary hover:underline"
								><span class="break-words font-medium"
									>{receipt.merchant_display_name ||
										receipt.merchant_name ||
										$t('receipts.unknownVendor')} · {receipt.date}</span
								><span class="font-semibold tabular-nums"
									>{formatMoney(receipt.total, receipt.currency)}</span
								></a
							>
						</li>
					{/each}
				</ul>{/if}
		</div>
		<div class="mt-4">
			<h3 class="text-sm font-medium">{$t('submission.openDraft')}</h3>
			{#if draftReview.isLoading}<p class="mt-2 text-sm">{$t('common.loading')}</p>{/if}
			{#if draftReview.error}<p class="mt-2 text-sm text-error">{$t('common.error')}</p>{/if}
			{#if reviewDrafts.length}<ul class="mt-1 divide-y divide-outline-variant">
					{#each reviewDrafts as image (image.id)}
						<li>
							<a
								href={resolve(`/upload?job=${image.id}&draft=1`)}
								class="block break-words py-3 text-sm font-medium text-primary hover:underline"
								>{$t('submission.openDraft')}: {image.original_filename ?? image.id}</a
							>
						</li>
					{/each}
				</ul>{/if}
		</div>
		{#if review.isSuccess && draftReview.isSuccess && !review.data?.length && !reviewDrafts.length}
			<p class="mt-3 text-sm text-on-surface-variant">{$t('dashboard.noOpenReview')}</p>
		{/if}
		<div class="mt-3 flex flex-wrap gap-4">
			<a href={resolve('/receipts')} class="text-sm font-medium text-primary hover:underline"
				>{$t('nav.receipts')}</a
			><a href={resolve('/queue')} class="text-sm font-medium text-primary hover:underline"
				>{$t('upload.openQueue')}</a
			>
		</div>
	</div>

	<!-- Active collection capture -->
	<div class="mt-6 rounded-xl border border-outline-variant bg-surface-container-low p-5">
		{#if active.isLoading}
			<p class="text-sm text-on-surface-variant">{$t('common.loading')}</p>
		{:else if active.error}
			<p class="text-sm text-on-surface-variant">{$t('common.error')}</p>
		{:else if activeCollection}
			<div class="flex flex-wrap items-center justify-between gap-3">
				<div class="flex min-w-0 items-center gap-2">
					<span class="size-3 shrink-0 rounded-full" style={colorToStyle(activeCollection.color)}
					></span>
					<p class="truncate text-sm">
						{$t('collections.capturing')}:
						<strong>{activeCollection.name}</strong>
					</p>
				</div>
				<button
					type="button"
					class="shrink-0 rounded-lg border border-outline-variant px-3 py-1.5 text-sm font-medium hover:bg-surface-container-high disabled:opacity-60"
					disabled={stopping}
					onclick={stopCapturing}
				>
					{$t('collections.stopCapturing')}
				</button>
			</div>
		{:else}
			<div>
				<p class="text-sm font-medium">{$t('collections.startCapturing')}</p>
				<div class="mt-2 max-w-xs">
					<CollectionPicker
						options={collections.data ?? []}
						mode="single"
						value=""
						placeholder={$t('collections.noneActive')}
						onchange={(v) => activate(v as string)}
					/>
				</div>
			</div>
		{/if}
	</div>

	{#if statistics.isLoading}
		<p class="mt-6 text-sm text-on-surface-variant">{$t('common.loading')}</p>
	{:else if statistics.error}
		<p
			class="mt-6 rounded-xl border border-outline-variant bg-surface-container-low p-4 text-sm text-on-surface-variant"
		>
			{$t('dashboard.loadError')}
		</p>
	{:else if stats?.verified_receipt_count === 0}
		<div class="mt-6 rounded-xl border border-dashed border-outline-variant p-8 text-center">
			<p class="font-medium">{$t('dashboard.emptyTitle')}</p>
			<p class="mt-1 text-sm text-on-surface-variant">{$t('dashboard.emptyBody')}</p>
		</div>
	{:else}
		<div class="mt-6 grid gap-4 sm:grid-cols-2">
			<div class="rounded-xl border border-outline-variant bg-surface-container-low p-5">
				<p class="text-sm text-on-surface-variant">{$t('dashboard.verifiedReceipts')}</p>
				<p class="mt-2 text-3xl font-semibold">{stats?.verified_receipt_count ?? 0}</p>
			</div>
			{#each weeklyCharts as chart (chart.currency)}
				<div class="rounded-xl border border-outline-variant bg-surface-container-low p-5">
					<p class="text-sm text-on-surface-variant">
						{$t('dashboard.twelveWeekTotal')} · {chart.currency}
					</p>
					<p class="mt-2 text-2xl font-semibold">
						{formatMoney(twelveWeekTotal(chart.points), chart.currency)}
					</p>
				</div>
			{/each}
		</div>

		<div class="mt-6 grid gap-6 lg:grid-cols-2">
			{#each weeklyCharts as chart (chart.currency)}
				<div class="rounded-xl border border-outline-variant bg-surface-container-low p-5">
					<h2 class="font-semibold">{$t('dashboard.weeklySpending')} · {chart.currency}</h2>
					<div
						class="mt-5 hidden h-40 items-end gap-1 sm:flex"
						role="img"
						aria-label={$t('dashboard.weeklySpending')}
					>
						{#each chart.points as point (point.week_start)}
							<div class="flex min-w-0 flex-1 flex-col items-center gap-1">
								<div class="flex h-32 w-full items-end">
									<div
										class="w-full rounded-t bg-primary"
										style={`height: ${(point.total / maximum(chart.points)) * 100}%`}
										title={`${point.week_start}: ${formatMoney(point.total, chart.currency)}`}
									></div>
								</div>
								<span class="text-xs text-on-surface-variant">{weekLabel(point.week_start)}</span>
							</div>
						{/each}
					</div>
					<details class="mt-4 text-sm" open>
						<summary class="cursor-pointer font-medium text-primary"
							>{$t('dashboard.chartData')}</summary
						>
						<table class="mt-3 w-full text-left text-sm">
							<caption class="sr-only">{$t('dashboard.weeklySpending')} · {chart.currency}</caption>
							<thead
								><tr class="border-b border-outline-variant"
									><th scope="col" class="py-2">{$t('dashboard.weekStarting')}</th><th
										scope="col"
										class="py-2 text-right">{$t('dashboard.total')}</th
									></tr
								></thead
							>
							<tbody
								>{#each chart.points as point (point.week_start)}<tr
										class="border-b border-outline-variant"
										><th scope="row" class="py-2 font-normal">{point.week_start}</th><td
											class="py-2 text-right tabular-nums"
											>{formatMoney(point.total, chart.currency)}</td
										></tr
									>{/each}</tbody
							>
						</table>
					</details>
				</div>
			{/each}
		</div>

		<div class="mt-6 grid gap-6 lg:grid-cols-2">
			{#each merchantGroups as group (group.currency)}
				<div class="rounded-xl border border-outline-variant bg-surface-container-low p-5">
					<h2 class="font-semibold">{$t('dashboard.topMerchants')} · {group.currency}</h2>
					<ul class="mt-4 space-y-3">
						{#each group.merchants as merchant (`${merchant.name}-${merchant.currency}`)}
							<li class="flex items-center justify-between gap-3 text-sm">
								<span class="min-w-0 truncate">{merchant.name} · {merchant.receipt_count}</span>
								<span class="shrink-0 font-medium"
									>{formatMoney(merchant.total, group.currency)}</span
								>
							</li>
						{/each}
					</ul>
				</div>
			{/each}
		</div>
	{/if}
</section>
