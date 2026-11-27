<script lang="ts">
	import { createQuery } from '@tanstack/svelte-query';
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
		await queryClient.invalidateQueries({ queryKey: queryKeys.collectionsActive() });
		await queryClient.invalidateQueries({ queryKey: queryKeys.collections() });
	}

	async function stopCapturing(): Promise<void> {
		if (stopping || !activeCollection) return;
		stopping = true;
		try {
			await api.deactivateCollection(activeCollection.id);
			await invalidateCapture();
		} catch {
			snackbar.notify('error', translate('collections.deactivateFailed'));
		} finally {
			stopping = false;
		}
	}

	async function activate(collectionId: string): Promise<void> {
		if (activating || !collectionId) return;
		activating = true;
		try {
			await api.activateCollection(collectionId);
			await invalidateCapture();
		} catch {
			snackbar.notify('error', translate('collections.activateFailed'));
		} finally {
			activating = false;
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
						class="mt-5 flex h-40 items-end gap-1"
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
								<span class="text-[10px] text-on-surface-variant"
									>{weekLabel(point.week_start)}</span
								>
							</div>
						{/each}
					</div>
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
									>{formatMoney(Number(merchant.total), group.currency)}</span
								>
							</li>
						{/each}
					</ul>
				</div>
			{/each}
		</div>
	{/if}
</section>
