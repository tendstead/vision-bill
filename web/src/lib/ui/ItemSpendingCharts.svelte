<script lang="ts">
	import { locale, t } from '$lib/i18n';
	import { formatMoney } from '$lib/ui/money';
	import type { ItemSpendingStatistics } from '$lib/types';

	let { stats }: { stats: ItemSpendingStatistics } = $props();
	let selectedCurrency = $state('');
	let selectedCategory = $state('');

	let currencies = $derived([...new Set(stats.reconciliation.map((row) => row.currency))]);
	let currency = $derived(
		currencies.includes(selectedCurrency) ? selectedCurrency : (currencies[0] ?? '')
	);
	let categories = $derived(stats.categories.filter((row) => row.currency === currency));
	let category = $derived(
		categories.some((row) => row.category === selectedCategory)
			? selectedCategory
			: (categories[0]?.category ?? '')
	);
	let months = $derived(
		stats.monthly.filter((row) => row.currency === currency && row.category === category)
	);
	let maximum = $derived(Math.max(1, ...months.map((row) => Math.abs(Number(row.total)))));
	let reconciliation = $derived(stats.reconciliation.find((row) => row.currency === currency));

	function monthLabel(month: string): string {
		return new Intl.DateTimeFormat($locale ?? 'en', { month: 'short', year: 'numeric' }).format(
			new Date(`${month}T12:00:00`)
		);
	}

	function shareLabel(value: string | null): string {
		return value === null ? '—' : `${value}%`;
	}
</script>

<div class="rounded-xl border border-outline-variant bg-surface-container-low p-4 sm:p-5">
	<h2 class="text-lg font-semibold">{$t('statistics.itemTitle')}</h2>
	<p class="mt-1 text-sm text-on-surface-variant">{$t('statistics.itemIntro')}</p>
	{#if currencies.length > 1}
		<label class="mt-4 block text-sm" for="item-currency">{$t('editor.currency')}</label>
		<select
			id="item-currency"
			class="mt-1 rounded-lg border border-outline-variant bg-surface-container-lowest px-3 py-2"
			value={currency}
			onchange={(event) => (selectedCurrency = event.currentTarget.value)}
		>
			{#each currencies as code (code)}<option value={code}>{code}</option>{/each}
		</select>
	{/if}

	{#if categories.length === 0}
		<p class="mt-4 text-sm text-on-surface-variant">{$t('statistics.itemEmpty')}</p>
	{:else}
		<div class="mt-5 grid gap-6 lg:grid-cols-2">
			<div>
				<h3 class="font-medium">{$t('statistics.byCategory')} · {currency}</h3>
				<p class="text-xs text-on-surface-variant">{$t('statistics.share')}</p>
				<ul class="mt-2 divide-y divide-outline-variant">
					{#each categories as row (row.category)}
						<li class="flex items-center justify-between gap-2 py-2 text-sm">
							<span>{$t(`spendingCategories.${row.category}`)}</span>
							<span class="text-right tabular-nums"
								>{formatMoney(row.total, currency)}
								<span class="block text-xs text-on-surface-variant">{shareLabel(row.share)}</span
								></span
							>
						</li>
					{/each}
				</ul>
				{#if categories.every((row) => row.share === null)}
					<p class="mt-2 text-xs text-on-surface-variant">{$t('statistics.noShare')}</p>
				{/if}
			</div>
			<div>
				<h3 class="font-medium">{$t('statistics.categoryTrend')}</h3>
				<label class="mt-2 block text-xs text-on-surface-variant" for="item-trend-category"
					>{$t('statistics.selectCategory')}</label
				>
				<select
					id="item-trend-category"
					class="mt-1 w-full rounded-lg border border-outline-variant bg-surface-container-lowest px-3 py-2 text-sm"
					value={category}
					onchange={(event) => (selectedCategory = event.currentTarget.value)}
				>
					{#each categories as row (row.category)}<option value={row.category}
							>{$t(`spendingCategories.${row.category}`)}</option
						>{/each}
				</select>
				<table class="mt-3 w-full text-sm">
					<caption class="sr-only"
						>{$t('statistics.categoryTrend')} · {$t(`spendingCategories.${category}`)} · {currency}</caption
					>
					<thead class="sr-only"
						><tr
							><th scope="col">{$t('statistics.months')}</th><th scope="col"
								>{$t('statistics.itemSpend')}</th
							><th scope="col">{$t('statistics.share')}</th></tr
						></thead
					>
					<tbody>
						{#each months as row (row.month)}
							<tr class="border-b border-outline-variant">
								<th scope="row" class="py-2 pr-2 text-left font-normal whitespace-nowrap"
									>{monthLabel(
										row.month
									)}{#if row.month.slice(0, 7) === stats.date_to.slice(0, 7)}<span
											class="ml-1 text-xs text-on-surface-variant"
											>({$t('statistics.partial')})</span
										>{/if}</th
								>
								<td class="w-full py-2"
									><div
										class="h-3 rounded bg-surface-container-high"
										role="img"
										aria-label={formatMoney(row.total, currency)}
									>
										<div
											class="h-3 rounded {Number(row.total) < 0 ? 'bg-error' : 'bg-primary'}"
											style:width={`${Math.min(100, (Math.abs(Number(row.total)) / maximum) * 100)}%`}
										></div>
									</div></td
								>
								<td class="py-2 pl-2 text-right tabular-nums whitespace-nowrap"
									>{formatMoney(row.total, currency)}<span
										class="block text-xs text-on-surface-variant">{shareLabel(row.share)}</span
									></td
								>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</div>
	{/if}
	{#if reconciliation}
		<details class="mt-5 border-t border-outline-variant pt-3">
			<summary class="cursor-pointer text-sm font-medium"
				>{$t('statistics.itemReconciliation')} · {currency}</summary
			>
			<p class="mt-2 text-xs text-on-surface-variant">{$t('statistics.reconciliationHelp')}</p>
			<dl class="mt-2 grid grid-cols-[1fr_auto] gap-x-4 gap-y-1 text-sm">
				<dt>{$t('statistics.receiptTotal')}</dt>
				<dd class="text-right tabular-nums">
					{formatMoney(reconciliation.receipt_total, currency)}
				</dd>
				<dt>{$t('statistics.itemTotal')}</dt>
				<dd class="text-right tabular-nums">{formatMoney(reconciliation.item_total, currency)}</dd>
				<dt>{$t('statistics.depositTotal')}</dt>
				<dd class="text-right tabular-nums">
					{formatMoney(reconciliation.deposit_total, currency)}
				</dd>
				<dt>{$t('statistics.unallocatedTotal')}</dt>
				<dd class="text-right tabular-nums">
					{formatMoney(reconciliation.unallocated_total, currency)}
				</dd>
			</dl>
		</details>
	{/if}
</div>
