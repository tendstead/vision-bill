<script lang="ts">
	import { t } from '$lib/i18n';
	import { weeklySeries, type WeeklyPoint } from '$lib/receipt-stats';
	import { formatMoney } from '$lib/ui/money';
	import type { ReceiptStatistics } from '$lib/types';

	let { stats }: { stats: ReceiptStatistics } = $props();

	let weeklyCharts = $derived.by(() =>
		stats.currencies.map((currency) => ({
			currency: currency.currency,
			points: weeklySeries(stats.weekly_spending, currency.currency)
		}))
	);

	function maximum(points: WeeklyPoint[]): number {
		return Math.max(...points.map((point) => point.total), 1);
	}

	function weekLabel(value: string): string {
		return value.slice(5);
	}
</script>

<div class="grid gap-4 sm:grid-cols-2">
	<div class="rounded-xl border border-outline-variant bg-surface-container-low p-5">
		<p class="text-sm text-on-surface-variant">{$t('statistics.receiptCount')}</p>
		<p class="mt-2 text-3xl font-semibold">{stats.verified_receipt_count}</p>
	</div>
	{#each stats.currencies as total (total.currency)}
		<div class="rounded-xl border border-outline-variant bg-surface-container-low p-5">
			<p class="text-sm text-on-surface-variant">
				{$t('statistics.totalSpend')} · {total.currency}
			</p>
			<p class="mt-2 text-2xl font-semibold">
				{formatMoney(Number(total.total), total.currency)}
			</p>
			<p class="mt-1 text-sm text-on-surface-variant">
				{$t('statistics.average')}: {formatMoney(Number(total.average), total.currency)}
			</p>
		</div>
	{/each}
</div>

<div class="mt-6 grid gap-6 lg:grid-cols-2">
	{#each stats.currencies as total (total.currency)}
		<div class="rounded-xl border border-outline-variant bg-surface-container-low p-5">
			<h2 class="font-semibold">{total.currency} · {$t('statistics.distribution')}</h2>
			<div class="mt-4 grid grid-cols-2 gap-3 text-sm">
				<div>
					<span class="text-on-surface-variant">{$t('statistics.median')}</span>
					<p class="font-medium">{formatMoney(Number(total.median), total.currency)}</p>
				</div>
				<div>
					<span class="text-on-surface-variant">{$t('statistics.minimum')}</span>
					<p class="font-medium">{formatMoney(Number(total.minimum), total.currency)}</p>
				</div>
				<div>
					<span class="text-on-surface-variant">{$t('statistics.maximum')}</span>
					<p class="font-medium">{formatMoney(Number(total.maximum), total.currency)}</p>
				</div>
				<div>
					<span class="text-on-surface-variant">{$t('statistics.subtotal')}</span>
					<p class="font-medium">{formatMoney(Number(total.subtotal), total.currency)}</p>
				</div>
				<div>
					<span class="text-on-surface-variant">{$t('statistics.discounts')}</span>
					<p class="font-medium">{formatMoney(Number(total.discounts), total.currency)}</p>
				</div>
				<div>
					<span class="text-on-surface-variant">{$t('statistics.taxes')}</span>
					<p class="font-medium">{formatMoney(Number(total.taxes), total.currency)}</p>
				</div>
				<div>
					<span class="text-on-surface-variant">{$t('statistics.tips')}</span>
					<p class="font-medium">{formatMoney(Number(total.tips), total.currency)}</p>
				</div>
			</div>
		</div>
	{/each}
</div>

<div class="mt-6 grid gap-6 lg:grid-cols-2">
	{#each [{ title: $t('statistics.byMerchant'), rows: stats.merchants }, { title: $t('statistics.byCategory'), rows: stats.categories }, { title: $t('statistics.byPaymentMethod'), rows: stats.payment_methods }] as group (group.title)}
		<div class="rounded-xl border border-outline-variant bg-surface-container-low p-5">
			<h2 class="font-semibold">{group.title}</h2>
			<ul class="mt-4 space-y-3">
				{#each group.rows.slice(0, 10) as item (`${group.title}-${item.name}-${item.currency}`)}
					<li class="flex items-center justify-between gap-3 text-sm">
						<span class="min-w-0 truncate">{item.name} · {item.receipt_count}</span>
						<span class="shrink-0 text-right font-medium">
							{formatMoney(Number(item.total), item.currency)}
							<span class="block text-xs font-normal text-on-surface-variant"
								>{$t('statistics.average')}: {formatMoney(
									Number(item.average),
									item.currency
								)}</span
							>
						</span>
					</li>
				{/each}
			</ul>
		</div>
	{/each}
</div>

<div class="mt-6 grid gap-6 lg:grid-cols-2">
	{#each weeklyCharts as chart (chart.currency)}
		<div class="rounded-xl border border-outline-variant bg-surface-container-low p-5">
			<h2 class="font-semibold">{$t('statistics.weeklySpending')} · {chart.currency}</h2>
			<div
				class="mt-5 flex h-40 items-end gap-1"
				role="img"
				aria-label={$t('statistics.weeklySpending')}
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
						<span class="text-[10px] text-on-surface-variant">{weekLabel(point.week_start)}</span>
					</div>
				{/each}
			</div>
		</div>
	{/each}
</div>

<div class="mt-6 rounded-xl border border-outline-variant bg-surface-container-low p-5">
	<h2 class="font-semibold">{$t('statistics.byWeekday')}</h2>
	<div class="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
		{#each stats.weekdays as item (`${item.weekday}-${item.currency}`)}
			<div class="rounded-lg bg-surface-container p-3 text-sm">
				<p class="font-medium">{$t(`weekdays.${item.weekday}`)} · {item.currency}</p>
				<p class="mt-1 text-on-surface-variant">
					{formatMoney(Number(item.total), item.currency)} · {item.receipt_count}
				</p>
			</div>
		{/each}
	</div>
</div>
