<script lang="ts">
	import { resolve } from '$app/paths';
	import {
		cellFor,
		formatMs,
		formatPercent,
		formatPoints,
		linearScale,
		shortId,
		type CompareCell,
		type Comparison,
		type Series
	} from '$lib/benchmark-compare';
	import ChartMarker from './ChartMarker.svelte';

	let { comparison, series }: { comparison: Comparison; series: Series[] } = $props();
	const uid = $props.id();

	const W = 720,
		left = 96,
		right = 24,
		top = 28,
		bottom = 34;
	const rowH = $derived(Math.max(26, series.length * 6 + 12));
	const height = $derived(top + bottom + comparison.receipts.length * rowH);
	const x = linearScale(0, 1, left, W - right);
	const ticks = [0, 0.25, 0.5, 0.75, 1];

	let pinned = $state<{ receiptId: string; series: Series; cell: CompareCell } | null>(null);
	const dy = (i: number) => (i - (series.length - 1) / 2) * 6;
</script>

<figure class="rounded-xl border border-outline-variant p-3" data-testid="chart-score-dots">
	<figcaption id="{uid}-t" class="text-sm font-medium">Score per receipt</figcaption>
	<svg
		viewBox="0 0 {W} {height}"
		class="mt-2 w-full"
		role="group"
		aria-labelledby="{uid}-t {uid}-d"
		style="max-height: 80vh"
	>
		<desc id="{uid}-d"
			>Dot plot: one row per receipt, one marker per model and run, positioned by extraction score
			from 0 to 100 percent. Each marker links to its receipt. A table with the same values follows.</desc
		>
		{#each ticks as tick (tick)}
			<line
				x1={x(tick)}
				x2={x(tick)}
				y1={top - 6}
				y2={height - bottom + 4}
				stroke="currentColor"
				opacity="0.15"
			/>
			<text
				x={x(tick)}
				y={height - bottom + 18}
				text-anchor="middle"
				font-size="11"
				fill="currentColor">{tick * 100}%</text
			>
		{/each}
		{#each comparison.receipts as receipt, ri (receipt.receipt_id)}
			{@const cy = top + ri * rowH + rowH / 2}
			<line
				x1={left}
				x2={W - right}
				y1={cy}
				y2={cy}
				stroke="currentColor"
				opacity="0.08"
				stroke-dasharray="2 3"
			/>
			<text
				x={left - 8}
				y={cy + 4}
				text-anchor="end"
				font-size="11"
				font-family="monospace"
				fill="currentColor">{shortId(receipt.receipt_id)}</text
			>
			{#each series as s, si (s.key)}
				{@const cell = cellFor(receipt, s.runId, s.modelId)}
				{#if cell}
					<a
						href={resolve(`/receipts/${receipt.receipt_id}`)}
						aria-label="Receipt {shortId(receipt.receipt_id)}, {s.label}: {cell.success
							? formatPercent(cell.score)
							: 'failed'}"
						onpointerenter={() => (pinned = { receiptId: receipt.receipt_id, series: s, cell })}
						onfocus={() => (pinned = { receiptId: receipt.receipt_id, series: s, cell })}
					>
						<ChartMarker
							x={x(cell.success ? (cell.score ?? 0) : 0)}
							y={cy + dy(si)}
							shape={s.shape}
							color={s.color}
							hollow={!cell.success}
						/>
						<title
							>{shortId(receipt.receipt_id)} · {s.label}: {cell.success
								? formatPercent(cell.score)
								: `failed${cell.error ? ` (${cell.error})` : ''}`}</title
						>
					</a>
				{/if}
			{/each}
		{/each}
	</svg>
	<ul class="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs" aria-label="Legend">
		{#each series as s (s.key)}
			<li class="inline-flex items-center gap-1.5">
				<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"
					><ChartMarker x={7} y={7} r={4} shape={s.shape} color={s.color} /></svg
				>{s.label}
			</li>
		{/each}
		<li class="text-on-surface-variant">Hollow marker = failed</li>
	</ul>
	<p
		class="mt-2 min-h-5 text-xs text-on-surface-variant"
		aria-live="polite"
		data-testid="chart-pinned"
	>
		{#if pinned}
			Receipt {shortId(pinned.receiptId)} · {pinned.series.label}:
			{pinned.cell.success ? formatPercent(pinned.cell.score) : 'failed'}
			{#if pinned.cell.score_delta != null}
				({formatPoints(pinned.cell.score_delta)} vs. baseline){/if}
			· {formatMs(pinned.cell.latency_ms)} · {pinned.cell.attempts} attempt(s)
		{:else}Hover or focus a marker to pin its values.{/if}
	</p>
</figure>
