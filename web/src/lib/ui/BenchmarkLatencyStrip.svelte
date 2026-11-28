<script lang="ts">
	import { resolve } from '$app/paths';
	import {
		cellFor,
		formatMs,
		linearScale,
		niceTicks,
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
	const max = $derived(
		Math.max(1, ...comparison.receipts.flatMap((r) => r.cells.map((c) => c.latency_ms ?? 0)))
	);
	const ticks = $derived(niceTicks(max, 5));
	const x = $derived(linearScale(0, ticks[ticks.length - 1], left, W - right));
	const dy = (i: number) => (i - (series.length - 1) / 2) * 6;
	const hasTelemetry = $derived(
		comparison.receipts.some((r) =>
			r.cells.some((c) => c.load_ms != null || c.generation_ms != null)
		)
	);
	function tip(receiptId: string, label: string, cell: CompareCell) {
		const parts: string[] = [];
		if (cell.load_ms != null) parts.push(`load ${formatMs(cell.load_ms)}`);
		if (cell.generation_ms != null) parts.push(`generation ${formatMs(cell.generation_ms)}`);
		return `${shortId(receiptId)} · ${label}: ${formatMs(cell.latency_ms)}${parts.length ? ` (${parts.join(', ')})` : ''}`;
	}
	// A slow outlier is a dot well beyond the typical (median) latency.
	const median = $derived.by(() => {
		const all = comparison.receipts
			.flatMap((r) => r.cells.map((c) => c.latency_ms))
			.filter((v): v is number => v != null)
			.sort((a, b) => a - b);
		return all.length ? all[Math.floor(all.length / 2)] : 0;
	});
</script>

<figure class="rounded-xl border border-outline-variant p-3" data-testid="chart-latency">
	<figcaption id="{uid}-t" class="text-sm font-medium">Latency per receipt</figcaption>
	<svg
		viewBox="0 0 {W} {height}"
		class="mt-2 w-full"
		role="group"
		aria-labelledby="{uid}-t {uid}-d"
	>
		<desc id="{uid}-d"
			>Strip plot: one row per receipt, one marker per model and run, positioned by total latency.
			Where the server reports it, a pale bar shows model load time and a coloured bar the prompt
			and generation time. Markers link to their receipt.</desc
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
				fill="currentColor">{formatMs(tick)}</text
			>
		{/each}
		{#each comparison.receipts as receipt, ri (receipt.receipt_id)}
			{@const cy = top + ri * rowH + rowH / 2}
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
				{#if cell?.latency_ms != null}
					{@const y = cy + dy(si)}
					{#if cell.load_ms != null}
						<line
							x1={x(0)}
							x2={x(cell.load_ms)}
							y1={y}
							y2={y}
							stroke="currentColor"
							opacity="0.3"
							stroke-width="3"
						/>
					{/if}
					{#if cell.generation_ms != null}
						<line
							x1={x(cell.load_ms ?? 0)}
							x2={x((cell.load_ms ?? 0) + cell.generation_ms)}
							y1={y}
							y2={y}
							stroke={s.color}
							opacity="0.5"
							stroke-width="3"
						/>
					{/if}
					<a
						href={resolve(`/receipts/${receipt.receipt_id}`)}
						aria-label="Receipt {shortId(receipt.receipt_id)}, {s.label}: {formatMs(
							cell.latency_ms
						)}{cell.latency_ms > 3 * median && median > 0 ? ', slow outlier' : ''}"
					>
						<ChartMarker
							x={x(cell.latency_ms)}
							{y}
							shape={s.shape}
							color={s.color}
							hollow={!cell.success}
						/>
						<title>{tip(receipt.receipt_id, s.label, cell)}</title> >
					</a>
				{/if}
			{/each}
		{/each}
	</svg>
	<p class="mt-1 text-xs text-on-surface-variant">
		{hasTelemetry
			? 'Pale bar: model load time. Coloured bar: prompt evaluation and generation.'
			: 'The inference server reports no load/generation split for these runs.'}
		Markers follow the legend of the score chart.
	</p>
</figure>
