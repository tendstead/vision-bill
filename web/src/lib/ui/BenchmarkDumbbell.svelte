<script lang="ts">
	import { resolve } from '$app/paths';
	import {
		deltaTone,
		dumbbellRows,
		formatPercent,
		formatPoints,
		labelColumnWidth,
		linearScale,
		truncateMiddle,
		shortId,
		type Comparison,
		type Series
	} from '$lib/benchmark-compare';
	import ChartMarker from './ChartMarker.svelte';
	import { previewProps } from './receipt-preview.svelte';

	let { comparison, series }: { comparison: Comparison; series: Series[] } = $props();
	const uid = $props.id();
	const rows = $derived(dumbbellRows(comparison));
	const info = $derived(new Map(comparison.receipts.map((r) => [r.receipt_id, r])));
	const multiModel = $derived(new Set(rows.map((r) => r.modelId)).size > 1);
	const multiRun = $derived(new Set(rows.map((r) => r.runId)).size > 1);
	const maxChars = 40;
	const W = 720,
		right = 70,
		top = 26,
		bottom = 32,
		rowH = 24;
	const left = $derived(
		labelColumnWidth(
			rows.map((r) => truncateMiddle(rowLabel(r), maxChars)),
			6.6,
			110,
			300
		)
	);
	const height = $derived(top + bottom + rows.length * rowH);
	const x = $derived(linearScale(0, 1, left, W - right));
	const ticks = [0, 0.25, 0.5, 0.75, 1];
	const colors = { good: '#1a7f37', bad: '#cf222e', neutral: '#6e7781' };
	const labelOf = (key: string) => series.find((s) => s.key === key);
	const rowLabel = (r: (typeof rows)[number]) => {
		const s = labelOf(r.seriesKey);
		const parts = [shortId(r.receiptId)];
		if (multiModel) parts.push(r.modelId);
		if (multiRun && s) parts.push(s.label.replace(`${r.modelId} · `, ''));
		return parts.join(' · ');
	};
</script>

<figure class="rounded-xl border border-outline-variant p-3" data-testid="chart-dumbbell">
	<figcaption id="{uid}-t" class="text-sm font-medium">Run-to-run difference</figcaption>
	{#if rows.length === 0}
		<p class="mt-2 text-sm text-on-surface-variant">
			No receipt has a score in both a run and its baseline yet.
		</p>
	{:else}
		<svg
			viewBox="0 0 {W} {height}"
			class="mt-2 w-full"
			role="group"
			aria-labelledby="{uid}-t {uid}-d"
		>
			<desc id="{uid}-d"
				>Dumbbell chart: per receipt and model the score in the baseline run (hollow marker) and in
				the later run (filled marker). Green lines are improvements, red lines are regressions. Each
				filled marker links to its receipt.</desc
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
			{#each rows as r, i (`${r.receiptId}:${r.seriesKey}`)}
				{@const cy = top + i * rowH + rowH / 2}
				{@const tone = deltaTone(r.delta, true)}
				<svelte:element
					this={info.get(r.receiptId)?.exists ? 'a' : 'g'}
					href={info.get(r.receiptId)?.exists ? resolve(`/receipts/${r.receiptId}`) : undefined}
					tabindex={info.get(r.receiptId)?.exists ? -1 : undefined}
					{...previewProps(info.get(r.receiptId)?.image_id)}
				>
					<text
						x={left - 8}
						y={cy + 4}
						text-anchor="end"
						font-size="11"
						font-family="monospace"
						text-decoration="underline"
						fill="currentColor"
						>{truncateMiddle(rowLabel(r), maxChars)}<title>{rowLabel(r)}</title></text
					>
				</svelte:element>
				<line x1={x(r.from)} x2={x(r.to)} y1={cy} y2={cy} stroke={colors[tone]} stroke-width="3" />
				<ChartMarker x={x(r.from)} y={cy} r={4} color={colors[tone]} hollow />
				<svelte:element
					this={info.get(r.receiptId)?.exists ? 'a' : 'g'}
					href={info.get(r.receiptId)?.exists ? resolve(`/receipts/${r.receiptId}`) : undefined}
					{...previewProps(info.get(r.receiptId)?.image_id)}
					aria-label="Receipt {shortId(r.receiptId)}, {r.modelId}: {formatPercent(
						r.from
					)} to {formatPercent(r.to)}"
				>
					<ChartMarker x={x(r.to)} y={cy} r={5} color={colors[tone]} />
					<title
						>{rowLabel(r)}: {formatPercent(r.from)} → {formatPercent(r.to)} ({formatPoints(
							r.delta
						)})</title
					>
				</svelte:element>
				<text x={W - right + 8} y={cy + 4} font-size="11" fill={colors[tone]}
					>{formatPoints(r.delta)}</text
				>
			{/each}
		</svg>
	{/if}
</figure>
