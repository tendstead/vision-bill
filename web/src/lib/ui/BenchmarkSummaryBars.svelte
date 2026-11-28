<script lang="ts">
	import {
		formatMs,
		formatPercent,
		labelColumnWidth,
		linearScale,
		truncateMiddle,
		type Comparison,
		type Series
	} from '$lib/benchmark-compare';

	let { comparison, series }: { comparison: Comparison; series: Series[] } = $props();
	const uid = $props.id();
	const summary = (s: Series) =>
		comparison.runs
			.find((r) => r.run.id === s.runId)
			?.summaries.find((m) => m.model_id === s.modelId);
	const rows = $derived(series.map((s) => ({ s, m: summary(s) })));

	// One row per series, three panels share the labels.
	const W = 760,
		right = 8,
		gap = 28,
		rowH = 24,
		top = 24;
	const maxChars = 40;
	const left = $derived(
		labelColumnWidth(
			series.map((s) => truncateMiddle(s.label, maxChars)),
			6.4,
			120,
			300
		)
	);
	const panelW = $derived((W - left - right - gap * 2) / 3);
	const height = $derived(top + rows.length * rowH + 6);
	const maxLatency = $derived(Math.max(1, ...rows.map((r) => r.m?.average_latency_ms ?? 0)));
	const maxFailed = $derived(Math.max(1, ...rows.map((r) => r.m?.failed ?? 0)));
	const textRoom = 56;
	const panels = $derived([
		{
			title: 'Average score',
			max: 1,
			value: (m?: ReturnType<typeof summary>) => m?.average_score ?? null,
			text: (m?: ReturnType<typeof summary>) => formatPercent(m?.average_score)
		},
		{
			title: 'Average latency',
			max: maxLatency,
			value: (m?: ReturnType<typeof summary>) => m?.average_latency_ms ?? null,
			text: (m?: ReturnType<typeof summary>) => formatMs(m?.average_latency_ms)
		},
		{
			title: 'Failures',
			max: maxFailed,
			value: (m?: ReturnType<typeof summary>) => m?.failed ?? null,
			text: (m?: ReturnType<typeof summary>) => (m ? String(m.failed) : '—')
		}
	]);
	const x0 = (p: number) => left + p * (panelW + gap);
	const desc = $derived(
		rows
			.map(
				({ s, m }) =>
					`${s.label}: score ${formatPercent(m?.average_score)}, latency ${formatMs(m?.average_latency_ms)}, ${m?.failed ?? 0} failed`
			)
			.join('; ')
	);
</script>

<figure class="rounded-xl border border-outline-variant p-3" data-testid="chart-summary-bars">
	<figcaption id="{uid}-t" class="text-sm font-medium">Model summary</figcaption>
	<svg viewBox="0 0 {W} {height}" class="mt-2 w-full" role="img" aria-labelledby="{uid}-t {uid}-d">
		<desc id="{uid}-d">{desc}</desc>
		{#each panels as panel, p (panel.title)}
			<text x={x0(p)} y={14} font-size="12" font-weight="600" fill="currentColor"
				>{panel.title}</text
			>
		{/each}
		{#each rows as { s, m }, i (s.key)}
			{@const cy = top + i * rowH}
			<text x={left - 8} y={cy + 16} text-anchor="end" font-size="12" fill="currentColor"
				>{truncateMiddle(s.label, maxChars)}<title>{s.label}</title></text
			>
			{#each panels as panel, p (panel.title)}
				{@const v = panel.value(m)}
				{@const scale = linearScale(0, panel.max || 1, 0, panelW - textRoom)}
				{#if v !== null}
					<rect
						x={x0(p)}
						y={cy + 5}
						width={Math.max(1, scale(v))}
						height="14"
						rx="2"
						fill={s.color}
					/>
				{/if}
				<text
					x={x0(p) + (v === null ? 0 : Math.max(1, scale(v))) + 5}
					y={cy + 16}
					font-size="11"
					fill="currentColor">{panel.text(m)}</text
				>
			{/each}
		{/each}
	</svg>
</figure>
