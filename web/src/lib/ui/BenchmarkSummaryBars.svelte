<script lang="ts">
	import { formatMs, formatPercent, type Comparison, type Series } from '$lib/benchmark-compare';
	import BenchmarkBarChart from './BenchmarkBarChart.svelte';

	let { comparison, series }: { comparison: Comparison; series: Series[] } = $props();
	const summary = (s: Series) =>
		comparison.runs
			.find((r) => r.run.id === s.runId)
			?.summaries.find((m) => m.model_id === s.modelId);
	const rows = $derived(series.map((s) => ({ s, m: summary(s) })));
	const maxLatency = $derived(Math.max(1, ...rows.map((r) => r.m?.average_latency_ms ?? 0)));
	const maxFailed = $derived(Math.max(1, ...rows.map((r) => r.m?.failed ?? 0)));
</script>

<div class="rounded-xl border border-outline-variant p-3" data-testid="chart-summary-bars">
	<h3 class="text-sm font-medium">Model summary</h3>
	<div class="mt-2 grid gap-4 lg:grid-cols-3">
		<BenchmarkBarChart
			title="Average score"
			max={1}
			bars={rows.map(({ s, m }) => ({
				key: s.key,
				label: s.label,
				color: s.color,
				value: m?.average_score ?? null,
				text: formatPercent(m?.average_score)
			}))}
		/>
		<BenchmarkBarChart
			title="Average latency"
			max={maxLatency}
			bars={rows.map(({ s, m }) => ({
				key: s.key,
				label: s.label,
				color: s.color,
				value: m?.average_latency_ms ?? null,
				text: formatMs(m?.average_latency_ms)
			}))}
		/>
		<BenchmarkBarChart
			title="Failures"
			max={maxFailed}
			bars={rows.map(({ s, m }) => ({
				key: s.key,
				label: s.label,
				color: s.color,
				value: m?.failed ?? null,
				text: m ? String(m.failed) : '—'
			}))}
		/>
	</div>
</div>
