<script lang="ts">
	import { linearScale } from '$lib/benchmark-compare';

	export type Bar = {
		key: string;
		label: string;
		color: string;
		value: number | null;
		text: string;
	};
	let { title, bars, max }: { title: string; bars: Bar[]; max: number } = $props();
	const uid = $props.id();
	const W = 300,
		left = 110,
		right = 56,
		rowH = 24;
	const height = $derived(bars.length * rowH + 8);
	const x = $derived(linearScale(0, max || 1, left, W - right));
	const clip = (s: string) => (s.length > 18 ? `${s.slice(0, 17)}…` : s);
</script>

<figure>
	<figcaption id="{uid}-t" class="text-xs font-medium">{title}</figcaption>
	<svg viewBox="0 0 {W} {height}" class="mt-1 w-full" role="img" aria-labelledby="{uid}-t {uid}-d">
		<desc id="{uid}-d">{bars.map((b) => `${b.label}: ${b.text}`).join('; ')}</desc>
		{#each bars as bar, i (bar.key)}
			<text x={left - 6} y={4 + i * rowH + 16} text-anchor="end" font-size="12" fill="currentColor"
				>{clip(bar.label)}<title>{bar.label}</title></text
			>
			{#if bar.value !== null}
				<rect
					x={left}
					y={4 + i * rowH + 5}
					width={Math.max(1, x(bar.value) - left)}
					height="14"
					rx="2"
					fill={bar.color}
				/>
			{/if}
			<text
				x={(bar.value === null ? left : x(bar.value)) + 5}
				y={4 + i * rowH + 16}
				font-size="12"
				fill="currentColor">{bar.text}</text
			>
		{/each}
	</svg>
</figure>
