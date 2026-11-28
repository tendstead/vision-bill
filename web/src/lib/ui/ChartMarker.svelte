<script lang="ts">
	import type { Shape } from '$lib/benchmark-compare';
	let {
		x,
		y,
		shape = 'circle',
		color,
		r = 5,
		hollow = false
	}: {
		x: number;
		y: number;
		shape?: Shape;
		color: string;
		r?: number;
		hollow?: boolean;
	} = $props();
	const fill = $derived(hollow ? 'var(--color-surface, #fff)' : color);
	const points = $derived(
		shape === 'diamond'
			? `${x},${y - r * 1.25} ${x + r * 1.25},${y} ${x},${y + r * 1.25} ${x - r * 1.25},${y}`
			: `${x},${y - r * 1.2} ${x + r * 1.1},${y + r * 0.9} ${x - r * 1.1},${y + r * 0.9}`
	);
</script>

{#if shape === 'square'}
	<rect
		x={x - r * 0.9}
		y={y - r * 0.9}
		width={r * 1.8}
		height={r * 1.8}
		{fill}
		stroke={color}
		stroke-width="2"
	/>
{:else if shape === 'diamond' || shape === 'triangle'}
	<polygon {points} {fill} stroke={color} stroke-width="2" />
{:else}
	<circle cx={x} cy={y} {r} {fill} stroke={color} stroke-width="2" />
{/if}
