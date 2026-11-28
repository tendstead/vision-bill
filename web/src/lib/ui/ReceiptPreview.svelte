<script lang="ts">
	import { api } from '$lib/api/client';
	import { hidePreview, preview } from './receipt-preview.svelte';

	// Thumbnails may not exist (no image helper, old uploads): fall back to the
	// stored photo, and show nothing when that fails too.
	let useFile = $state(false);
	let lastId: string | null = null;
	$effect(() => {
		if (preview.imageId !== lastId) {
			lastId = preview.imageId;
			useFile = false;
		}
	});
	const src = $derived(
		preview.imageId
			? useFile
				? api.imageFileUrl(preview.imageId)
				: api.imageThumbUrl(preview.imageId)
			: null
	);
</script>

<svelte:window onscroll={hidePreview} />
{#if src}
	<img
		{src}
		alt=""
		aria-hidden="true"
		data-testid="receipt-preview"
		class="pointer-events-none fixed z-50 max-h-52 w-40 rounded-lg border border-outline-variant bg-surface object-contain shadow-lg"
		style="left:{preview.left}px; top:{preview.top}px"
		onerror={() => {
			if (useFile) hidePreview();
			else useFile = true;
		}}
	/>
{/if}
