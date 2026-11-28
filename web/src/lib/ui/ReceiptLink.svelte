<script lang="ts">
	import { resolve } from '$app/paths';
	import { shortId } from '$lib/benchmark-compare';
	import { previewProps } from './receipt-preview.svelte';

	let {
		receiptId,
		imageId = null,
		exists = true
	}: { receiptId: string; imageId?: string | null; exists?: boolean } = $props();
</script>

{#if exists}
	<a
		class="font-mono underline"
		href={resolve(`/receipts/${receiptId}`)}
		title={receiptId}
		{...previewProps(imageId)}>{shortId(receiptId)}</a
	>
{:else}
	<span class="font-mono line-through" title="This receipt was deleted">{shortId(receiptId)}</span>
{/if}
