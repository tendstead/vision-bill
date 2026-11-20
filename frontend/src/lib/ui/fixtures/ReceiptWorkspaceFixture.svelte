<script lang="ts">
	import ReceiptWorkspace from '../ReceiptWorkspace.svelte';
	let value = $state('Original');
	let { hasPhoto = true }: { hasPhoto?: boolean } = $props();
	let request = $state<object | null>(null);
</script>

<ReceiptWorkspace {hasPhoto} showReceipt={request}>
	{#snippet photo()}<div data-testid="photo-position" style="overflow:auto;height:100px">
			<img src="/receipt.png" alt="Test receipt" />
		</div>{/snippet}
	<button type="button" onclick={() => (request = {})}>Show evidence</button>
	<label>Merchant<input bind:value /></label>
</ReceiptWorkspace>
