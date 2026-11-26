<script lang="ts">
	import { createQuery } from '@tanstack/svelte-query';
	import { api } from '$lib/api/client';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	const receipts = createQuery(
		() => ({ queryKey: queryKeys.receipts(), queryFn: () => api.listReceipts() }),
		() => queryClient
	);
</script>

{#each receipts.data ?? [] as receipt (receipt.id)}<p>{receipt.merchant_name}</p>{/each}
