<script lang="ts">
	import { createQuery } from '@tanstack/svelte-query';
	import { api } from '$lib/api/client';
	import { queryClient } from '$lib/query/client';
	import { t } from '$lib/i18n';
	import ReceiptImage from './ReceiptImage.svelte';

	let {
		imageId,
		thumbnailPath,
		alt
	}: { imageId: string; thumbnailPath: string | null; alt: string } = $props();
	let selected = $state(0);
	const image = createQuery(
		() => ({ queryKey: ['image', imageId], queryFn: () => api.getImage(imageId) }),
		() => queryClient
	);
	let count = $derived(1 + (image.data?.additional_images?.length ?? 0));
</script>

{#if count > 1}
	<div class="mb-3 flex flex-wrap gap-2" aria-label={$t('submission.photos')}>
		{#each Array.from({ length: count }, (_, i) => i) as part (part)}
			<button
				type="button"
				class="rounded-lg border border-outline-variant px-3 py-2 text-sm"
				class:bg-primary-container={selected === part}
				aria-pressed={selected === part}
				onclick={() => (selected = part)}
				>{$t('submission.photo', { values: { number: part + 1 } })}</button
			>
		{/each}
	</div>
{/if}
{#if selected === 0}
	<ReceiptImage
		{imageId}
		{thumbnailPath}
		{alt}
		class="mx-auto block h-auto max-w-full rounded-xl border border-outline-variant bg-surface-container object-contain lg:max-h-[calc(100vh-10rem)] lg:w-full"
		clickable={false}
	/>
{:else}
	<img
		src={api.imageFileUrl(imageId, selected)}
		{alt}
		class="mx-auto block h-auto max-w-full rounded-xl border border-outline-variant object-contain lg:max-h-[calc(100vh-10rem)]"
	/>
{/if}
