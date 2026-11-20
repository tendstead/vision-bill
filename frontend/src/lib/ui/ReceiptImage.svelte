<script lang="ts">
	import { onMount } from 'svelte';
	import { captureSessionActor } from '$lib/session-lifetime';
	import { api } from '$lib/api/client';
	import { t } from '$lib/i18n';
	import Dialog from '$lib/ui/Dialog.svelte';
	import Icon from '$lib/ui/Icon.svelte';
	import { missingThumbSource } from '$lib/ui/image-policy';

	let {
		imageId,
		thumbnailPath,
		alt,
		previewLabel,
		class: className = ''
	}: {
		imageId: string | null;
		thumbnailPath: string | null;
		alt: string;
		previewLabel?: string;
		class?: string;
	} = $props();

	const actor = captureSessionActor();
	const compactViewport =
		typeof window === 'undefined' ? undefined : window.matchMedia?.('(max-width: 768px)');
	let compact = $state(compactViewport?.matches ?? false);
	let lightboxOpen = $state(false);
	let failedThumbnail = $state<string | null>(null);
	const thumbnail = $derived(
		imageId && thumbnailPath && compact ? api.imageThumbUrl(imageId, actor) : null
	);
	const imageSource = $derived(
		thumbnail && thumbnail !== failedThumbnail
			? thumbnail
			: imageId
				? missingThumbSource(imageId, actor)
				: null
	);

	onMount(() => {
		const onViewportChange = (event: MediaQueryListEvent) => (compact = event.matches);
		compactViewport?.addEventListener('change', onViewportChange);
		return () => compactViewport?.removeEventListener('change', onViewportChange);
	});
</script>

{#if imageSource === null || imageSource === 'icon'}
	<span class="flex items-center justify-center bg-surface-container {className}">
		<Icon icon="receipts" />
	</span>
{:else}
	<button
		type="button"
		class="appearance-none p-0 {className}"
		onclick={() => (lightboxOpen = true)}
		aria-haspopup="dialog"
		aria-label={previewLabel || alt || $t('receipt.viewFullImage')}
	>
		<img
			src={imageSource}
			alt=""
			loading="lazy"
			decoding="async"
			onerror={() => {
				if (imageSource === thumbnail) failedThumbnail = imageSource;
			}}
			class="h-full w-full object-contain"
		/>
	</button>
{/if}

{#if lightboxOpen && imageId !== null}
	<Dialog title={$t('receipt.viewFullImage')} wide onCancel={() => (lightboxOpen = false)}>
		<div class="flex justify-end p-1">
			<button
				type="button"
				class="rounded-lg p-2 hover:bg-surface-container-high"
				aria-label={$t('common.close')}
				onclick={() => (lightboxOpen = false)}
			>
				<Icon icon="close" />
			</button>
		</div>
		<img
			src={api.imageFileUrl(imageId, 0, actor)}
			alt={alt || $t('submission.photo', { values: { number: 1 } })}
			class="mx-auto max-h-[calc(90dvh-4.5rem)] max-w-full rounded-lg object-contain"
		/>
	</Dialog>
{/if}
