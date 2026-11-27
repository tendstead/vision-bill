<script lang="ts">
	import { api } from '$lib/api/client';
	import { t } from '$lib/i18n';
	import Icon from '$lib/ui/Icon.svelte';
	import { missingThumbSource } from '$lib/ui/image-policy';

	let {
		imageId,
		thumbnailPath,
		alt,
		class: className = '',
		clickable = true
	}: {
		imageId: string | null;
		thumbnailPath: string | null;
		alt: string;
		class?: string;
		clickable?: boolean;
	} = $props();

	let lightboxOpen = $state(false);
	let thumbFailed = $state(false);

	function showThumb(): boolean {
		// Expecting tablet and desktop view to have enough bandwidth to load the proper image
		let phoneView = window.matchMedia('(max-width: 768px)').matches;
		return imageId !== null && thumbnailPath !== null && !thumbFailed && phoneView;
	}

	function fallback(): string | 'icon' {
		return imageId === null ? 'icon' : missingThumbSource(imageId);
	}

	function open(e: MouseEvent) {
		if (!clickable || imageId === null) return;
		e.preventDefault();
		e.stopPropagation();
		lightboxOpen = true;
	}

	$effect(() => {
		if (!lightboxOpen) return;
		function onKey(e: KeyboardEvent) {
			if (e.key === 'Escape') lightboxOpen = false;
		}
		window.addEventListener('keydown', onKey);
		return () => window.removeEventListener('keydown', onKey);
	});
</script>

{#if imageId === null}
	<span class="flex items-center justify-center bg-surface-container {className}">
		<Icon icon="receipts" />
	</span>
{:else if showThumb()}
	<button
		type="button"
		class="appearance-none p-0 {className}"
		onclick={open}
		tabindex={clickable ? 0 : -1}
		aria-label={alt}
	>
		<img
			src={api.imageThumbUrl(imageId)}
			{alt}
			loading="lazy"
			decoding="async"
			onerror={() => (thumbFailed = true)}
			class="w-full h-full object-contain"
		/>
	</button>
{:else}
	{@const fb = fallback()}
	{#if fb === 'icon'}
		<span class="flex items-center justify-center bg-surface-container {className}">
			<Icon icon="receipts" />
		</span>
	{:else}
		<button
			type="button"
			class="appearance-none p-0 {className}"
			onclick={open}
			tabindex={clickable ? 0 : -1}
			aria-label={alt}
		>
			<img src={fb} {alt} loading="lazy" decoding="async" class="w-full h-full object-contain" />
		</button>
	{/if}
{/if}

{#if lightboxOpen && imageId !== null}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
		role="dialog"
		aria-modal="true"
		aria-label={$t('receipt.viewFullImage')}
		tabindex="-1"
		onclick={(e) => e.target === e.currentTarget && (lightboxOpen = false)}
		onkeydown={(e) => {
			if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') lightboxOpen = false;
		}}
	>
		<button
			type="button"
			class="absolute right-4 top-4 rounded-full bg-black/50 p-2 text-white"
			aria-label={$t('common.close')}
			onclick={() => (lightboxOpen = false)}
		>
			<Icon icon="close" />
		</button>
		<img
			src={api.imageFileUrl(imageId)}
			{alt}
			class="max-h-full max-w-full rounded-lg object-contain"
		/>
	</div>
{/if}
