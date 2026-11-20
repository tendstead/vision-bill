<script lang="ts">
	import { captureSessionActor } from '$lib/session-lifetime';
	import { createQuery } from '@tanstack/svelte-query';
	import { api } from '$lib/api/client';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	import { t } from '$lib/i18n';
	import ExtractionSourceViewer from './ExtractionSourceViewer.svelte';
	import type { SourceView, SourceRef } from '$lib/types';

	let {
		imageId,
		alt,
		views = [],
		sources = null
	}: {
		imageId: string;
		alt: string;
		views?: SourceView[];
		sources?: SourceRef[] | null;
	} = $props();
	const actor = captureSessionActor();
	let selected = $state(0);
	let working = $state(false);
	let manualWorking = $state(false);
	function showWorking() {
		manualWorking = true;
		working = true;
	}
	let zoom = $state(100);
	let failed = $state(false);
	$effect(() => {
		void selected;
		void imageId;
		failed = false;
	});
	$effect(() => {
		if (sources === null) return;
		const source = sources[0];
		manualWorking = false;
		if (!source) {
			working = true;
			return;
		}
		if (
			source.coordinate_space === 'original' &&
			Number.isInteger(source.photo_index) &&
			source.photo_index >= 0 &&
			source.photo_index < count
		) {
			selected = source.photo_index;
			working = false;
		} else {
			working = true;
		}
	});
	$effect(() => {
		void imageId;
		selected = 0;
	});
	const image = createQuery(
		() => ({ queryKey: queryKeys.image(imageId), queryFn: () => api.getImage(imageId) }),
		() => queryClient
	);
	let count = $derived(1 + (image.data?.additional_images?.length ?? 0));
</script>

{#if views.length || working}<div class="mb-3 flex flex-wrap gap-2">
		<button
			type="button"
			class="min-h-10 rounded-lg border border-outline-variant px-3 py-2 font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
			class:bg-primary-container={!working}
			class:text-on-primary-container={!working}
			aria-pressed={!working}
			onclick={() => (working = false)}>{$t('review.original')}</button
		>
		<button
			type="button"
			class="min-h-10 rounded-lg border border-outline-variant px-3 py-2 font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
			class:bg-primary-container={working}
			class:text-on-primary-container={working}
			aria-pressed={working}
			onclick={showWorking}>{$t('receiptWorkspace.working')}</button
		>
	</div>{/if}
<div class:hidden={!working}>
	<ExtractionSourceViewer
		views={views.filter((view) => view.coordinate_space === 'working')}
		sources={manualWorking ? [] : (sources ?? [])}
		requireMatch={!manualWorking && sources !== null}
		preferredPhotoIndex={manualWorking ? selected : undefined}
	/>
</div>
<div class:hidden={working}>
	<div class="mb-2 flex items-center gap-3">
		<button
			type="button"
			class="flex size-10 items-center justify-center rounded-lg border border-outline-variant bg-surface-container text-xl font-medium text-on-surface focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary disabled:opacity-40"
			aria-label={$t('receiptWorkspace.zoomOut')}
			disabled={zoom <= 100}
			onclick={() => (zoom = Math.max(100, zoom - 25))}>−</button
		><output>{zoom}%</output><button
			type="button"
			class="flex size-10 items-center justify-center rounded-lg border border-outline-variant bg-surface-container text-xl font-medium text-on-surface focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary disabled:opacity-40"
			aria-label={$t('receiptWorkspace.zoomIn')}
			disabled={zoom >= 300}
			onclick={() => (zoom = Math.min(300, zoom + 25))}>+</button
		>
	</div>
	{#if count > 1}
		<div class="mb-3 flex flex-wrap gap-2" aria-label={$t('submission.photos')}>
			{#each Array.from({ length: count }, (_, i) => i) as part (part)}
				<button
					type="button"
					class="min-h-10 rounded-lg border border-outline-variant px-3 py-2 text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
					class:bg-primary-container={selected === part}
					class:text-on-primary-container={selected === part}
					aria-pressed={selected === part}
					onclick={() => (selected = part)}
					>{$t('submission.photo', { values: { number: part + 1 } })}</button
				>
			{/each}
		</div>
	{/if}
	<!-- svelte-ignore a11y_no_noninteractive_tabindex (Keyboard users must be able to scroll the zoomed image.) -->
	<div
		class="max-h-[70vh] overflow-auto rounded-xl border border-outline-variant"
		role="region"
		tabindex="0"
		aria-label={alt}
	>
		<div style:width={`${zoom}%`}>
			<img
				src={api.imageFileUrl(imageId, selected, actor)}
				{alt}
				class="block h-auto w-full"
				onerror={() => (failed = true)}
			/>
			{#if failed}<p role="alert">{$t('review.sourceUnavailable')}</p>{/if}
		</div>
	</div>
</div>
