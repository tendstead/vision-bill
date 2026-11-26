<script lang="ts">
	import { captureSessionActor } from '$lib/session-lifetime';
	import { api } from '$lib/api/client';
	import { t } from '$lib/i18n';
	import type { SourceView, SourceRef } from '$lib/types';
	let {
		views,
		sources = [],
		preferredPhotoIndex,
		requireMatch = false
	}: {
		views: SourceView[];
		sources?: SourceRef[];
		preferredPhotoIndex?: number;
		requireMatch?: boolean;
	} = $props();
	const actor = captureSessionActor();
	let failed = $state(false);
	let zoom = $state(100);
	let matching = $derived(
		views.find((v) =>
			sources.some(
				(s) =>
					s.view_id === v.view_id &&
					s.photo_index === v.photo_index &&
					s.coordinate_space === v.coordinate_space
			)
		)
	);
	let selected = $state('');
	$effect(() => {
		const request = sources;
		selected =
			matching?.view_id ??
			(request.length || requireMatch
				? ''
				: (views.find((view) => view.photo_index === preferredPhotoIndex)?.view_id ??
					views[0]?.view_id ??
					''));
	});
	let view = $derived(views.find((v) => v.view_id === selected));
	$effect(() => {
		void view?.view_id;
		void view?.url;
		failed = false;
	});
	let boxes = $derived(
		view
			? sources.filter(
					(s) =>
						s.view_id === view.view_id &&
						s.photo_index === view.photo_index &&
						s.coordinate_space === view.coordinate_space &&
						s.bbox &&
						s.bbox.length >= 3 &&
						s.bbox.every(
							(p) =>
								p.length === 2 &&
								Number.isFinite(p[0]) &&
								Number.isFinite(p[1]) &&
								p[0] >= 0 &&
								p[1] >= 0 &&
								p[0] <= view.width &&
								p[1] <= view.height
						)
				)
			: []
	);
</script>

<div class="my-4">
	<div class="mb-2 flex items-center gap-3">
		<button
			type="button"
			class="flex size-10 items-center justify-center rounded-lg border border-outline-variant bg-surface-container text-xl font-medium text-on-surface focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary disabled:opacity-40"
			aria-label={$t('receiptWorkspace.zoomOut')}
			disabled={zoom <= 100}
			onclick={() => (zoom = Math.max(100, zoom - 25))}>−</button
		>
		<output>{zoom}%</output>
		<button
			type="button"
			class="flex size-10 items-center justify-center rounded-lg border border-outline-variant bg-surface-container text-xl font-medium text-on-surface focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary disabled:opacity-40"
			aria-label={$t('receiptWorkspace.zoomIn')}
			disabled={zoom >= 300}
			onclick={() => (zoom = Math.min(300, zoom + 25))}>+</button
		>
	</div>
	{#if views.length > 1}<label
			>{$t('review.sourceView')}<select bind:value={selected}
				>{#if selected === ''}<option value="" disabled>{$t('review.mappingUnknown')}</option
					>{/if}{#each views as item (item.view_id)}<option value={item.view_id}
						>{item.photo_index + 1}: {item.coordinate_space} ({item.width} × {item.height})</option
					>{/each}</select
			></label
		>{/if}
	{#if view}
		<p class="text-xs text-on-surface-variant">{view.width} × {view.height}</p>
		<!-- svelte-ignore a11y_no_noninteractive_tabindex (Keyboard users must be able to scroll the zoomed image.) -->
		<div
			class="max-h-[70vh] overflow-auto rounded-xl border border-outline-variant"
			tabindex="0"
			role="region"
			aria-label={$t('review.sourceView')}
		>
			<div class="relative" style:width={`${zoom}%`}>
				<img
					src={api.sourceViewUrl(view.url, actor)}
					alt={$t('review.sourceView')}
					width={view.width}
					height={view.height}
					class="block h-auto w-full"
					onerror={() => (failed = true)}
				/>
				{#if !failed && boxes.length}<svg
						class="pointer-events-none absolute inset-0 h-full w-full"
						viewBox={`0 0 ${view.width} ${view.height}`}
						aria-label={$t('review.sourceBoxes')}
					>
						{#each boxes as source, index (index)}<polygon
								points={source.bbox!.map((p) => p.join(',')).join(' ')}
								fill="var(--vb-evidence-fill)"
								stroke="var(--vb-evidence-stroke)"
								stroke-width="3"
							/>{/each}
					</svg>{/if}
			</div>
		</div>
	{/if}
	{#if !boxes.length}<p class="text-sm">{$t('review.mappingUnknown')}</p>{/if}
	{#if failed}<p role="alert">{$t('review.sourceUnavailable')}</p>{/if}
</div>
