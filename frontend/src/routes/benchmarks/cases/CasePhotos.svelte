<script lang="ts">
	import { t } from '$lib/i18n';
	import { caseApi } from '$lib/api/benchmark-cases';
	import type { SessionActor } from '$lib/session-lifetime';
	let { id, count, actor }: { id: string; count: number; actor: SessionActor } = $props();
	let selected = $state(0);
	let zoom = $state(100);
	let failed = $state(false);
	const url = $derived(caseApi.photoUrl(id, selected, actor));
	function select(index: number) {
		selected = index;
		failed = false;
		zoom = 100;
	}
</script>

<div class="space-y-3">
	<h2 class="text-xl font-semibold">{$t('review.original')}</h2>
	<div class="flex flex-wrap items-center gap-2">
		{#if count > 1}
			{#each Array.from({ length: count }, (_, i) => i) as i (i)}
				<button
					type="button"
					aria-pressed={selected === i}
					class="case-secondary"
					onclick={() => select(i)}>{$t('submission.photo', { values: { number: i + 1 } })}</button
				>
			{/each}
		{/if}
		<button
			type="button"
			class="case-secondary"
			aria-label={$t('receiptWorkspace.zoomOut')}
			disabled={zoom <= 100}
			onclick={() => (zoom -= 25)}>−</button
		>
		<output>{zoom}%</output>
		<button
			type="button"
			class="case-secondary"
			aria-label={$t('receiptWorkspace.zoomIn')}
			disabled={zoom >= 300}
			onclick={() => (zoom += 25)}>+</button
		>
	</div>
	<!-- svelte-ignore a11y_no_noninteractive_tabindex (Keyboard users can pan the zoomed original.) -->
	<div
		class="max-h-[70vh] overflow-auto rounded-lg border border-outline-variant"
		role="region"
		tabindex="0"
		aria-label={$t('review.original')}
	>
		{#key url}<div style:width={zoom + '%'}>
				{#if url}<img
						class="h-auto w-full"
						src={url}
						alt={$t('submission.photo', { values: { number: selected + 1 } })}
						onerror={() => (failed = true)}
					/>{/if}
			</div>{/key}
		{#if failed || !url}<p role="alert">{$t('review.sourceUnavailable')}</p>{/if}
	</div>
</div>
