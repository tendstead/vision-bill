<script lang="ts">
	import { tick, type Snippet } from 'svelte';
	import { t } from '$lib/i18n';
	let {
		children,
		photo,
		showReceipt = false,
		hasPhoto = true,
		active = $bindable('data')
	}: {
		children: Snippet;
		photo: Snippet;
		showReceipt?: unknown;
		hasPhoto?: boolean;
		active?: string;
	} = $props();
	let lastField: HTMLElement | null = null;
	let receiptButton: HTMLButtonElement | undefined = $state();
	async function showData() {
		active = 'data';
		await tick();
		lastField?.focus({ preventScroll: true });
	}
	$effect(() => {
		if (hasPhoto && showReceipt) {
			const origin = document.activeElement;
			if (origin instanceof HTMLElement && origin !== document.body) lastField = origin;
			active = 'receipt';
			void tick().then(() => {
				if (window.matchMedia?.('(max-width: 1023px)')?.matches)
					receiptButton?.focus({ preventScroll: true });
			});
		}
	});
</script>

{#if hasPhoto}<div
		class="my-4 flex gap-2 lg:hidden"
		role="group"
		aria-label={$t('review.sourceView')}
	>
		<button
			type="button"
			class="min-h-10 rounded-lg border border-outline-variant px-4 py-2 font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
			class:bg-primary-container={active === 'data'}
			class:text-on-primary-container={active === 'data'}
			aria-pressed={active === 'data'}
			onclick={showData}>{$t('receiptWorkspace.data')}</button
		>
		<button
			type="button"
			class="min-h-10 rounded-lg border border-outline-variant px-4 py-2 font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary"
			bind:this={receiptButton}
			class:bg-primary-container={active === 'receipt'}
			class:text-on-primary-container={active === 'receipt'}
			aria-pressed={active === 'receipt'}
			onclick={() => (active = 'receipt')}>{$t('receiptWorkspace.receipt')}</button
		>
	</div>{/if}
<div class="workspace" class:without-photo={!hasPhoto} data-active={active}>
	{#if hasPhoto}<aside class="photo min-w-0">
			<div class="lg:sticky lg:top-20">{@render photo()}</div>
		</aside>{/if}
	<div
		class="data min-w-0"
		onfocusin={(event) => {
			if (event.target instanceof HTMLElement) lastField = event.target;
		}}
	>
		{@render children()}
	</div>
</div>

<style>
	.workspace {
		display: grid;
		gap: 1.5rem;
		min-width: 0;
	}
	@media (min-width: 1024px) {
		.workspace {
			grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
		}
	}
	.workspace.without-photo {
		grid-template-columns: minmax(0, 1fr);
	}
	@media (max-width: 1023px) {
		[data-active='data'] > .photo,
		[data-active='receipt']:not(.without-photo) > .data {
			display: none;
		}
	}
</style>
