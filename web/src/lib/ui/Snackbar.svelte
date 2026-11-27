<script lang="ts">
	import { snackbar } from './snackbar.svelte.ts';
	import { t } from '$lib/i18n';
	import Icon from './Icon.svelte';
</script>

{#if $snackbar.length > 0}
	<div class="fixed inset-x-0 bottom-24 z-30 flex flex-col items-center gap-2 px-4 md:bottom-8">
		{#each $snackbar as item (item.id)}
			<div
				role={item.kind === 'error' ? 'alert' : 'status'}
				class="flex max-w-md items-center gap-3 rounded-lg px-4 py-3 shadow-elevation-3
					{item.kind === 'error'
					? 'bg-error-container text-on-error-container'
					: 'bg-inverse-surface text-inverse-on-surface'}"
			>
				<span class="flex-1 text-sm">{item.text}</span>
				<button
					type="button"
					class="rounded-full p-1 hover:bg-black/10"
					aria-label={$t('common.close')}
					onclick={() => snackbar.close(item.id)}
				>
					<Icon icon="close" />
				</button>
			</div>
		{/each}
	</div>
{/if}
