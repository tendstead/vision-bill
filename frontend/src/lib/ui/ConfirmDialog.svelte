<script lang="ts">
	import { t } from '$lib/i18n';
	import Dialog from './Dialog.svelte';
	import type { Snippet } from 'svelte';

	let {
		open = $bindable(false),
		title,
		body,
		confirmLabel,
		busy = false,
		confirmDisabled = false,
		children,
		onConfirm,
		onCancel
	}: {
		open?: boolean;
		title: string;
		body: string;
		confirmLabel?: string;
		busy?: boolean;
		confirmDisabled?: boolean;
		children?: Snippet;
		onConfirm: () => void;
		onCancel: () => void;
	} = $props();
</script>

{#if open}
	<Dialog {title} {busy} {onCancel} compact>
		<h2 class="text-lg font-semibold">{title}</h2>
		<p class="mt-2 text-sm text-on-surface-variant">{body}</p>
		{@render children?.()}
		<div class="mt-5 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
			<button
				type="button"
				class="rounded-lg px-4 py-2.5 text-sm font-medium text-on-surface-variant hover:bg-surface-container-high disabled:opacity-50"
				disabled={busy}
				onclick={onCancel}
			>
				{$t('common.cancel')}
			</button>
			<button
				type="button"
				class="rounded-lg bg-action px-4 py-2.5 text-sm font-medium text-on-primary hover:opacity-90 disabled:opacity-50"
				disabled={busy || confirmDisabled}
				onclick={onConfirm}
			>
				{confirmLabel ?? $t('common.confirm')}
			</button>
		</div>
	</Dialog>
{/if}
