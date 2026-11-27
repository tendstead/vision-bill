<script lang="ts">
	import { t } from '$lib/i18n';

	let {
		open = $bindable(false),
		title,
		body,
		confirmLabel,
		busy = false,
		onConfirm,
		onCancel
	}: {
		open?: boolean;
		title: string;
		body: string;
		confirmLabel?: string;
		busy?: boolean;
		onConfirm: () => void;
		onCancel: () => void;
	} = $props();

	function onKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape' && !busy) {
			onCancel();
		}
	}
</script>

{#if open}
	<div class="fixed inset-0 z-40 flex items-center justify-center p-4" role="presentation">
		<div
			class="absolute inset-0 bg-black/50"
			role="presentation"
			onclick={busy ? undefined : onCancel}
		></div>
		<div
			role="dialog"
			aria-modal="true"
			aria-label={title}
			class="relative w-full max-w-sm rounded-xl bg-surface p-5 shadow-elevation-4"
			tabindex={-1}
			onkeydown={onKeydown}
		>
			<h2 class="text-lg font-semibold">{title}</h2>
			<p class="mt-2 text-sm text-on-surface-variant">{body}</p>
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
					class="rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-on-primary hover:opacity-90 disabled:opacity-50"
					disabled={busy}
					onclick={onConfirm}
				>
					{confirmLabel ?? $t('common.confirm')}
				</button>
			</div>
		</div>
	</div>
{/if}
