<script lang="ts">
	import type { Snippet } from 'svelte';
	import { onMount } from 'svelte';
	let {
		title,
		busy = false,
		onCancel,
		children,
		compact = false,
		wide = false
	}: {
		title: string;
		busy?: boolean;
		onCancel: () => void;
		children: Snippet;
		compact?: boolean;
		wide?: boolean;
	} = $props();
	let element: HTMLDialogElement;
	const focusable = () =>
		Array.from(
			element.querySelectorAll<HTMLElement>(
				'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href], [tabindex="0"]'
			)
		);
	onMount(() => {
		const previous = document.activeElement;
		if (typeof element.showModal === 'function') element.showModal();
		else element.setAttribute('open', '');
		(focusable()[0] ?? element).focus();
		return () => {
			if (typeof element.close === 'function') element.close();
			if (previous instanceof HTMLElement && previous.isConnected) previous.focus();
		};
	});
	function keydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			event.preventDefault();
			event.stopPropagation();
			if (!busy) onCancel();
		}
		if (event.key === 'Tab') {
			const targets = focusable();
			const first = targets[0] ?? element;
			const last = targets.at(-1) ?? element;
			if (
				event.shiftKey &&
				(document.activeElement === first || document.activeElement === element)
			) {
				event.preventDefault();
				last.focus();
			} else if (!event.shiftKey && document.activeElement === last) {
				event.preventDefault();
				first.focus();
			}
		}
	}
</script>

<dialog
	bind:this={element}
	aria-label={title}
	aria-modal="true"
	tabindex="-1"
	class="fixed inset-0 m-auto max-h-[90dvh] w-[calc(100%-2rem)] overflow-y-auto rounded-xl border border-outline-variant bg-surface text-on-surface shadow-elevation-4 {wide
		? 'max-w-5xl p-2'
		: 'p-5'} {wide ? '' : compact ? 'max-w-sm' : 'max-w-md'}"
	onclick={(event) => {
		if (busy || event.target !== element) return;
		const rect = element.getBoundingClientRect();
		if (
			event.clientX < rect.left ||
			event.clientX > rect.right ||
			event.clientY < rect.top ||
			event.clientY > rect.bottom
		)
			onCancel();
	}}
	onkeydown={keydown}
	oncancel={(event) => {
		event.preventDefault();
		if (!busy) onCancel();
	}}
>
	{@render children()}
</dialog>

<style>
	dialog::backdrop {
		background: rgb(0 0 0 / 0.5);
	}
</style>
