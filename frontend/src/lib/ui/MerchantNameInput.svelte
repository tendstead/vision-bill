<script lang="ts">
	import { api } from '$lib/api/client';
	import { t } from '$lib/i18n';
	import { captureSessionActor, isCurrentSessionActor } from '$lib/session-lifetime';

	let {
		id,
		value,
		printed,
		address,
		receiptId,
		imageId,
		onChange,
		onCommit
	}: {
		id: string;
		value: string | null | undefined;
		printed: string | null | undefined;
		address: string | null | undefined;
		receiptId?: string;
		imageId?: string;
		onChange: (name: string) => void;
		onCommit?: () => void;
	} = $props();

	const actor = captureSessionActor();
	let matches = $state<string[]>([]);
	let suggestion = $state<string | null>(null);
	let failed = $state(false);

	$effect(() => {
		const params = {
			merchant_name: printed?.trim() || undefined,
			merchant_address: address?.trim() || undefined,
			q: value?.trim() || undefined,
			receipt_id: receiptId,
			image_id: imageId
		};
		let current = true;
		matches = [];
		suggestion = null;
		failed = false;
		if (!isCurrentSessionActor(actor)) return;
		const timer = setTimeout(async () => {
			try {
				const result = await api.listMerchants(params, actor);
				if (current && isCurrentSessionActor(actor)) {
					matches = result.merchants;
					suggestion = result.suggestion;
				}
			} catch {
				if (current && isCurrentSessionActor(actor)) failed = true;
			}
		}, 150);
		return () => {
			current = false;
			clearTimeout(timer);
		};
	});
</script>

<label class="mb-1 block text-xs font-medium text-on-surface-variant" for={id}
	>{$t('editor.merchantDisplayName')}</label
>
<input
	{id}
	list={id + '-options'}
	value={value ?? ''}
	maxlength="200"
	aria-describedby={id + '-hint'}
	class="block w-full rounded border p-2"
	oninput={(event) => onChange(event.currentTarget.value)}
	onblur={onCommit}
/>
<datalist id={id + '-options'}>
	{#each matches as name (name)}<option value={name}></option>{/each}
</datalist>
<p id={id + '-hint'} class="text-xs text-on-surface-variant">{$t('editor.merchantDisplayHint')}</p>
{#if suggestion && suggestion !== value?.trim()}
	<button
		type="button"
		class="text-primary underline"
		onclick={() => {
			if (suggestion) {
				onChange(suggestion);
				onCommit?.();
			}
		}}
	>
		{$t('editor.useMerchantSuggestion', { values: { name: suggestion } })}
	</button>
{/if}
{#if failed}<p role="status">{$t('editor.merchantLookupFailed')}</p>{/if}
