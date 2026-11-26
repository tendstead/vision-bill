<script lang="ts">
	import { resolve } from '$app/paths';
	import { t } from '$lib/i18n';
	import { api } from '$lib/api/client';
	import { captureSessionActor, isCurrentSessionActor } from '$lib/session-lifetime';
	import { selectedReceiptIds } from '$lib/benchmark-experiments';
	import type { ReceiptRow } from '$lib/types';

	let {
		receiptIds = $bindable(),
		onPreview,
		busy
	}: { receiptIds: string; onPreview: (id: string) => void; busy: boolean } = $props();
	const actor = captureSessionActor();
	const pageSize = 10;
	let search = $state('');
	let offset = $state(0);
	let rows = $state<ReceiptRow[]>([]);
	let loading = $state(true);
	let failed = $state(false);
	let retry = $state(0);
	let labels = $state<Record<string, string>>({});
	const ids = $derived(selectedReceiptIds(receiptIds));

	$effect(() => {
		if (!actor || !isCurrentSessionActor(actor)) return;
		const filters = { search: search.trim(), offset, limit: pageSize + 1 };
		void retry;
		let active = true;
		loading = true;
		failed = false;
		const timer = setTimeout(
			async () => {
				try {
					const receipts = await api.listReceipts(filters, actor);
					if (active && isCurrentSessionActor(actor)) rows = receipts;
				} catch {
					if (active && isCurrentSessionActor(actor)) failed = true;
				} finally {
					if (active && isCurrentSessionActor(actor)) loading = false;
				}
			},
			filters.search ? 200 : 0
		);
		return () => {
			active = false;
			clearTimeout(timer);
		};
	});
	function toggle(id: string) {
		receiptIds = (ids.includes(id) ? ids.filter((value) => value !== id) : [...ids, id]).join(',');
	}
	function eligible(row: ReceiptRow) {
		return (
			row.verified &&
			row.status === 'verified' &&
			row.verification_source === 'human' &&
			!!row.image_id
		);
	}
</script>

<label class="block text-sm font-medium">
	{$t('benchmarkForm.search')}
	<input
		class="mt-1 block w-full rounded-lg border border-outline-variant p-2"
		type="search"
		bind:value={search}
		oninput={() => (offset = 0)}
	/>
</label>
{#if loading}<p class="my-2 text-sm">{$t('common.loading')}</p>
{:else if failed}<p class="my-2 text-sm">
		{$t('benchmarkForm.receiptsError')}
		<button class="text-primary underline" type="button" onclick={() => retry++}
			>{$t('benchmarkForm.retry')}</button
		>
	</p>
{:else}
	<ul class="mt-2 divide-y divide-outline-variant">
		{#each rows.slice(0, pageSize) as row (row.id)}
			<li class="flex flex-wrap items-center justify-between gap-2 py-2">
				<label class="inline-flex items-center gap-3 text-sm">
					<input
						type="checkbox"
						checked={ids.includes(row.id)}
						disabled={!eligible(row)}
						onchange={() => {
							labels[row.id] = row.merchant_display_name || row.merchant_name || row.id;
							toggle(row.id);
						}}
					/>
					{#if row.image_id && row.thumbnail_path}
						<img
							src={api.imageThumbUrl(row.image_id, actor)}
							alt=""
							width="40"
							height="48"
							loading="lazy"
							class="h-12 w-10 shrink-0 rounded bg-surface-container object-contain"
						/>
					{/if}
					<span
						><strong
							>{row.merchant_display_name ||
								row.merchant_name ||
								$t('receipts.unknownVendor')}</strong
						>
						· {row.date} · {row.total}
						{row.currency}
						<small class="block text-on-surface-variant"
							>{$t(
								!row.image_id
									? 'benchmarkForm.missingImage'
									: eligible(row)
										? 'benchmarkForm.humanReviewed'
										: 'benchmarkForm.needsReview'
							)}</small
						>
					</span>
				</label>
				{#if ids.includes(row.id)}
					<button
						type="button"
						disabled={busy}
						class="text-sm text-primary underline"
						onclick={() => onPreview(row.id)}
						aria-label={$t('benchmarkForm.previewOne', {
							values: { name: row.merchant_display_name || row.merchant_name || row.id }
						})}>{$t('benchmarkForm.preview')}</button
					>
				{/if}
				<a class="text-sm text-primary underline" href={resolve('/receipts/[id]', { id: row.id })}
					>{$t('benchmarkUi.inspect')}</a
				>
			</li>
		{:else}
			<li class="py-2 text-sm text-on-surface-variant">{$t('benchmarkForm.noReceipts')}</li>
		{/each}
	</ul>
	{#if offset || rows.length > pageSize}
		<div class="mt-2 flex gap-4 text-sm">
			<button
				type="button"
				class="text-primary disabled:opacity-50"
				disabled={!offset}
				onclick={() => (offset -= pageSize)}>{$t('benchmarkForm.previous')}</button
			>
			<button
				type="button"
				class="text-primary disabled:opacity-50"
				disabled={rows.length <= pageSize}
				onclick={() => (offset += pageSize)}>{$t('benchmarkForm.next')}</button
			>
		</div>
	{/if}
{/if}
{#if ids.length}
	<div class="mt-3 flex flex-wrap gap-2 text-sm" aria-label={$t('benchmarkForm.selectedReceipts')}>
		{#each ids as id (id)}
			<button
				type="button"
				class="break-all rounded-lg border border-outline-variant px-2 py-1"
				onclick={() => toggle(id)}
				aria-label={$t('benchmarkForm.remove', { values: { name: labels[id] || id } })}
				>{labels[id] || id} ×</button
			>
		{/each}
	</div>
{/if}
