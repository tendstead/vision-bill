<script lang="ts">
	import { createQuery } from '@tanstack/svelte-query';
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import { goto } from '$app/navigation';
	import { t, translate } from '$lib/i18n';
	import { api, ApiError } from '$lib/api/client';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	import { formatMoney } from '$lib/ui/money';
	import { formatRelativeTime } from '$lib/ui/time';
	import { snackbar } from '$lib/ui/snackbar.svelte';
	import { PRESET_COLORS, colorToStyle } from '$lib/collection-colors';
	import ConfirmDialog from '$lib/ui/ConfirmDialog.svelte';
	import Icon from '$lib/ui/Icon.svelte';
	import StatisticsCharts from '$lib/ui/StatisticsCharts.svelte';
	import type { CollectionDetail, ReceiptRow } from '$lib/types';

	let id = $derived(page.url.pathname.split('/').pop() ?? '');
	const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

	let deleteOpen = $state(false);
	let deleting = $state(false);

	let editOpen = $state(false);
	let saving = $state(false);
	let name = $state('');
	let color = $state<string | null>(PRESET_COLORS[0] ?? null);
	let startDate = $state('');
	let endDate = $state('');
	let formError = $state('');

	let unassigningId: string | null = $state(null);

	const detail = createQuery(
		() => ({
			queryKey: queryKeys.collection(id),
			queryFn: () => api.getCollection(id),
			enabled: uuidPattern.test(id)
		}),
		() => queryClient
	);

	const statistics = createQuery(
		() => ({
			queryKey: queryKeys.statistics(12, id),
			queryFn: () => api.getStatistics(12, id),
			enabled: uuidPattern.test(id)
		}),
		() => queryClient
	);

	let notFound = $derived(
		!uuidPattern.test(id) ||
			(detail.status === 'error' && detail.error instanceof ApiError && detail.error.status === 404)
	);
	let data = $derived(detail.data ?? null);
	let stats = $derived(statistics.data ?? null);

	async function refresh(): Promise<void> {
		await queryClient.invalidateQueries({ queryKey: queryKeys.collections() });
		await queryClient.invalidateQueries({ queryKey: queryKeys.statistics(12, id) });
	}

	function openEdit(): void {
		if (!data) return;
		name = data.name;
		color = data.color ?? PRESET_COLORS[0] ?? null;
		startDate = data.start_date ?? '';
		endDate = data.end_date ?? '';
		formError = '';
		editOpen = true;
	}

	function closeEdit(): void {
		if (saving) return;
		editOpen = false;
	}

	// True when the chosen color is not one of the presets (the native picker
	// returns lowercase hex, so compare case-insensitively).
	let isCustomColor = $derived.by(() => {
		const current = color?.toLowerCase() ?? null;
		return current !== null && !PRESET_COLORS.some((p) => p.toLowerCase() === current);
	});

	function onColorPicker(event: Event): void {
		color = (event.currentTarget as HTMLInputElement).value || null;
	}

	async function submitEdit(): Promise<void> {
		if (saving || !data) return;
		if (!name.trim()) {
			formError = translate('collections.name');
			return;
		}
		if (startDate && endDate && startDate > endDate) {
			formError = translate('collections.dateInvalid');
			return;
		}
		saving = true;
		formError = '';
		try {
			await api.updateCollection(id, {
				name: name.trim(),
				color,
				start_date: startDate || null,
				end_date: endDate || null
			});
			editOpen = false;
			await refresh();
		} catch {
			snackbar.notify('error', translate('collections.updateFailed'));
		} finally {
			saving = false;
		}
	}

	async function removeCollection(): Promise<void> {
		if (deleting) return;
		deleting = true;
		try {
			await api.deleteCollection(id);
			await queryClient.invalidateQueries({ queryKey: queryKeys.collections() });
			await goto(resolve('/collections'));
		} catch {
			snackbar.notify('error', translate('collections.deleteFailed'));
			deleting = false;
		}
	}

	async function removeReceipt(receipt: ReceiptRow): Promise<void> {
		if (unassigningId) return;
		unassigningId = receipt.id;
		try {
			await api.unassignReceiptFromCollection(id, receipt.id);
			snackbar.notify('success', translate('collections.removedFromCollection'));
			await refresh();
		} catch {
			snackbar.notify('error', translate('common.error'));
		} finally {
			unassigningId = null;
		}
	}

	function dateRange(c: CollectionDetail): string {
		if (!c.start_date && !c.end_date) return translate('collections.noDate');
		return `${c.start_date ?? '…'} – ${c.end_date ?? '…'}`;
	}

	function vendor(receipt: ReceiptRow): string {
		return receipt.merchant_name || $t('receipts.unknownVendor');
	}
</script>

<svelte:head>
	<title>{$t('collections.title')} – {$t('app.name')}</title>
</svelte:head>

{#if detail.isLoading}
	<p class="text-sm text-on-surface-variant">{$t('common.loading')}</p>
{:else if notFound}
	<div class="rounded-xl border border-outline-variant bg-surface-container-low p-4 text-sm">
		<p class="font-medium">{$t('collections.notFoundTitle')}</p>
		<p class="mt-1 text-on-surface-variant">{$t('collections.notFoundBody')}</p>
		<a
			href={resolve('/collections')}
			class="mt-3 inline-block rounded-lg bg-primary px-4 py-2 text-on-primary hover:opacity-90"
		>
			{$t('common.back')}
		</a>
	</div>
{:else if detail.error}
	<div
		class="rounded-xl border border-outline-variant bg-surface-container-low p-4 text-sm text-on-surface-variant"
	>
		{$t('common.error')}
	</div>
{:else if data}
	<section class="mx-auto max-w-5xl">
		<div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
			<div class="flex min-w-0 items-center gap-3">
				<a
					href={resolve('/collections')}
					class="rounded-lg p-2 text-on-surface-variant hover:bg-surface-container"
					aria-label={$t('common.back')}
				>
					<Icon icon="close" />
				</a>
				<span class="size-4 shrink-0 rounded-full" style={colorToStyle(data.color)}></span>
				<div class="min-w-0">
					<h1 class="truncate text-xl font-semibold">{data.name}</h1>
					<p class="truncate text-xs text-on-surface-variant">
						{dateRange(data)}
						{#if data.active}
							· {$t('collections.capturing')}
						{/if}
					</p>
				</div>
			</div>
			<div class="flex shrink-0 gap-2">
				<button
					type="button"
					class="inline-flex items-center gap-2 rounded-lg border border-outline-variant px-4 py-2 text-sm font-medium hover:bg-surface-container-high"
					onclick={openEdit}
				>
					<Icon icon="edit" />
					{$t('collections.edit')}
				</button>
				<button
					type="button"
					class="inline-flex items-center gap-2 rounded-lg border border-error/50 px-4 py-2 text-sm font-medium text-error hover:bg-error-container"
					onclick={() => (deleteOpen = true)}
				>
					<Icon icon="trash" />
					{$t('collections.delete')}
				</button>
			</div>
		</div>

		{#if data.totals.length}
			<p class="mt-4 text-sm">
				<span class="text-on-surface-variant">{$t('collections.total')}: </span>
				{#each data.totals as total, i (total.currency)}
					<span class="font-medium">{formatMoney(total.total, total.currency)}</span>
					{#if i < data.totals.length - 1}
						·
					{/if}
				{/each}
			</p>
		{/if}

		<div class="mt-6">
			<h2 class="text-lg font-semibold">{$t('collections.receipts')}</h2>
			{#if data.receipts.length === 0}
				<div class="mt-4 rounded-xl border border-dashed border-outline-variant p-8 text-center">
					<p class="text-sm text-on-surface-variant">{$t('collections.noReceipts')}</p>
				</div>
			{:else}
				<ul class="mt-4 space-y-3">
					{#each data.receipts as receipt (receipt.id)}
						<li class="flex items-center gap-3">
							<a
								href={resolve(`/receipts/${receipt.id}`)}
								class="flex min-w-0 flex-1 items-center gap-3 rounded-xl border border-outline-variant bg-surface-container-low p-4 transition-colors hover:border-primary/50"
							>
								<span
									class="flex size-12 shrink-0 items-center justify-center rounded-lg bg-surface-container"
								>
									<Icon icon="receipts" />
								</span>
								<div class="min-w-0 flex-1">
									<p class="truncate text-sm font-medium">{vendor(receipt)}</p>
									<p class="text-xs text-on-surface-variant">
										{formatMoney(receipt.total, receipt.currency)}
										{#if receipt.date}
											· {receipt.date}
										{/if}
										{#if receipt.created_at}
											· {formatRelativeTime(receipt.created_at)}
										{/if}
									</p>
								</div>
								<span
									class="shrink-0 rounded-full px-2.5 py-1 text-xs font-medium
										{receipt.status === 'verified'
										? 'bg-primary-container text-on-primary-container'
										: 'bg-surface-container-highest text-on-surface-variant'}"
								>
									{$t(receipt.status === 'verified' ? 'receipt.verified' : 'receipt.unverified')}
								</span>
							</a>
							<button
								type="button"
								class="shrink-0 rounded-lg p-2 text-on-surface-variant hover:bg-error-container hover:text-on-error-container disabled:opacity-60"
								aria-label={translate('collections.removeReceipt', {
									values: { name: vendor(receipt) }
								})}
								title={translate('collections.removeReceipt', {
									values: { name: vendor(receipt) }
								})}
								disabled={unassigningId !== null}
								onclick={() => removeReceipt(receipt)}
							>
								<Icon icon="remove" />
							</button>
						</li>
					{/each}
				</ul>
			{/if}
		</div>

		<div class="mt-8">
			<h2 class="text-lg font-semibold">{$t('pages.statistics.title')}</h2>
			{#if statistics.isLoading}
				<p class="mt-4 text-sm text-on-surface-variant">{$t('common.loading')}</p>
			{:else if statistics.error}
				<p
					class="mt-4 rounded-xl border border-outline-variant bg-surface-container-low p-4 text-sm text-on-surface-variant"
				>
					{$t('common.error')}
				</p>
			{:else if stats?.verified_receipt_count === 0}
				<p class="mt-4 text-sm text-on-surface-variant">{$t('statistics.emptyBody')}</p>
			{:else if stats}
				<div class="mt-4">
					<StatisticsCharts {stats} />
				</div>
			{/if}
		</div>
	</section>

	{#if editOpen}
		<div class="fixed inset-0 z-40 flex items-center justify-center p-4" role="presentation">
			<div class="absolute inset-0 bg-black/50" role="presentation" onclick={closeEdit}></div>
			<div
				role="dialog"
				aria-modal="true"
				aria-label={$t('collections.edit')}
				class="relative w-full max-w-md rounded-xl bg-surface p-5 shadow-elevation-4"
			>
				<form
					onsubmit={(event) => {
						event.preventDefault();
						void submitEdit();
					}}
				>
					<h2 class="text-lg font-semibold">{$t('collections.edit')}</h2>

					<label class="mt-4 block text-sm">
						{$t('collections.name')}
						<input
							type="text"
							required
							class="mt-1 w-full rounded-lg border border-outline-variant bg-surface-container-lowest px-3 py-2 text-sm"
							value={name}
							oninput={(event) => (name = (event.currentTarget as HTMLInputElement).value)}
						/>
					</label>

					<div class="mt-4">
						<span class="block text-sm">{$t('collections.color')}</span>
						<div class="mt-1 flex flex-wrap items-center gap-2">
							{#each PRESET_COLORS as preset (preset)}
								<button
									type="button"
									aria-label={preset}
									aria-pressed={color === preset}
									class="size-8 rounded-full ring-2 ring-offset-2 {color === preset
										? 'ring-primary'
										: 'ring-transparent'}"
									style={colorToStyle(preset)}
									onclick={() => (color = preset)}
								></button>
							{/each}
							<!-- Custom color: a labeled pill (clearly not a preset dot) that
						     opens the native color picker; selected when a non-preset color is set. -->
							<label
								class="flex h-8 cursor-pointer items-center gap-2 rounded-lg border border-outline-variant px-2.5 text-sm transition-colors hover:border-primary/50 focus-within:border-primary focus-within:ring-1 focus-within:ring-primary {isCustomColor
									? 'border-primary ring-2 ring-primary/30'
									: ''}"
							>
								<span
									class="size-5 shrink-0 rounded-full border border-outline-variant"
									style={colorToStyle(color)}
								></span>
								{$t('collections.colorCustom')}
								<input
									type="color"
									class="sr-only"
									aria-label={$t('collections.colorCustom')}
									value={color && color !== '' ? color : '#64748B'}
									oninput={onColorPicker}
								/>
							</label>
						</div>
					</div>

					<div class="mt-4 grid grid-cols-2 gap-3">
						<label class="block text-sm">
							{$t('collections.start')}
							<input
								type="date"
								class="mt-1 w-full rounded-lg border border-outline-variant bg-surface-container-lowest px-3 py-2 text-sm"
								value={startDate}
								oninput={(event) => (startDate = (event.currentTarget as HTMLInputElement).value)}
							/>
						</label>
						<label class="block text-sm">
							{$t('collections.end')}
							<input
								type="date"
								class="mt-1 w-full rounded-lg border border-outline-variant bg-surface-container-lowest px-3 py-2 text-sm"
								value={endDate}
								oninput={(event) => (endDate = (event.currentTarget as HTMLInputElement).value)}
							/>
						</label>
					</div>

					{#if formError}
						<p class="mt-3 text-sm text-error" role="alert">{formError}</p>
					{/if}

					<div class="mt-5 flex justify-end gap-2">
						<button
							type="button"
							class="rounded-lg border border-outline-variant px-4 py-2 text-sm font-medium hover:bg-surface-container-high"
							onclick={closeEdit}
						>
							{$t('collections.cancel')}
						</button>
						<button
							type="submit"
							disabled={saving}
							class="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-on-primary hover:bg-primary/90 disabled:opacity-60"
						>
							{$t('collections.save')}
						</button>
					</div>
				</form>
			</div>
		</div>
	{/if}

	<ConfirmDialog
		open={deleteOpen}
		title={$t('collections.deleteTitle')}
		body={$t('collections.deleteBody')}
		confirmLabel={$t('collections.delete')}
		busy={deleting}
		onConfirm={removeCollection}
		onCancel={() => (deleteOpen = false)}
	/>
{/if}
