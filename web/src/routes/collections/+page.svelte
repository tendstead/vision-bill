<script lang="ts">
	import { createQuery } from '@tanstack/svelte-query';
	import { resolve } from '$app/paths';
	import { t, translate } from '$lib/i18n';
	import { api } from '$lib/api/client';
	import { queryKeys } from '$lib/query/keys';
	import { queryClient } from '$lib/query/client';
	import { formatMoney } from '$lib/ui/money';
	import { snackbar } from '$lib/ui/snackbar.svelte';
	import { PRESET_COLORS, colorToStyle } from '$lib/collection-colors';
	import ConfirmDialog from '$lib/ui/ConfirmDialog.svelte';
	import Icon from '$lib/ui/Icon.svelte';
	import type { CollectionSummary } from '$lib/types';

	let dialogOpen = $state(false);
	let creating = $state(false);
	let name = $state('');
	let color = $state<string | null>(PRESET_COLORS[0] ?? null);
	let startDate = $state('');
	let endDate = $state('');
	let formError = $state('');

	let confirmDelete: CollectionSummary | null = $state(null);
	let deleting = $state(false);
	let togglingId: string | null = $state(null);

	const list = createQuery(
		() => ({ queryKey: queryKeys.collections(), queryFn: () => api.listCollections() }),
		() => queryClient
	);

	function openDialog(): void {
		name = '';
		color = PRESET_COLORS[0] ?? null;
		startDate = '';
		endDate = '';
		formError = '';
		dialogOpen = true;
	}

	function closeDialog(): void {
		if (creating) return;
		dialogOpen = false;
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

	async function submitForm(): Promise<void> {
		if (creating) return;
		if (!name.trim()) {
			formError = translate('collections.name');
			return;
		}
		if (startDate && endDate && startDate > endDate) {
			formError = translate('collections.dateInvalid');
			return;
		}
		creating = true;
		formError = '';
		try {
			await api.createCollection({
				name: name.trim(),
				color,
				start_date: startDate || null,
				end_date: endDate || null
			});
			await queryClient.invalidateQueries({ queryKey: queryKeys.collections() });
			dialogOpen = false;
		} catch {
			snackbar.notify('error', translate('collections.createFailed'));
		} finally {
			creating = false;
		}
	}

	async function toggleActive(c: CollectionSummary): Promise<void> {
		if (togglingId) return;
		togglingId = c.id;
		try {
			if (c.active) await api.deactivateCollection(c.id);
			else await api.activateCollection(c.id);
			await queryClient.invalidateQueries({ queryKey: queryKeys.collections() });
		} catch {
			snackbar.notify(
				'error',
				translate(c.active ? 'collections.deactivateFailed' : 'collections.activateFailed')
			);
		} finally {
			togglingId = null;
		}
	}

	async function removeCollection(): Promise<void> {
		if (!confirmDelete || deleting) return;
		deleting = true;
		try {
			await api.deleteCollection(confirmDelete.id);
			confirmDelete = null;
			await queryClient.invalidateQueries({ queryKey: queryKeys.collections() });
		} catch {
			snackbar.notify('error', translate('collections.deleteFailed'));
		} finally {
			deleting = false;
		}
	}
</script>

<svelte:head>
	<title>{$t('collections.title')} – {$t('app.name')}</title>
</svelte:head>

<section class="mx-auto max-w-3xl">
	<div class="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
		<div>
			<h1 class="text-2xl font-semibold">{$t('collections.title')}</h1>
			<p class="mt-1 text-sm text-on-surface-variant">{$t('collections.subtitle')}</p>
		</div>
		<button
			type="button"
			class="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-on-primary hover:bg-primary/90"
			onclick={openDialog}
		>
			<Icon icon="plus" />
			{$t('collections.new')}
		</button>
	</div>

	<p class="mt-2 text-xs text-on-surface-variant">{$t('collections.doubleCountNote')}</p>

	{#if list.isLoading}
		<p class="mt-6 text-sm text-on-surface-variant">{$t('common.loading')}</p>
	{:else if list.error}
		<div
			class="mt-6 rounded-xl border border-outline-variant bg-surface-container-low p-4 text-sm text-on-surface-variant"
		>
			{$t('common.error')}
		</div>
	{:else if list.data?.length === 0}
		<div class="mt-6 rounded-xl border border-dashed border-outline-variant p-8 text-center">
			<p class="text-sm text-on-surface-variant">{$t('collections.empty')}</p>
		</div>
	{:else}
		<ul class="mt-4 space-y-3">
			{#each list.data as c (c.id)}
				<li class="flex items-center gap-3">
					<a
						href={resolve(`/collections/${c.id}`)}
						class="flex min-w-0 flex-1 items-center gap-3 rounded-xl border border-outline-variant bg-surface-container-low p-4 transition-colors hover:border-primary/50"
					>
						<span
							class="flex size-12 shrink-0 items-center justify-center rounded-lg"
							style={colorToStyle(c.color)}
						></span>
						<div class="min-w-0 flex-1">
							<p class="truncate text-sm font-medium">{c.name}</p>
							<p class="text-xs text-on-surface-variant">
								{$t('collections.receiptCount', { values: { count: c.receipt_count } })}
								{#if c.start_date || c.end_date}
									· {c.start_date ?? '…'} – {c.end_date ?? '…'}
								{/if}
							</p>
							{#if c.totals.length}
								<p class="mt-1 text-xs font-medium text-on-surface">
									{#each c.totals as total, i (total.currency)}
										{formatMoney(total.total, total.currency)}
										{#if i < c.totals.length - 1}
											·
										{/if}
									{/each}
								</p>
							{/if}
						</div>
						{#if c.active}
							<span
								class="shrink-0 rounded-full bg-primary-container px-2.5 py-1 text-xs font-medium text-on-primary-container"
								>{$t('collections.capturing')}</span
							>
						{/if}
					</a>
					<button
						type="button"
						role="switch"
						aria-checked={c.active}
						aria-label={translate(
							c.active ? 'collections.stopCapturing' : 'collections.startCapturing',
							{
								values: { name: c.name }
							}
						)}
						disabled={togglingId !== null}
						class="relative h-6 w-11 shrink-0 rounded-full transition-colors {c.active
							? 'bg-primary'
							: 'bg-surface-container-highest'}"
						onclick={() => toggleActive(c)}
					>
						<span
							class="absolute top-0.5 size-5 rounded-full bg-surface-container-lowest shadow transition-all {c.active
								? 'left-[22px]'
								: 'left-0.5'}"
						></span>
					</button>
					<button
						type="button"
						class="shrink-0 rounded-lg p-2 text-on-surface-variant hover:bg-error-container hover:text-on-error-container"
						aria-label={$t('common.remove')}
						title={$t('common.remove')}
						onclick={() => (confirmDelete = c)}
					>
						<Icon icon="trash" />
					</button>
				</li>
			{/each}
		</ul>
	{/if}
</section>

{#if dialogOpen}
	<div class="fixed inset-0 z-40 flex items-center justify-center p-4" role="presentation">
		<div class="absolute inset-0 bg-black/50" role="presentation" onclick={closeDialog}></div>
		<div
			role="dialog"
			aria-modal="true"
			aria-label={$t('collections.new')}
			class="relative w-full max-w-md rounded-xl bg-surface p-5 shadow-elevation-4"
		>
			<form
				onsubmit={(event) => {
					event.preventDefault();
					void submitForm();
				}}
			>
				<h2 class="text-lg font-semibold">{$t('collections.new')}</h2>

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
						onclick={closeDialog}
					>
						{$t('collections.cancel')}
					</button>
					<button
						type="submit"
						disabled={creating}
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
	open={confirmDelete !== null}
	title={$t('collections.deleteTitle')}
	body={$t('collections.deleteBody')}
	confirmLabel={$t('collections.delete')}
	busy={deleting}
	onConfirm={removeCollection}
	onCancel={() => (confirmDelete = null)}
/>
