<script lang="ts">
	import { createQuery } from '@tanstack/svelte-query';
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import { t, translate } from '$lib/i18n';
	import { api, ApiError } from '$lib/api/client';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	import { snackbar } from '$lib/ui/snackbar.svelte';
	import ConfirmDialog from '$lib/ui/ConfirmDialog.svelte';
	import Icon from '$lib/ui/Icon.svelte';
	import ReceiptEditor from '$lib/ui/ReceiptEditor.svelte';
	import ReceiptPhotos from '$lib/ui/ReceiptPhotos.svelte';
	import type { ReceiptWrite } from '$lib/types';

	let id = $derived(page.url.pathname.split('/').pop() ?? '');
	const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
	let verifyOpen = $state(false);
	let verifying = $state(false);
	let saving = $state(false);
	let editorRevision = $state(0);
	let savedRevision = $state(0);
	let editorDirty = $state(false);

	const detail = createQuery(
		() => ({
			queryKey: queryKeys.receipt(id),
			queryFn: () => api.getReceipt(id),
			enabled: uuidPattern.test(id)
		}),
		() => queryClient
	);

	let notFound = $derived(
		!uuidPattern.test(id) ||
			(detail.status === 'error' && detail.error instanceof ApiError && detail.error.status === 404)
	);
	let data = $derived(detail.data ?? null);
	let canVerify = $derived(data !== null && data.receipt.status === 'unverified');

	async function refresh(): Promise<void> {
		const fresh = await api.getReceipt(id);
		queryClient.setQueryData(queryKeys.receipt(id), fresh);
		editorRevision++;
		await queryClient.invalidateQueries({ queryKey: ['statistics'] });
	}

	async function saveOnly(write: ReceiptWrite) {
		saving = true;
		try {
			await api.updateReceipt(id, write);
			savedRevision++;
			snackbar.notify('success', translate('receipt.saved'));
			await refresh();
		} catch {
			snackbar.notify('error', translate('receipt.saveFailed'));
		} finally {
			saving = false;
		}
	}

	async function saveAndVerify(write: ReceiptWrite) {
		saving = true;
		try {
			await api.updateReceipt(id, write);
			savedRevision++;
			await api.verifyReceipt(id);
			snackbar.notify('success', translate('receipt.verifySuccess'));
			await refresh();
		} catch (error) {
			snackbar.notify(
				'error',
				error instanceof ApiError && error.status === 409
					? translate('receipt.verifyConflict')
					: translate('receipt.saveFailed')
			);
		} finally {
			saving = false;
		}
	}

	async function doVerify() {
		if (verifying) return;
		verifying = true;
		try {
			await api.verifyReceipt(id);
			snackbar.notify('success', translate('receipt.verifySuccess'));
			verifyOpen = false;
			await refresh();
		} catch (error) {
			snackbar.notify(
				'error',
				error instanceof ApiError && error.status === 409
					? translate('receipt.verifyConflict')
					: translate('receipt.verifyFailed')
			);
		} finally {
			verifying = false;
		}
	}
</script>

<svelte:head>
	<title>{$t('pages.receiptDetail.title', { values: { id } })} – {$t('app.name')}</title>
</svelte:head>

{#if detail.isLoading}
	<p class="text-sm text-on-surface-variant">{$t('common.loading')}</p>
{:else if notFound}
	<div class="rounded-xl border border-outline-variant bg-surface-container-low p-4 text-sm">
		<p class="font-medium">{$t('receipt.notFoundTitle')}</p>
		<p class="mt-1 text-on-surface-variant">{$t('receipt.notFoundBody')}</p>
		<a
			href={resolve('/receipts')}
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
	<section class="mx-auto min-w-0 max-w-6xl">
		<div class="flex items-center gap-3">
			<a
				href={resolve('/receipts')}
				class="rounded-lg p-2 text-on-surface-variant hover:bg-surface-container"
				aria-label={$t('common.back')}
			>
				<Icon icon="close" />
			</a>
			<h1 class="flex-1 truncate text-xl font-semibold">
				{data.receipt.merchant_name || $t('receipts.unknownVendor')}
			</h1>
			{#if canVerify}
				<button
					type="button"
					class="flex items-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-on-primary hover:opacity-90 disabled:opacity-50"
					disabled={editorDirty || saving || verifying}
					onclick={() => (verifyOpen = true)}
				>
					<Icon icon="check" />
					{$t('receipt.verify')}
				</button>
			{:else}
				<span
					class="rounded-full bg-primary-container px-3 py-1 text-xs font-medium text-on-primary-container"
				>
					{$t('receipt.verified')}
				</span>
			{/if}
		</div>

		<div class="mt-6 min-w-0 lg:grid lg:grid-cols-[minmax(20rem,0.8fr)_minmax(0,1.2fr)] lg:gap-8">
			{#if data.receipt.image_id !== null}
				<aside class="mb-6 min-w-0 lg:mb-0">
					<div class="min-w-0 lg:sticky lg:top-20">
						<ReceiptPhotos
							imageId={data.receipt.image_id}
							thumbnailPath={data.receipt.thumbnail_path ?? null}
							alt={data.receipt.merchant_name || $t('receipts.unknownVendor')}
						/>
					</div>
				</aside>
			{/if}

			<div class="min-w-0 {data.receipt.image_id === null ? 'lg:col-span-2' : ''}">
				<ReceiptEditor
					receipt={data.receipt}
					revision={editorRevision}
					{savedRevision}
					lineItems={data.line_items}
					taxes={data.taxes}
					collectionIds={data.collection_ids}
					busy={saving || verifying}
					onSave={saveOnly}
					onSaveAndVerify={canVerify ? saveAndVerify : undefined}
					onDirtyChange={(dirty) => (editorDirty = dirty)}
				/>
			</div>
		</div>
	</section>

	<ConfirmDialog
		open={verifyOpen}
		title={$t('receipt.verifyTitle')}
		body={$t('receipt.verifyBody')}
		confirmLabel={$t('receipt.verify')}
		busy={verifying}
		onConfirm={doVerify}
		onCancel={() => (verifyOpen = false)}
	/>
{/if}
