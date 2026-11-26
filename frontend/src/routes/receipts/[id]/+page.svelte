<script lang="ts">
	import { tick } from 'svelte';
	import { createQuery } from '@tanstack/svelte-query';
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import { t, translate } from '$lib/i18n';
	import { session } from '$lib/auth';
	import { api, ApiError } from '$lib/api/client';
	import { captureSessionActor, isCurrentSessionActor } from '$lib/session-lifetime';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	import { snackbar } from '$lib/ui/snackbar.svelte';
	import ConfirmDialog from '$lib/ui/ConfirmDialog.svelte';
	import Icon from '$lib/ui/Icon.svelte';
	import ReceiptEditor from '$lib/ui/ReceiptEditor.svelte';
	import ReceiptPhotos from '$lib/ui/ReceiptPhotos.svelte';
	import type {
		ReceiptWrite,
		ReceiptRow,
		ReviewFinding,
		ReviewPreview,
		ReviewRequest,
		SourceRef
	} from '$lib/types';
	import ReviewFindings from '$lib/ui/ReviewFindings.svelte';
	import ReceiptWorkspace from '$lib/ui/ReceiptWorkspace.svelte';

	let id = $derived(page.url.pathname.split('/').pop() ?? '');
	const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
	type ReviewVersion = Pick<ReceiptRow, 'operation_revision' | 'review_revision'>;
	let renderedVersion = $state<(ReviewVersion & { id: string }) | null>(null);
	let previewing = $state(false);
	let humanPreview = $state<ReviewPreview | null>(null);
	let humanAcknowledgements = $state<ReviewRequest['acknowledgements']>([]);
	let verifying = $state(false);
	let saving = $state(false);
	let editorRevision = $state(0);
	let savedRevision = $state(0);
	let editorDirty = $state(false);
	let childPending = $state(false);
	let reviewMessage = $state('');
	let needsReload = $state(false);
	let selectedSources = $state<SourceRef[] | null>(null);
	let workspaceView = $state('data');
	const actor = captureSessionActor();
	let activeView: { id: string } | null = null;
	const isCurrentView = (view: typeof activeView) =>
		view !== null && activeView === view && view.id === id && isCurrentSessionActor(actor);
	$effect(() => {
		const view = { id };
		activeView = view;
		editorRevision = 0;
		savedRevision = 0;
		saving = false;
		verifying = false;
		previewing = false;
		humanPreview = null;
		humanAcknowledgements = [];
		selectedSources = null;
		editorDirty = false;
		childPending = false;
		reviewMessage = '';
		needsReload = false;
		return () => {
			if (activeView === view) activeView = null;
		};
	});

	const detail = createQuery(
		() => ({
			queryKey: queryKeys.receipt(id),
			queryFn: () => api.getReceipt(id),
			enabled: uuidPattern.test(id)
		}),
		() => queryClient
	);

	async function copyId() {
		const view = activeView;
		try {
			await navigator.clipboard.writeText(id);
			if (isCurrentView(view)) snackbar.notify('success', translate('benchmarkForm.copied'));
		} catch {
			if (isCurrentView(view)) snackbar.notify('info', translate('benchmarkForm.copyManually'));
		}
	}
	let notFound = $derived(
		!uuidPattern.test(id) ||
			(detail.status === 'error' && detail.error instanceof ApiError && detail.error.status === 404)
	);
	let data = $derived(detail.data ?? null);
	let enrichment = $derived(data?.classification_enrichment);
	const sourceJob = createQuery(
		() => ({
			queryKey: queryKeys.image(data?.receipt.image_id ?? ''),
			queryFn: () => api.getImage(data!.receipt.image_id!),
			enabled: !!data?.receipt.image_id
		}),
		() => queryClient
	);
	let canVerify = $derived(data !== null && data.receipt.status === 'unverified');
	let canReviewAsHuman = $derived(data?.can_rereview ?? false);
	let editorStale = $derived(
		renderedVersion !== null &&
			(renderedVersion.id !== id || !sameVersion(renderedVersion, data?.receipt))
	);
	let reviewBlocked = $derived(
		renderedVersion === null ||
			editorStale ||
			editorDirty ||
			saving ||
			verifying ||
			needsReload ||
			childPending
	);
	function sameVersion(left: ReviewVersion, right: ReviewVersion | null | undefined) {
		return (
			left.operation_revision === right?.operation_revision &&
			left.review_revision === right.review_revision
		);
	}
	function matchesReview(version: ReviewVersion) {
		return (
			renderedVersion?.id === id &&
			sameVersion(version, data?.receipt) &&
			sameVersion(version, renderedVersion)
		);
	}
	$effect(() => {
		if (humanPreview && !matchesReview(humanPreview)) closeReview();
	});

	function findingsReady(findings: ReviewFinding[], comments: ReviewRequest['acknowledgements']) {
		return findings.every(
			(finding) =>
				finding.severity === 'info' ||
				(finding.severity === 'review' &&
					comments.some((comment) => comment.finding_id === finding.id && comment.comment.trim()))
		);
	}
	let humanReviewReady = $derived(
		!!humanPreview && findingsReady(humanPreview.findings, humanAcknowledgements)
	);
	function reviewBody(receipt: ReviewVersion, comments: ReviewRequest['acknowledgements']) {
		return {
			expected_operation_revision: receipt.operation_revision,
			expected_review_revision: receipt.review_revision,
			acknowledgements: comments
		};
	}
	async function refresh(requestId = id, view = activeView): Promise<void> {
		if (!isCurrentView(view)) return;
		const fresh = await api.getReceipt(requestId);
		if (!isCurrentView(view)) return;
		queryClient.setQueryData(queryKeys.receipt(requestId), fresh);
		editorRevision++;
		needsReload = false;
		void queryClient.invalidateQueries({ queryKey: ['statistics'] });
	}
	async function discardReload() {
		const requestId = id;
		const view = activeView;
		if (!isCurrentView(view)) return;
		saving = true;
		try {
			await refresh(requestId, view);
			if (!isCurrentView(view)) return;
			reviewMessage = '';
		} catch {
			if (!isCurrentView(view)) return;
			reviewMessage = translate('receipt.saveFailed');
		} finally {
			if (isCurrentView(view)) saving = false;
		}
	}
	async function save(write: ReceiptWrite, verify: boolean) {
		if (saving || needsReload || editorStale || childPending) return;
		const requestId = id;
		const view = activeView;
		if (!isCurrentView(view)) return;
		saving = true;
		reviewMessage = '';
		let saved: ReceiptRow | null = null;
		try {
			saved = await api.updateReceipt(requestId, write);
			if (!isCurrentView(view)) return;
			savedRevision++;
			needsReload = true;
			// Only a full detail response may advance the cache's review revisions.
			void queryClient.invalidateQueries({
				queryKey: queryKeys.receipt(requestId),
				exact: true,
				refetchType: 'none'
			});
			snackbar.notify('success', translate('receipt.saved'));
			await refresh(requestId, view);
			if (!isCurrentView(view)) return;
			if (verify) {
				await tick();
				if (!isCurrentView(view)) return;
				if (!matchesReview(saved)) {
					reviewMessage = translate('review.conflict');
					return;
				}
				saving = false;
				await openReviewPreview(true);
			}
		} catch (error) {
			if (!isCurrentView(view)) return;
			reviewMessage =
				error instanceof ApiError && error.status === 409
					? translate('review.conflict')
					: translate(saved ? 'receipt.verifyFailed' : 'receipt.saveFailed');
			snackbar.notify('error', reviewMessage);
		} finally {
			if (isCurrentView(view)) saving = false;
		}
	}
	function saveOnly(write: ReceiptWrite) {
		return save(write, false);
	}
	function saveAndVerify(write: ReceiptWrite) {
		return save(write, true);
	}
	function closeReview() {
		humanPreview = null;
		humanAcknowledgements = [];
	}
	async function openReviewPreview(verifyIfClean = false) {
		if ((!canReviewAsHuman && !canVerify) || previewing || reviewBlocked || !renderedVersion)
			return;
		const version = { ...renderedVersion };
		const requestId = id;
		const view = activeView;
		if (!isCurrentView(view)) return;
		previewing = true;
		reviewMessage = '';
		try {
			const preview = await api.previewHumanReview(requestId);
			if (!isCurrentView(view)) return;
			if (!sameVersion(version, preview) || !matchesReview(version)) {
				reviewMessage = translate('review.conflict');
				return;
			}
			if (reviewBlocked) return;
			if (verifyIfClean && findingsReady(preview.findings, [])) {
				await submitReview(preview, []);
			} else {
				humanPreview = preview;
			}
		} catch (error) {
			if (!isCurrentView(view)) return;
			reviewMessage =
				error instanceof ApiError && error.status === 409
					? translate('review.conflict')
					: translate('receipt.verifyFailed');
		} finally {
			if (isCurrentView(view)) previewing = false;
		}
	}
	async function doVerify() {
		if (humanPreview && humanReviewReady)
			await submitReview(humanPreview, [...humanAcknowledgements]);
	}
	async function submitReview(preview: ReviewPreview, comments: ReviewRequest['acknowledgements']) {
		const requestId = id;
		const view = activeView;
		if (!isCurrentView(view) || reviewBlocked) return;
		if (!matchesReview(preview)) {
			closeReview();
			reviewMessage = translate('review.conflict');
			return;
		}
		const request = reviewBody(preview, comments);
		verifying = true;
		try {
			await api.verifyReceipt(requestId, request);
			if (!isCurrentView(view)) return;
			snackbar.notify('success', translate('receipt.verifySuccess'));
			closeReview();
			await refresh(requestId, view);
		} catch (error) {
			if (!isCurrentView(view)) return;
			closeReview();
			reviewMessage =
				error instanceof ApiError && error.status === 409
					? translate('review.conflict')
					: translate('receipt.verifyFailed');
			snackbar.notify('error', reviewMessage);
		} finally {
			if (isCurrentView(view)) verifying = false;
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
			class="mt-3 inline-block rounded-lg bg-action px-4 py-2 text-on-primary hover:opacity-90"
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
				{data.receipt.merchant_display_name ||
					data.receipt.merchant_name ||
					$t('receipts.unknownVendor')}
			</h1>
			{#if canVerify || canReviewAsHuman}
				<button
					type="button"
					class="flex items-center gap-2 rounded-lg bg-action px-4 py-2.5 text-sm font-medium text-on-primary hover:opacity-90 disabled:opacity-50"
					disabled={reviewBlocked || previewing}
					onclick={() => openReviewPreview()}
				>
					<Icon icon="check" />
					{$t(canReviewAsHuman ? 'receipt.humanReview' : 'receipt.verify')}
				</button>
			{/if}
			{#if !canVerify}
				<span
					class="rounded-full bg-primary-container px-3 py-1 text-xs font-medium text-on-primary-container"
				>
					{$t('receipt.verified')}
				</span>
			{/if}
		</div>

		<div class="my-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-on-surface-variant">
			<span>{$t('benchmarkForm.receiptId')} <code class="select-all break-all">{id}</code></span>
			<button type="button" class="text-primary underline" onclick={copyId}
				>{$t('benchmarkForm.copyId')}</button
			>
			{#if $session && $session !== 'loading' && $session.is_admin}
				<a
					class="text-primary underline"
					href={resolve(`/benchmarks?receipt=${encodeURIComponent(id)}`)}
					>{$t('benchmarkForm.useReceipt')}</a
				>
			{/if}
		</div>

		{#if reviewMessage}<p class="my-3 text-error" role="alert">{reviewMessage}</p>{/if}
		{#if enrichment?.state === 'unavailable' || (enrichment?.state === 'skipped' && enrichment.reason_code === 'classification_budget_exhausted' && enrichment.missing_fields.length > 0)}
			<p class="my-3 text-sm text-on-surface-variant">{$t('receipt.optionalLabelsUnavailable')}</p>
		{/if}
		{#if reviewMessage || needsReload || editorStale}<button
				type="button"
				disabled={saving || verifying}
				onclick={discardReload}
				class="my-3 text-primary underline">{$t('review.discardReload')}</button
			>{/if}
		<ReviewFindings
			onField={async (path) => {
				workspaceView = 'data';
				await tick();
				const fields: Record<string, string> = {
					merchant_name: 'merchant',
					merchant_address: 'address',
					receipt_number: 'number',
					payment_method: 'payment'
				};
				const item = /^line_items[.[]([0-9]+)\]?[.]([^.]*)$/.exec(path);
				const tax = /^taxes[.[]([0-9]+)\]?[.]([^.]*)$/.exec(path);
				const target = item
					? `li-${item[1]}-${({ quantity: 'qty', unit_price: 'unit', total_price: 'total', spending_category: 'category' } as Record<string, string>)[item[2]] ?? 'description'}`
					: tax
						? `tax-${tax[1]}-${tax[2]}`
						: `re-${fields[path] ?? path}`;
				document.getElementById(target)?.focus();
			}}
			findings={data.receipt.findings}
			revision={data.receipt.review_revision}
			editable={false}
			onSource={(sources) => (selectedSources = [...sources])}
		/>
		<ReceiptWorkspace
			hasPhoto={data.receipt.image_id !== null}
			bind:active={workspaceView}
			showReceipt={selectedSources}
		>
			{#snippet photo()}
				{#if data.receipt.image_id !== null}
					<ReceiptPhotos
						imageId={data.receipt.image_id}
						alt={data.receipt.merchant_display_name ||
							data.receipt.merchant_name ||
							$t('receipts.unknownVendor')}
						views={sourceJob.data?.source_views ?? []}
						sources={selectedSources}
					/>
				{/if}
			{/snippet}
			<div class="min-w-0 {data.receipt.image_id === null ? 'lg:col-span-2' : ''}">
				<ReceiptEditor
					receipt={data.receipt}
					revision={editorRevision}
					{savedRevision}
					lineItems={data.line_items}
					taxes={data.taxes}
					collectionIds={data.collection_ids}
					busy={saving || verifying || needsReload || editorStale}
					onSave={saveOnly}
					onSaveAndVerify={canVerify ? saveAndVerify : undefined}
					onDirtyChange={(dirty) => (editorDirty = dirty)}
					onSnapshotChange={(version) => (renderedVersion = version)}
					onSource={(sources) => (selectedSources = [...sources])}
					onPendingChange={(pending) => (childPending = pending)}
				/>
			</div>
		</ReceiptWorkspace>
	</section>

	<ConfirmDialog
		open={humanPreview !== null && matchesReview(humanPreview)}
		title={$t(canReviewAsHuman ? 'receipt.humanReviewTitle' : 'receipt.verifyTitle')}
		body={$t(canReviewAsHuman ? 'receipt.humanReviewBody' : 'receipt.verifyBody')}
		confirmLabel={$t(canReviewAsHuman ? 'receipt.humanReview' : 'receipt.verify')}
		busy={verifying}
		confirmDisabled={reviewBlocked || !humanReviewReady}
		onConfirm={doVerify}
		onCancel={closeReview}
	>
		{#if humanPreview && humanPreview.findings.length}
			<ReviewFindings
				findings={humanPreview.findings}
				revision={humanPreview.operation_revision}
				onChange={(value) => (humanAcknowledgements = value)}
			/>
		{/if}
	</ConfirmDialog>
{/if}
