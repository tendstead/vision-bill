<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { onDestroy } from 'svelte';
	import { resolve } from '$app/paths';
	import { createQuery } from '@tanstack/svelte-query';
	import { t, translate } from '$lib/i18n';
	import { api, ApiError } from '$lib/api/client';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	import { session } from '$lib/auth';
	import { captureSessionActor, isCurrentSessionActor } from '$lib/session-lifetime';
	import {
		prepareUpload,
		readPendingUpload,
		clearPendingUpload,
		matchPendingUpload,
		type PendingUpload
	} from '$lib/api/upload';
	import Icon from '$lib/ui/Icon.svelte';
	import SubmissionStatus from '$lib/ui/SubmissionStatus.svelte';
	import ExtractionDraftEditor from '$lib/ui/ExtractionDraftEditor.svelte';

	const ACCEPTED_TYPES = 'image/jpeg,image/png,image/webp,image/heic,image/heif,image/gif';

	let galleryInput: HTMLInputElement | undefined = $state();
	let cameraInput: HTMLInputElement | undefined = $state();

	let photos = $state<{ file: File; url: string }[]>([]);
	let file = $derived(photos[0]?.file ?? null);
	let jobId = $derived(page.url.searchParams.get('job'));
	const actor = captureSessionActor();
	let alive = true;
	onDestroy(() => {
		alive = false;
		photos.forEach((photo) => URL.revokeObjectURL(photo.url));
	});
	let dragging = $state(false);
	let uploading = $state(false);
	let bypassChoice = $state<boolean>();
	let recoveredUpload = $state.raw<PendingUpload | null>(null);
	let checkingRecovery = $state(false);
	let recoveryFailed = $state(false);
	let recoveryOwner = $state('');
	let recovering = $state(false);
	let uploadError = $state('');

	const uiConfig = createQuery(
		() => ({
			queryKey: queryKeys.uiConfig(),
			queryFn: () => api.getUiConfig(),
			staleTime: Infinity
		}),
		() => queryClient
	);

	const bypassReview = $derived(
		bypassChoice ??
			recoveredUpload?.options.bypass_review ??
			uiConfig.data?.bypass_review_default ??
			false
	);

	$effect(() => {
		if ($session && $session !== 'loading' && recoveryOwner !== $session.id) {
			recoveryOwner = $session.id;
			recovering = !!readPendingUpload($session.id);
			bypassChoice = undefined;
		}
	});

	function setPhotos(next: typeof photos) {
		photos = next;
		const selection = photos;
		const origin = jobId;
		recoveredUpload = null;
		checkingRecovery = false;
		recoveryFailed = false;
		if (origin || !selection.length || !isCurrentSessionActor(actor)) return;

		const sameSelection = () => alive && photos === selection && isCurrentSessionActor(actor);
		const current = () => sameSelection() && jobId === origin;
		checkingRecovery = true;
		void matchPendingUpload(
			actor.userId,
			selection.map((photo) => photo.file),
			() => {
				if (!current()) throw new DOMException('Upload selection changed', 'AbortError');
			}
		).then(
			(pending) => {
				if (!sameSelection()) return;
				if (current()) {
					recoveredUpload = pending;
					if (pending) recovering = true;
				} else {
					recoveryFailed = true;
				}
				checkingRecovery = false;
			},
			() => {
				if (!sameSelection()) return;
				recoveryFailed = true;
				checkingRecovery = false;
			}
		);
	}

	function addFiles(next: File[]) {
		if (uploading || !next.length) return;
		if (photos.length + next.length > 10 || next.some((file) => file.size > 20 * 1024 * 1024)) {
			uploadError = translate('submission.limits');
			return;
		}
		setPhotos([...photos, ...next.map((file) => ({ file, url: URL.createObjectURL(file) }))]);
		uploadError = '';
	}

	function removePhoto(index: number) {
		URL.revokeObjectURL(photos[index].url);
		setPhotos(photos.filter((_, i) => i !== index));
	}

	function movePhoto(index: number, direction: number) {
		const next = [...photos];
		[next[index], next[index + direction]] = [next[index + direction], next[index]];
		setPhotos(next);
	}

	function onFileChange(input: HTMLInputElement) {
		addFiles(Array.from(input.files ?? []));
		input.value = '';
	}

	function onDrop(event: DragEvent) {
		event.preventDefault();
		dragging = false;
		addFiles(Array.from(event.dataTransfer?.files ?? []));
	}

	function onBypassChange(event: Event) {
		const checkbox = event.currentTarget as HTMLInputElement;
		bypassChoice = checkbox.checked;
	}

	async function upload() {
		if (
			!alive ||
			jobId ||
			!file ||
			uploading ||
			checkingRecovery ||
			recoveryFailed ||
			!isCurrentSessionActor(actor)
		)
			return;
		const owner = actor.userId;
		const selection = photos;
		const current = () => alive && !jobId && photos === selection && isCurrentSessionActor(actor);
		uploading = true;
		uploadError = '';
		try {
			const files = selection.map((photo) => photo.file);
			const recovery = recoveredUpload;
			const options = { ...recovery?.options };
			if (bypassChoice !== undefined) options.bypass_review = bypassChoice;
			const pending =
				recovery && options.bypass_review === recovery.options.bypass_review
					? recovery
					: await prepareUpload(owner, files, options, () => {
							if (!current()) throw new DOMException('Upload view changed', 'AbortError');
						});
			if (!current()) return;
			recoveredUpload = pending;
			const result = await api.uploadImages(files, pending.options, pending.key, actor);
			if (!current()) return;
			// Treat a malformed successful response as ambiguous, retaining the key/options.
			if (!result.id) throw new ApiError(202, 'Response did not include a submission ID');
			clearPendingUpload(owner, pending.key);
			photos.forEach((photo) => URL.revokeObjectURL(photo.url));
			setPhotos([]);
			void queryClient.invalidateQueries({ queryKey: ['images'] });
			void queryClient.invalidateQueries({ queryKey: ['receipts'] });
			await goto(resolve(`/upload?job=${result.id}`));
		} catch (error) {
			if (!current()) return;
			recovering = !!readPendingUpload(owner);
			if (error instanceof ApiError && error.imageId) {
				await goto(resolve(`/upload?job=${error.imageId}`));
				return;
			}
			uploadError = error instanceof Error ? error.message : translate('upload.errorGeneric');
		} finally {
			if (alive && isCurrentSessionActor(actor)) uploading = false;
		}
	}

	function formatSize(bytes: number | undefined): string {
		if (bytes === undefined || bytes === null) {
			return '';
		}
		if (bytes < 1024 * 1024) {
			return `${(bytes / 1024).toFixed(0)} KB`;
		}
		return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
	}
</script>

<svelte:head>
	<title>{$t('pages.upload.title')} – {$t('app.name')}</title>
</svelte:head>

<section
	class="mx-auto min-w-0 {jobId && page.url.searchParams.get('draft') === '1'
		? 'max-w-6xl'
		: 'max-w-2xl'}"
>
	<h1 class="text-2xl font-semibold">{$t('pages.upload.title')}</h1>
	{#if jobId}
		<SubmissionStatus id={jobId} editingDraft={page.url.searchParams.get('draft') === '1'} />
		{#if page.url.searchParams.get('draft') === '1'}<ExtractionDraftEditor id={jobId} />{/if}
	{:else}
		<p class="mt-3 text-sm text-on-surface-variant">{$t('submission.captureHint')}</p>

		{#if recovering}
			<p class="mt-4 rounded-xl border border-outline-variant p-4 text-sm" role="status">
				{$t('submission.recovery')}
			</p>
		{/if}

		<p class="mt-3 text-sm text-on-surface-variant">{$t('submission.limits')}</p>
		<p class="mt-2 text-sm font-medium" role="status">
			{$t('receiptWorkspace.selection', { values: { count: photos.length } })}
		</p>
		<!-- Dropzone / preview -->
		{#each photos as photo, index (photo.url)}
			<div
				class="mt-4 overflow-hidden rounded-xl border border-outline-variant bg-surface-container-low"
			>
				<img
					src={photo.url}
					alt={photo.file.name}
					class="max-h-80 w-full bg-surface-container object-contain"
				/>
				<div class="flex flex-wrap items-center justify-between gap-3 p-3">
					<div class="min-w-0">
						<p class="truncate text-sm font-medium">{index + 1}. {photo.file.name}</p>
						<p class="text-xs text-on-surface-variant">{formatSize(photo.file.size)}</p>
					</div>
					<button
						type="button"
						class="rounded-lg p-2 disabled:opacity-30"
						aria-label={$t('submission.moveUp')}
						disabled={uploading || index === 0}
						onclick={() => movePhoto(index, -1)}>↑</button
					>
					<button
						type="button"
						class="rounded-lg p-2 disabled:opacity-30"
						aria-label={$t('submission.moveDown')}
						disabled={uploading || index === photos.length - 1}
						onclick={() => movePhoto(index, 1)}>↓</button
					>
					<button
						type="button"
						class="flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm text-on-surface-variant hover:bg-error-container hover:text-on-error-container"
						disabled={uploading}
						onclick={() => removePhoto(index)}
					>
						<Icon icon="trash" />
						{$t('common.remove')}
					</button>
				</div>
			</div>
		{/each}
		<div
			class="mt-4 flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed border-outline-variant bg-surface-container-low p-8 text-center
				{dragging ? 'border-primary bg-primary-container/20' : ''}"
			role="group"
			ondragover={(e) => {
				e.preventDefault();
				dragging = true;
			}}
			ondragleave={() => (dragging = false)}
			ondrop={onDrop}
		>
			<Icon icon="upload" />
			<p class="text-sm text-on-surface-variant">{$t('upload.dropHint')}</p>
			<div class="flex flex-wrap items-center justify-center gap-2">
				<button
					type="button"
					class="flex items-center gap-2 rounded-lg bg-action px-4 py-2.5 text-sm font-medium text-on-primary hover:opacity-90"
					onclick={() => galleryInput?.click()}
					disabled={uploading}
				>
					<Icon icon="upload" />
					{$t('upload.chooseFile')}
				</button>
				<button
					type="button"
					class="flex items-center gap-2 rounded-lg bg-secondary-container px-4 py-2.5 text-sm font-medium text-on-secondary-container hover:opacity-90"
					onclick={() => cameraInput?.click()}
					disabled={uploading}
				>
					<Icon icon="camera" />
					{$t('upload.takePhoto')}
				</button>
			</div>
			<input
				bind:this={galleryInput}
				multiple
				type="file"
				accept={ACCEPTED_TYPES}
				class="hidden"
				aria-hidden="true"
				tabindex={-1}
				onchange={(e) => onFileChange(e.currentTarget as HTMLInputElement)}
			/>
			<input
				bind:this={cameraInput}
				type="file"
				accept={ACCEPTED_TYPES}
				capture="environment"
				class="hidden"
				aria-hidden="true"
				tabindex={-1}
				onchange={(e) => onFileChange(e.currentTarget as HTMLInputElement)}
			/>
		</div>

		{#if recoveryFailed}
			<div class="mt-3 flex items-center gap-2 text-sm text-error" role="alert">
				<Icon icon="alert" />
				<p>{$t('upload.recoveryReadError')}</p>
				<button
					type="button"
					class="rounded-lg px-2 py-1 font-medium text-primary underline"
					onclick={() => setPhotos([...photos])}
				>
					{$t('common.retry')}
				</button>
			</div>
		{/if}

		{#if uploadError}
			<p class="mt-3 flex items-center gap-2 text-sm text-error" role="alert">
				<Icon icon="alert" />
				{uploadError}
			</p>
		{/if}

		<div class="mt-4 rounded-xl border border-outline-variant bg-surface-container-low p-4">
			<label class="flex cursor-pointer items-start gap-3">
				<input
					type="checkbox"
					class="mt-1 size-4"
					checked={bypassReview}
					disabled={uploading || checkingRecovery || recoveryFailed}
					onchange={onBypassChange}
				/>
				<span>
					<span class="block text-sm font-medium">{$t('upload.bypassLabel')}</span>
					<span class="block text-xs text-on-surface-variant">{$t('upload.bypassHint')}</span>
				</span>
			</label>
		</div>

		<button
			type="button"
			class="mt-4 w-full rounded-lg bg-action px-4 py-3 text-sm font-medium text-on-primary hover:opacity-90 disabled:opacity-50"
			disabled={!file ||
				uploading ||
				checkingRecovery ||
				recoveryFailed ||
				!$session ||
				$session === 'loading'}
			onclick={upload}
		>
			{#if uploading}
				<span class="inline-flex items-center gap-2">
					<span class="animate-spin"><Icon icon="refresh" /></span>
					{$t('upload.uploading', { values: { file: file ? file.name : '' } })}
				</span>
			{:else}
				{$t('upload.upload')}
			{/if}
		</button>

		{#if uploading}
			<div
				class="mt-3 rounded-xl border border-primary/30 bg-primary-container/40 p-4"
				role="status"
				aria-live="polite"
			>
				<p class="text-sm font-medium text-on-primary-container">{$t('upload.processingTitle')}</p>
				<p class="mt-1 text-xs text-on-primary-container/80">{$t('upload.processingBody')}</p>
				<div
					class="mt-3 h-2 overflow-hidden rounded-full bg-primary/15"
					role="progressbar"
					aria-label={$t('upload.processingTitle')}
				>
					<div class="analysis-progress h-full rounded-full bg-primary"></div>
				</div>
			</div>
		{/if}
	{/if}
</section>

<style>
	.analysis-progress {
		width: 35%;
		animation: analysis-progress 1.6s ease-in-out infinite;
	}

	@keyframes analysis-progress {
		from {
			transform: translateX(-110%);
		}
		to {
			transform: translateX(320%);
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.analysis-progress {
			animation: none;
			width: 100%;
			opacity: 0.55;
		}
	}
</style>
