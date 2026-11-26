<script lang="ts">
	import './cases.css';
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { t } from '$lib/i18n';
	import { caseApi, type BenchmarkCase, type BenchmarkCaseSummary } from '$lib/api/benchmark-cases';
	import { ApiError } from '$lib/api/client';
	import { captureSessionActor, session } from '$lib/session-lifetime';
	import { sessionViewGuard } from '$lib/session-view';
	import type { ExtractionDraft } from '$lib/types';
	import ReceiptWorkspace from '$lib/ui/ReceiptWorkspace.svelte';
	import CaseExpected from './CaseExpected.svelte';
	import { expectedIssue } from './expected-validation';
	import CasePhotos from './CasePhotos.svelte';
	import CaseUpload from './CaseUpload.svelte';

	const actor = captureSessionActor();
	const current = sessionViewGuard();
	let cases = $state<BenchmarkCaseSummary[]>([]);
	let mode = $state<'list' | 'upload' | 'edit'>('list');
	let selected = $state<BenchmarkCase | null>(null);
	let name = $state('');
	let expected = $state<ExtractionDraft>({});
	let approved = $state<string | null>(null);
	let busy = $state(false);
	let error = $state('');
	let conflict = $state(false);
	let notice = $state('');
	let form = $state<HTMLFormElement>();
	const snapshot = $derived(JSON.stringify([name, expected]));
	const dirty = $derived(
		!!selected && snapshot !== JSON.stringify([selected.name, selected.expected])
	);
	const consent = $derived(approved === snapshot);
	function adopt(detail: BenchmarkCase) {
		selected = detail;
		name = detail.name;
		expected = structuredClone(detail.expected ?? {});
		approved = null;
		cases = [detail, ...cases.filter((row) => row.id !== detail.id)];
		mode = 'edit';
	}
	async function run(action: () => Promise<void>) {
		if (!actor || !current() || busy) return;
		busy = true;
		error = '';
		conflict = false;
		notice = '';
		try {
			await action();
		} catch (cause) {
			if (current()) {
				conflict = cause instanceof ApiError && cause.status === 409 && mode === 'edit';
				const statusErrors: Record<number, string> = {
					413: 'submission.limits',
					415: 'upload.errorUnsupported',
					422: 'benchmarkCase.invalid',
					503: 'upload.errorDb'
				};
				error = conflict
					? 'review.conflict'
					: cause instanceof ApiError
						? (statusErrors[cause.status] ?? 'common.error')
						: 'common.error';
			}
		} finally {
			if (current()) busy = false;
		}
	}
	function load() {
		return run(async () => {
			const list = await caseApi.list(actor!);
			if (current()) cases = list;
		});
	}
	function open(id: string) {
		return run(async () => {
			const item = await caseApi.get(id, actor!);
			if (current()) adopt(item);
		});
	}
	function create(name: string, files: File[]) {
		return run(async () => {
			const item = await caseApi.create(name, files, actor!);
			if (current()) adopt(item);
		});
	}
	function save(confirm: boolean) {
		if (!selected || !name.trim() || (confirm && !consent) || !form?.reportValidity()) return;
		const issue = expectedIssue(expected, confirm);
		if (issue) {
			error = issue;
			notice = '';
			return;
		}
		const id = selected.id;
		const update = {
			name: name.trim(),
			expected: $state.snapshot(expected),
			expected_revision: selected.revision,
			confirm
		};
		return run(async () => {
			const saved = await caseApi.save(id, update, actor!);
			if (current()) {
				adopt(saved);
				notice = confirm ? 'benchmarkCase.referenceSaved' : 'benchmarkCase.draftSaved';
			}
		});
	}
	function focusFeedback(node: HTMLElement) {
		node.focus();
	}
	function back() {
		mode = 'list';
		selected = null;
		notice = '';
		error = '';
		conflict = false;
	}
	onMount(() => {
		if ($session && $session !== 'loading') {
			if ($session.is_admin) void load();
			else void goto(resolve('/'));
		}
	});
</script>

<svelte:head><title>{$t('benchmarkCase.title')} · Vision Bill</title></svelte:head>
{#if actor && $session && $session !== 'loading' && $session.is_admin && $session.id === actor.userId}
	<section class="benchmark-cases mx-auto max-w-7xl space-y-5">
		<a class="text-primary underline" href={resolve('/benchmarks')}>{$t('nav.benchmarks')}</a>
		<h1 class="text-3xl font-semibold">{$t('benchmarkCase.title')}</h1>
		{#if error}<p role="alert" tabindex="-1" use:focusFeedback class="text-error">{$t(error)}</p>
			{#if conflict && selected}<button
					type="button"
					class="case-secondary"
					disabled={busy}
					onclick={() => open(selected!.id)}>{$t('review.discardReload')}</button
				>{:else if mode === 'list'}<button
					type="button"
					class="case-secondary"
					disabled={busy}
					onclick={load}>{$t('common.retry')}</button
				>{/if}
		{/if}
		{#if mode === 'list'}
			<p class="text-on-surface-variant">{$t('benchmarkCase.intro')}</p>
			<button
				type="button"
				class="case-primary"
				disabled={busy}
				onclick={() => {
					error = '';
					mode = 'upload';
				}}>{$t('benchmarkCase.new')}</button
			>
			{#if busy}<p role="status">{$t('common.loading')}</p>{:else if !cases.length && !error}<p>
					{$t('benchmarkCase.empty')}
				</p>{/if}
			<ul class="divide-y divide-outline-variant">
				{#each cases as entry (entry.id)}
					<li class="flex flex-wrap items-center justify-between gap-3 py-3">
						<button
							class="text-left font-semibold text-primary underline"
							type="button"
							disabled={busy}
							onclick={() => open(entry.id)}>{entry.name}</button
						>
						<span
							>{$t('benchmarkCase.photos', { values: { count: entry.photo_count } })} · {$t(
								entry.confirmed ? 'benchmarkCase.qualified' : 'benchmarkCase.draft'
							)}</span
						>
					</li>
				{/each}
			</ul>
		{:else}
			<button type="button" class="case-secondary" disabled={busy} onclick={back}
				>{$t(dirty ? 'benchmarkCase.discard' : 'benchmarkCase.back')}</button
			>
			{#if mode === 'upload'}<CaseUpload {busy} oncreate={create} />{:else if selected}
				{@const item = selected}
				<div class="flex flex-wrap items-baseline gap-3">
					<h2 class="text-2xl font-semibold">{selected.name}</h2>
					<span>{$t(selected.confirmed ? 'benchmarkCase.qualified' : 'benchmarkCase.draft')}</span>
				</div>
				{#if notice && selected.confirmed && !dirty}<p
						role="status"
						tabindex="-1"
						use:focusFeedback
					>
						{$t(notice)}
					</p>{/if}
				{#if selected.confirmed && !dirty}<a
						class="case-primary inline-block"
						href={resolve(`/benchmarks?case=${encodeURIComponent(selected.id)}`)}
						>{$t('benchmarkCase.use')}</a
					>{/if}
				<ReceiptWorkspace>
					{#snippet photo()}{#key item.id}<CasePhotos
								id={item.id}
								count={item.photo_count}
								{actor}
							/>{/key}{/snippet}
					<form
						bind:this={form}
						onsubmit={(e) => {
							e.preventDefault();
							if (dirty) void save(false);
						}}
					>
						<fieldset disabled={busy} class="space-y-4">
							<legend class="text-xl font-semibold">{$t('benchmarkCase.expected')}</legend>
							<p class="text-sm text-on-surface-variant">{$t('benchmarkCase.expectedHint')}</p>
							<label
								>{$t('benchmarkCase.name')}<input
									maxlength="160"
									required
									bind:value={name}
								/></label
							>
							<CaseExpected bind:expected />
							<button type="submit" class="case-secondary" disabled={!dirty || !name.trim()}
								>{$t('review.saveDraft')}</button
							>
							{#if notice && !selected.confirmed && !dirty}<p role="status">{$t(notice)}</p>{/if}
							{#if !selected.confirmed || dirty}
								<div class="case-panel space-y-3">
									<label class="case-check"
										><input
											type="checkbox"
											checked={consent}
											onchange={(e) => (approved = e.currentTarget.checked ? snapshot : null)}
										/>{$t('benchmarkCase.consent')}</label
									>
									<button
										type="button"
										class="case-primary"
										disabled={!consent || !name.trim()}
										onclick={() => save(true)}>{$t('benchmarkCase.confirm')}</button
									>
								</div>
							{/if}
						</fieldset>
					</form>
				</ReceiptWorkspace>
			{/if}
		{/if}
	</section>
{/if}
