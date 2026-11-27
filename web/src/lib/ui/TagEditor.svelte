<script lang="ts">
	import { t } from '$lib/i18n';
	import { api } from '$lib/api/client';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	import { snackbar } from '$lib/ui/snackbar.svelte';
	import Icon from './Icon.svelte';
	import { buildTagRows, moveActiveIndex, normalizeTag, type TagRow } from './tag-editor-logic';

	let {
		value,
		tagOptions,
		id
	}: {
		/** The line item's tags (standard + suggested). Mutated in place. */
		value: string[];
		/** The tag vocabulary loaded from the database. */
		tagOptions: string[];
		/** Unique prefix used to build stable input ids. */
		id: string;
	} = $props();

	const labelClass = 'mb-1 block text-xs font-medium text-on-surface-variant';
	let listId = $derived(`${id}-tags-listbox`);

	let open = $state(false);
	let query = $state('');
	let creating = $state(false);
	let wasOpen = $state(false);
	/** Keyboard highlight offset within `selectableRows` (−1 = nothing). */
	let activeOffset = $state(-1);

	let boxEl: HTMLDivElement | undefined = $state();
	let inlineInputEl: HTMLInputElement | undefined = $state();
	let ddSearchEl: HTMLInputElement | undefined = $state();

	function isStandard(tag: string): boolean {
		const lower = tag.toLowerCase();
		return tagOptions.some((option) => option.toLowerCase() === lower);
	}

	// `value` is mutated in place (ReceiptEditor relies on the stable array
	// reference for its dirty check). Svelte 5 only tracks the prop reference,
	// so every mutation bumps `revision` to invalidate the derived values.
	let revision = $state(0);

	let standardTags = $derived.by(() => {
		void revision;
		return value.filter((tag) => isStandard(tag));
	});
	let suggestedTags = $derived.by(() => {
		void revision;
		return value.filter((tag) => !isStandard(tag));
	});

	/** Replace the standard tags while preserving any suggested ones. */
	function setStandardTags(nextStandard: string[]): void {
		const next = [...nextStandard, ...suggestedTags];
		value.length = 0;
		value.push(...next);
		revision++;
	}

	function hasStandardTag(tag: string): boolean {
		return standardTags.some((entry) => entry.toLowerCase() === tag.toLowerCase());
	}

	function toggleStandardTag(tag: string): void {
		const next = hasStandardTag(tag)
			? standardTags.filter((entry) => entry.toLowerCase() !== tag.toLowerCase())
			: [...standardTags, tag];
		setStandardTags(next);
	}

	function removeStandardTag(tag: string): void {
		setStandardTags(standardTags.filter((entry) => entry.toLowerCase() !== tag.toLowerCase()));
	}

	function removeSuggested(tag: string): void {
		const next = value.filter((entry) => entry !== tag);
		value.length = 0;
		value.push(...next);
		revision++;
	}

	async function refreshVocabulary(): Promise<void> {
		await queryClient.invalidateQueries({ queryKey: queryKeys.tags() });
	}

	/** Promote a suggested (non-vocabulary) tag into the standard vocabulary. */
	async function keepSuggested(tag: string): Promise<void> {
		if (creating) return;
		creating = true;
		try {
			await api.createTag(tag);
			await refreshVocabulary();
		} catch {
			snackbar.notify('error', $t('editor.tagsCreateFailed'));
		} finally {
			creating = false;
		}
	}

	/** Create the tag currently typed into the search and attach it to the line. */
	async function createFromQuery(): Promise<void> {
		const name = normalizeTag(query);
		if (name === '' || creating) return;
		creating = true;
		try {
			const { name: created } = await api.createTag(name);
			if (!value.includes(created)) {
				value.push(created);
				revision++;
			}
			query = '';
			await refreshVocabulary();
		} catch {
			snackbar.notify('error', $t('editor.tagsCreateFailed'));
		} finally {
			creating = false;
		}
	}

	// Computed synchronously (plain function, not $derived): the keyboard
	// handlers read it inside the same tick they update `activeOffset`, and a
	// derived here would be re-evaluated by the reactivity scheduler between
	// two synchronous key events, clobbering the highlight.
	function computeSelectableRows(): Array<Extract<TagRow, { kind: 'tag' | 'create' }>> {
		return buildTagRows(tagOptions, query, standardTags).filter(
			(row): row is Extract<TagRow, { kind: 'tag' | 'create' }> =>
				row.kind === 'tag' || row.kind === 'create'
		);
	}

	let rows = $derived(buildTagRows(tagOptions, query, standardTags));

	function activateRow(row: Extract<TagRow, { kind: 'tag' | 'create' }>): void {
		if (row.kind === 'tag') {
			toggleStandardTag(row.tag);
		} else {
			void createFromQuery();
		}
	}

	function openDropdown(): void {
		if (open) return;
		open = true;
		// Highlight the first row immediately (no-op when the list is empty).
		activeOffset = 0;
	}

	function closeDropdown(refocus: boolean): void {
		open = false;
		query = '';
		activeOffset = -1;
		if (refocus) {
			void inlineInputEl?.focus();
		}
	}

	// When the dropdown opens, move focus to its search field (the list is right below it).
	$effect(() => {
		if (open && !wasOpen) {
			void ddSearchEl?.focus();
		}
		wasOpen = open;
	});

	// Close on outside click (the dropdown is nested inside boxEl, so only true
	// outside clicks close it).
	$effect(() => {
		if (!open) return;
		function onPointerDown(event: PointerEvent): void {
			if (boxEl && !boxEl.contains(event.target as Node)) {
				closeDropdown(true);
			}
		}
		document.addEventListener('pointerdown', onPointerDown);
		return () => document.removeEventListener('pointerdown', onPointerDown);
	});

	function onKeydown(event: KeyboardEvent): void {
		const selectable = computeSelectableRows();
		switch (event.key) {
			case 'ArrowDown':
				event.preventDefault();
				if (selectable.length > 0) {
					activeOffset = moveActiveIndex(selectable.length, activeOffset, 1);
				}
				break;
			case 'ArrowUp':
				event.preventDefault();
				if (selectable.length > 0) {
					activeOffset = moveActiveIndex(selectable.length, activeOffset, -1);
				}
				break;
			case 'Enter':
				if (!open) {
					event.preventDefault();
					openDropdown();
					return;
				}
				if (activeOffset >= 0 && activeOffset < selectable.length) {
					event.preventDefault();
					activateRow(selectable[activeOffset]);
				}
				break;
			case 'Escape':
				if (open) {
					event.preventDefault();
					closeDropdown(true);
				}
				break;
			case 'Backspace':
				if (query === '' && value.length > 0) {
					const last = value[value.length - 1];
					if (isStandard(last)) {
						removeStandardTag(last);
					} else {
						removeSuggested(last);
					}
				}
				break;
		}
	}
</script>

<div>
	<span id="{id}-tags-label" class={labelClass}>{$t('editor.tags')}</span>
	<div
		bind:this={boxEl}
		class="relative"
		role="button"
		tabindex="0"
		aria-label={$t('editor.tags')}
		onclick={openDropdown}
		onkeydown={(event) => {
			if (event.key === 'Enter' || event.key === ' ') {
				event.preventDefault();
				openDropdown();
			}
		}}
	>
		<div
			class="flex min-h-11 cursor-text flex-wrap items-center gap-1.5 rounded-lg border border-outline-variant bg-surface-container-lowest px-2 py-1.5 focus-within:border-primary focus-within:ring-1 focus-within:ring-primary"
		>
			{#each standardTags as tag (tag)}
				<span
					class="inline-flex items-center gap-1 rounded-full bg-primary-container py-1 pl-3 pr-1 text-sm font-medium text-on-primary-container"
				>
					{tag}
					<button
						type="button"
						class="flex size-5 items-center justify-center rounded-full text-xs hover:bg-on-primary-container/10 max-sm:min-h-11 max-sm:min-w-11"
						aria-label={$t('editor.tagsRemove', { values: { tag } })}
						onclick={(event) => {
							event.stopPropagation();
							removeStandardTag(tag);
						}}
					>
						✕
					</button>
				</span>
			{/each}
			{#each suggestedTags as tag (tag)}
				<span
					class="inline-flex items-center gap-1.5 rounded-full border border-warning/50 bg-warning-container py-1 pl-3 pr-1 text-sm font-medium text-on-warning-container"
				>
					<span
						class="rounded-full bg-warning px-1.5 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wide text-on-warning"
					>
						{$t('editor.tagsSuggestedBadge')}
					</span>
					{tag}
					<button
						type="button"
						class="flex size-5 items-center justify-center rounded-full text-xs font-bold hover:bg-on-warning-container/10 max-sm:min-h-11 max-sm:min-w-11"
						title={$t('editor.tagsKeep', { values: { tag } })}
						aria-label={$t('editor.tagsKeep', { values: { tag } })}
						disabled={creating}
						onclick={(event) => {
							event.stopPropagation();
							void keepSuggested(tag);
						}}
					>
						✓
					</button>
					<button
						type="button"
						class="flex size-5 items-center justify-center rounded-full text-xs hover:bg-error-container hover:text-on-error-container max-sm:min-h-11 max-sm:min-w-11"
						title={$t('editor.tagsRemove', { values: { tag } })}
						aria-label={$t('editor.tagsRemove', { values: { tag } })}
						disabled={creating}
						onclick={(event) => {
							event.stopPropagation();
							removeSuggested(tag);
						}}
					>
						✕
					</button>
				</span>
			{/each}
			<input
				bind:this={inlineInputEl}
				role="combobox"
				aria-expanded={open}
				aria-controls={open ? listId : undefined}
				aria-activedescendant={open &&
				activeOffset >= 0 &&
				activeOffset < computeSelectableRows().length
					? `${listId}-opt-${activeOffset}`
					: undefined}
				aria-label={$t('editor.tags')}
				class="min-w-32 flex-1 bg-transparent text-sm text-on-surface outline-none"
				placeholder={$t('editor.tagsPlaceholder')}
				value={query}
				oninput={(event) => (query = (event.currentTarget as HTMLInputElement).value)}
				onkeydown={onKeydown}
			/>
		</div>

		{#if open}
			<div
				class="absolute left-0 right-0 top-full z-10 mt-1 overflow-hidden rounded-lg border border-outline-variant bg-surface-container-lowest shadow-elevation-2"
			>
				<div
					class="flex items-center gap-2 border-b border-outline-variant px-3 py-2.5 text-on-surface-variant"
				>
					<Icon icon="search" />
					<input
						bind:this={ddSearchEl}
						class="flex-1 bg-transparent text-sm text-on-surface outline-none"
						aria-label={$t('editor.tagsSearch')}
						placeholder={$t('editor.tagsSearch')}
						value={query}
						oninput={(event) => (query = (event.currentTarget as HTMLInputElement).value)}
						onkeydown={onKeydown}
					/>
				</div>
				<ul id={listId} role="listbox" class="max-h-52 overflow-y-auto p-1">
					{#each rows as row (row.kind === 'tag' ? `tag-${row.tag}` : row.kind === 'create' ? `create-${row.tag}` : row.kind === 'nomatch' ? `nomatch-${row.query}` : 'empty')}
						{#if row.kind === 'tag'}
							<li
								id="{listId}-opt-{row.optionIndex}"
								role="option"
								aria-selected={row.selected}
								class="flex min-h-9 cursor-pointer items-center justify-between rounded-md px-2.5 text-sm text-on-surface hover:bg-surface-container {activeOffset ===
								row.optionIndex
									? 'bg-surface-container'
									: ''}"
								onclick={() => {
									activeOffset = row.optionIndex;
									activateRow(row);
								}}
								onkeydown={(event) => {
									if (event.key === 'Enter' || event.key === ' ') {
										event.preventDefault();
										activeOffset = row.optionIndex;
										activateRow(row);
									}
								}}
							>
								<span>{row.tag}</span>
								{#if row.selected}
									<span class="text-primary"><Icon icon="check" /></span>
								{/if}
							</li>
						{:else if row.kind === 'create'}
							<li
								id="{listId}-opt-{row.optionIndex}"
								role="option"
								aria-selected="false"
								class="flex min-h-9 cursor-pointer items-center gap-1.5 rounded-md px-2.5 text-sm font-medium text-primary hover:bg-surface-container {activeOffset ===
								row.optionIndex
									? 'bg-surface-container'
									: ''}"
								onclick={() => {
									activeOffset = row.optionIndex;
									activateRow(row);
								}}
								onkeydown={(event) => {
									if (event.key === 'Enter' || event.key === ' ') {
										event.preventDefault();
										activeOffset = row.optionIndex;
										activateRow(row);
									}
								}}
							>
								<Icon icon="plus" />
								{$t('editor.tagsCreate', { values: { tag: row.tag } })}
							</li>
						{:else if row.kind === 'nomatch'}
							<li class="px-2.5 py-2 text-xs text-on-surface-variant">
								{$t('editor.tagsNoMatch', { values: { query: row.query } })}
							</li>
						{:else}
							<li class="px-2.5 py-2 text-xs text-on-surface-variant">
								{$t('editor.tagsEmpty')}
							</li>
						{/if}
					{/each}
				</ul>
			</div>
		{/if}
	</div>
</div>
