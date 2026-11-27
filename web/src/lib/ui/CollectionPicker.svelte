<script lang="ts">
	import { t } from '$lib/i18n';
	import { filterCollections } from '$lib/collection-picker-logic';
	import { colorToStyle } from '$lib/collection-colors';
	import type { CollectionSummary } from '$lib/types';

	let {
		options,
		mode,
		value,
		allowCreate = false,
		onchange,
		oncreate,
		placeholder = ''
	}: {
		options: CollectionSummary[];
		mode: 'single' | 'multi';
		value: string | string[];
		allowCreate?: boolean;
		onchange: (v: string | string[]) => void;
		oncreate?: (name: string) => void;
		placeholder?: string;
	} = $props();

	let open = $state(false);
	let query = $state('');
	let active = $state(0);
	let boxEl: HTMLDivElement | undefined = $state();

	let selectedIds = $derived(mode === 'multi' ? (value as string[]) : []);
	let singleId = $derived(mode === 'single' ? (value as string) : '');
	let rows = $derived(filterCollections(options, query, selectedIds));
	let listId = $state('cp-list-' + Math.random().toString(36).slice(2));

	// In single mode the "All collections" row occupies index 0 and the
	// filtered options are offset by 1; multi mode has no such row.
	function optionId(id: string): string {
		return listId + '-opt-' + id;
	}

	let activeOptionId = $derived.by(() => {
		if (!open) return undefined;
		if (mode === 'single') {
			if (active === 0) return listId + '-all';
			const row = rows[active - 1];
			return row ? optionId(row.id) : undefined;
		}
		const row = rows[active];
		return row ? optionId(row.id) : undefined;
	});

	// Option-row index currently highlighted (single mode is offset by 1
	// because the "All collections" row takes index 0).
	let activeRow = $derived(mode === 'single' ? active - 1 : active);

	// Total navigable rows: single mode includes the "All" row at index 0.
	let totalRows = $derived(mode === 'single' ? rows.length + 1 : rows.length);

	// Clamp the highlight index when the filtered list shrinks (e.g. a multi
	// toggle removes the highlighted row).
	$effect(() => {
		if (active >= totalRows) active = Math.max(0, totalRows - 1);
	});

	function labelFor(id: string): string {
		return options.find((o) => o.id === id)?.name ?? '';
	}

	function pickSingle(id: string): void {
		onchange(id);
		open = false;
		query = '';
	}

	function toggleMulti(id: string): void {
		const next = selectedIds.includes(id)
			? selectedIds.filter((x) => x !== id)
			: [...selectedIds, id];
		onchange(next);
	}

	function removeMulti(id: string): void {
		onchange(selectedIds.filter((x) => x !== id));
	}

	function create(): void {
		if (!allowCreate || !query.trim()) return;
		oncreate?.(query.trim());
		query = '';
	}

	function activateOption(id: string): void {
		if (mode === 'single') pickSingle(id);
		else toggleMulti(id);
	}

	function optionKeydown(id: string | null, e: KeyboardEvent): void {
		if (e.key === 'Enter' || e.key === ' ') {
			e.preventDefault();
			if (id === null) create();
			else activateOption(id);
		}
	}

	function onKeydown(e: KeyboardEvent): void {
		if (e.key === 'ArrowDown') {
			active = (active + 1) % Math.max(totalRows, 1);
			e.preventDefault();
		} else if (e.key === 'ArrowUp') {
			active = (active - 1 + Math.max(totalRows, 1)) % Math.max(totalRows, 1);
			e.preventDefault();
		} else if (e.key === 'Enter') {
			e.preventDefault();
			if (mode === 'single') {
				pickSingle(active === 0 ? '' : (rows[active - 1]?.id ?? ''));
			} else if (rows[active]) {
				toggleMulti(rows[active].id);
			} else {
				create();
			}
		} else if (e.key === 'Escape') {
			open = false;
		}
	}

	// Close on outside click (the dropdown is nested inside boxEl, so only true
	// outside clicks close it).
	$effect(() => {
		if (!open) return;
		function onPointerDown(event: PointerEvent): void {
			if (boxEl && !boxEl.contains(event.target as Node)) {
				open = false;
			}
		}
		document.addEventListener('pointerdown', onPointerDown);
		return () => document.removeEventListener('pointerdown', onPointerDown);
	});
</script>

<div bind:this={boxEl} class="relative">
	<div
		class="flex min-h-11 cursor-text flex-wrap items-center gap-1 rounded-lg border border-outline-variant bg-surface-container-lowest px-2 py-1.5 focus-within:border-primary focus-within:ring-1 focus-within:ring-primary"
	>
		{#if mode === 'multi'}
			{#each selectedIds as id (id)}
				<span
					class="inline-flex items-center gap-1 rounded-full bg-primary-container py-0.5 pl-2 pr-1 text-sm font-medium text-on-primary-container"
				>
					{labelFor(id)}
					<button
						type="button"
						class="flex size-5 items-center justify-center rounded-full text-xs hover:bg-on-primary-container/10 max-sm:min-h-11 max-sm:min-w-11"
						aria-label={$t('collections.remove', { values: { name: labelFor(id) } })}
						onclick={(event) => {
							event.stopPropagation();
							removeMulti(id);
						}}
					>
						✕
					</button>
				</span>
			{/each}
		{/if}
		<input
			type="checkbox"
			id={listId + '-input'}
			role="combobox"
			aria-expanded={open}
			aria-controls={open ? listId : undefined}
			aria-activedescendant={activeOptionId}
			aria-autocomplete="list"
			aria-label={placeholder || $t('collections.searchPlaceholder')}
			placeholder={mode === 'single' ? labelFor(singleId) || placeholder : placeholder}
			value={query}
			class="flex-1 borderless bg-transparent text-sm outline-none"
			onfocus={() => (open = true)}
			onclick={() => (open = true)}
			oninput={(e) => (query = (e.currentTarget as HTMLInputElement).value)}
			onkeydown={onKeydown}
		/>
	</div>

	{#if open}
		<ul
			id={listId}
			class="z-10 mt-1 max-h-52 overflow-y-auto rounded-lg border border-outline-variant bg-surface-container-lowest p-1 shadow-elevation-2"
			role="listbox"
			aria-label={$t('collections.searchPlaceholder')}
		>
			{#if mode === 'single'}
				<li
					role="option"
					id={listId + '-all'}
					tabindex="-1"
					aria-selected={singleId === ''}
					class="flex min-h-9 cursor-pointer items-center gap-2 rounded-md px-2.5 text-sm {singleId ===
					''
						? 'font-medium'
						: ''} {active === 0 ? 'bg-surface-container' : ''}"
					onclick={() => pickSingle('')}
					onkeydown={(e) => optionKeydown('', e)}
				>
					{$t('statistics.collectionAll')}
				</li>
			{/if}
			{#each rows as c, i (c.id)}
				<li
					role="option"
					id={optionId(c.id)}
					tabindex="-1"
					aria-selected={mode === 'single' ? singleId === c.id : selectedIds.includes(c.id)}
					class="flex min-h-9 cursor-pointer items-center gap-2 rounded-md px-2.5 text-sm {i ===
					activeRow
						? 'bg-surface-container'
						: ''}"
					onclick={() => (mode === 'single' ? pickSingle(c.id) : toggleMulti(c.id))}
					onkeydown={(e) => optionKeydown(c.id, e)}
				>
					<span class="size-3 shrink-0 rounded-full" style={colorToStyle(c.color)}></span>
					<span class="truncate">{c.name}</span>
				</li>
			{/each}
			{#if allowCreate && query.trim() && rows.length === 0}
				<li
					role="option"
					id={listId + '-create'}
					tabindex="-1"
					aria-selected="false"
					class="flex min-h-9 cursor-pointer items-center gap-1.5 rounded-md px-2.5 text-sm font-medium text-primary"
					onclick={create}
					onkeydown={(e) => optionKeydown(null, e)}
				>
					+ {$t('collections.create', { values: { name: query.trim() } })}
				</li>
			{/if}
			{#if rows.length === 0 && !(allowCreate && query.trim()) && mode !== 'single'}
				<li class="px-2.5 py-2 text-xs text-on-surface-variant">
					{$t('collections.noMatch', { values: { query } })}
				</li>
			{/if}
		</ul>
	{/if}
</div>
