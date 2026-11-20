<script lang="ts">
	import { untrack } from 'svelte';
	import { SvelteSet } from 'svelte/reactivity';
	import { createQuery } from '@tanstack/svelte-query';
	import { sessionViewGuard } from '$lib/session-view';
	import { t, translate } from '$lib/i18n';
	import { api } from '$lib/api/client';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	import { isPlainDecimal, isSignedDecimal, validReviewDecimal } from '$lib/ui/money';
	import { snackbar } from '$lib/ui/snackbar.svelte';
	import Icon from '$lib/ui/Icon.svelte';
	import TagEditor from '$lib/ui/TagEditor.svelte';
	import CollectionPicker from '$lib/ui/CollectionPicker.svelte';
	import AmountPolicyEditor from './AmountPolicyEditor.svelte';
	import { hasOverlapRows, unknownOverlap } from './overlap-decisions';
	import type {
		Category,
		AmountPolicy,
		SourceRef,
		LineItemRow,
		PaymentMethod,
		ReceiptRow,
		ReceiptWrite,
		TaxLineRow
	} from '$lib/types';

	const CATEGORIES: Category[] = [
		'grocery',
		'electronics',
		'clothing',
		'restaurant',
		'fuel',
		'pharmacy',
		'entertainment',
		'other'
	];

	const PAYMENT_METHODS: PaymentMethod[] = [
		'cash',
		'credit_card',
		'debit_card',
		'mobile_payment',
		'check',
		'other',
		'unknown'
	];

	const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

	/** All form values are strings (amounts travel as decimal strings). */
	interface EditorLine {
		id?: string;
		description: string;
		original_description: string;
		quantity: string;
		unit_price: string;
		total_price: string;
		spending_category: string;
		category_source: LineItemRow['category_source'];
		manualChoice: boolean;
		remember_category: boolean;
		suggestion: string | null;
		lookupVersion: number;
		tags: string[];
	}

	interface EditorTax {
		name: string;
		rate: string;
		amount: string;
	}

	interface EditorForm {
		expected_operation_revision: number;
		amount_policy: AmountPolicy;
		field_sources: Record<string, SourceRef[]>;
		overlap_decisions: Record<string, unknown>[];
		confidence: string;
		merchant_name: string;
		merchant_address: string;
		receipt_number: string;
		date: string;
		time: string;
		currency: string;
		category: string;
		payment_method: string;
		subtotal: string;
		discount_total: string;
		tax_total: string;
		tip: string;
		total: string;
		lines: EditorLine[];
		taxLines: EditorTax[];
		collection_ids: string[];
	}

	function buildForm(
		receipt: ReceiptRow,
		lineItems: LineItemRow[],
		taxes: TaxLineRow[],
		collectionIds: string[]
	): EditorForm {
		return {
			expected_operation_revision: receipt.operation_revision,
			amount_policy: structuredClone($state.snapshot(receipt.amount_policy)),
			field_sources: structuredClone($state.snapshot(receipt.field_sources)),
			overlap_decisions: structuredClone($state.snapshot(receipt.overlap_decisions)),
			confidence: String(receipt.confidence),
			merchant_name: receipt.merchant_name,
			merchant_address: receipt.merchant_address ?? '',
			receipt_number: receipt.receipt_number ?? '',
			date: receipt.date,
			time: receipt.time?.replace(/^(\d{2}:\d{2}):00(?:\.0+)?$/, '$1') ?? '',
			currency: receipt.currency,
			category: receipt.category,
			payment_method: receipt.payment_method,
			subtotal: String(receipt.subtotal),
			discount_total: String(receipt.discount_total ?? ''),
			tax_total: String(receipt.tax_total ?? ''),
			tip: receipt.tip === null ? '' : String(receipt.tip),
			total: String(receipt.total),
			lines: lineItems.map((line) => ({
				id: line.id,
				description: line.description,
				original_description: line.original_description ?? line.description,
				quantity: String(line.quantity ?? ''),
				unit_price: String(line.unit_price ?? ''),
				total_price: String(line.total_price),
				spending_category: line.spending_category ?? 'unknown',
				category_source: line.category_source ?? 'unknown',
				manualChoice: line.category_source === 'user',
				remember_category: false,
				suggestion: null,
				lookupVersion: 0,
				tags: [...(line.tags ?? [])]
			})),
			taxLines: taxes.map((tax) => ({
				name: tax.name,
				rate: tax.rate === null ? '' : String(tax.rate),
				amount: String(tax.amount)
			})),
			collection_ids: [...collectionIds]
		};
	}

	let {
		receipt,
		lineItems,
		taxes,
		revision = 0,
		savedRevision = 0,
		collectionIds = [],
		busy = false,
		onSave,
		onSaveAndVerify,
		onDirtyChange,
		onSnapshotChange,
		onSource,
		onPendingChange
	}: {
		receipt: ReceiptRow;
		lineItems: LineItemRow[];
		taxes: TaxLineRow[];
		/** Bumped after a successful detail refetch to clear pending memory choices. */
		revision?: number;
		/** Bumped once PUT succeeds, even if a later refetch fails. */
		savedRevision?: number;
		/** Collections the receipt currently belongs to (detail view). */
		collectionIds?: string[];
		/** Disable the action buttons while a request is in flight. */
		busy?: boolean;
		/** Offered when the page wants a plain save (no verify). */
		onSave?: (write: ReceiptWrite) => void;
		/** Offered when the receipt can still be verified. */
		onSaveAndVerify?: (write: ReceiptWrite) => void;
		/** Notified with the current dirty state after (re)initialisation. */
		onDirtyChange?: (dirty: boolean) => void;
		/** Revisions of the server data currently shown in this editor. */
		onSnapshotChange?: (
			snapshot: Pick<ReceiptRow, 'id' | 'operation_revision' | 'review_revision'>
		) => void;
		onSource?: (sources: SourceRef[]) => void;
		onPendingChange?: (pending: boolean) => void;
	} = $props();

	const isCurrentView = sessionViewGuard();

	function initialForm(): EditorForm {
		return buildForm(receipt, lineItems, taxes, collectionIds);
	}

	let form = $state<EditorForm>(initialForm());
	let snapshot = $state<EditorForm>(initialForm());
	let initializedReceiptId: string | undefined;
	let initializedRevision = -1;
	let rendered = $state<Pick<ReceiptRow, 'id' | 'operation_revision' | 'review_revision'> | null>(
		null
	);
	let initializedSavedRevision = 0;
	let merchantVersion = 0;
	const pendingTags = new SvelteSet<EditorLine>();
	const pendingCollections = new SvelteSet<object>();
	let childPending = $derived(pendingTags.size > 0 || pendingCollections.size > 0);
	$effect(() => {
		onPendingChange?.(childPending);
	});
	function tagPending(line: EditorLine, pending: boolean) {
		if (pending) pendingTags.add(line);
		else pendingTags.delete(line);
	}

	function dirtyForm(value: EditorForm): string {
		return JSON.stringify(value, (key, item: unknown) =>
			['manualChoice', 'suggestion', 'lookupVersion', 'category_source'].includes(key)
				? undefined
				: item
		);
	}

	// Tag vocabulary (the <select> source), fetched once per editor mount.
	const tagList = createQuery(
		() => ({ queryKey: queryKeys.tags(), queryFn: () => api.listTags() }),
		() => queryClient
	);
	let tagOptions = $derived<string[]>(tagList.data ?? []);
	const categoryList = createQuery(
		() => ({
			queryKey: queryKeys.spendingCategories(),
			queryFn: () => api.listSpendingCategories()
		}),
		() => queryClient
	);
	let spendingCategories = $derived(categoryList.data ?? []);

	// Collection vocabulary for the multi-select, fetched once per editor mount.
	const collections = createQuery(
		() => ({ queryKey: queryKeys.collections(), queryFn: () => api.listCollections() }),
		() => queryClient
	);

	$effect(() => {
		const receiptId = receipt.id;
		const operationRevision = receipt.operation_revision;
		const reviewRevision = receipt.review_revision;
		const reload = receiptId !== initializedReceiptId || revision !== initializedRevision;
		const incomingRevision = revision;
		untrack(() => {
			const changed =
				operationRevision !== rendered?.operation_revision ||
				reviewRevision !== rendered?.review_revision;
			if (!reload && (!changed || dirty || childPending)) return;
			pendingTags.clear();
			pendingCollections.clear();
			merchantVersion++;
			initializedReceiptId = receiptId;
			initializedRevision = incomingRevision;
			const fresh = initialForm();
			form = fresh;
			snapshot = fresh;
			rendered = {
				id: receiptId,
				operation_revision: operationRevision,
				review_revision: reviewRevision
			};
		});
	});
	$effect(() => {
		if (rendered) onSnapshotChange?.(rendered);
	});
	$effect(() => {
		if (savedRevision !== initializedSavedRevision) {
			initializedSavedRevision = savedRevision;
			for (const line of form.lines) line.remember_category = false;
		}
	});

	let dirty = $derived(dirtyForm(form) !== dirtyForm(snapshot));
	$effect(() => {
		onDirtyChange?.(dirty);
	});

	const REQUIRED_AMOUNTS = ['subtotal', 'total'] as const;

	let errors = $derived.by(() => {
		const set = new SvelteSet<string>();
		if (!form.merchant_name.trim()) {
			set.add('editor.merchantRequired');
		}
		if (!DATE_RE.test(form.date)) {
			set.add('editor.dateRequired');
		}
		for (const key of REQUIRED_AMOUNTS) {
			if (!isSignedDecimal(form[key])) {
				set.add('editor.invalidNumber');
			}
		}
		if (!/^[a-z]{3}$/i.test(form.currency.trim())) {
			set.add('editor.invalidCurrency');
		}
		if (!isPlainDecimal(form.tax_total)) set.add('editor.invalidAmount');
		for (const amount of [form.discount_total, form.tip])
			if (amount.trim() !== '' && !isPlainDecimal(amount)) set.add('editor.invalidAmount');
		const confidence = Number(form.confidence);
		if (
			form.confidence.trim() === '' ||
			!Number.isInteger(confidence) ||
			confidence < 0 ||
			confidence > 100
		) {
			set.add('editor.invalidConfidence');
		}
		for (const line of form.lines) {
			if (!line.description.trim()) set.add('editor.descriptionRequired');
			if (line.quantity.trim() && !validReviewDecimal(line.quantity, 'positive')) {
				set.add('editor.invalidQuantity');
			}
			if (
				(line.unit_price.trim() !== '' && !isSignedDecimal(line.unit_price)) ||
				!isSignedDecimal(line.total_price)
			) {
				set.add('editor.invalidNumber');
			}
		}
		for (const tax of form.taxLines) {
			if (!isPlainDecimal(tax.amount)) {
				set.add('editor.invalidAmount');
			}
			if (tax.rate.trim() !== '') {
				if (!validReviewDecimal(tax.rate, 'rate')) {
					set.add('editor.invalidRate');
				}
			}
		}
		return [...set];
	});
	let valid = $derived(errors.length === 0);

	let hasActions = $derived(Boolean(onSave) || Boolean(onSaveAndVerify));

	function buildWrite(): ReceiptWrite {
		return {
			expected_operation_revision: form.expected_operation_revision,
			amount_policy: form.amount_policy,
			field_sources: form.field_sources,
			overlap_decisions: form.overlap_decisions,
			confidence: Number(form.confidence),
			merchant_name: form.merchant_name.trim(),
			merchant_address: form.merchant_address.trim() || null,
			receipt_number: form.receipt_number.trim() || null,
			date: form.date,
			time: form.time.trim() || null,
			currency: form.currency.trim().toUpperCase(),
			category: form.category as Category,
			line_items: form.lines.map((line) => ({
				...(line.id ? { id: line.id } : {}),
				description: line.description.trim(),
				quantity: line.quantity.trim() || null,
				unit_price: line.unit_price.trim() || null,
				total_price: line.total_price.trim(),
				spending_category: line.spending_category,
				...(line.remember_category ? { remember_category: true } : {}),
				tags: line.tags.map((tag) => tag.trim()).filter((tag) => tag.length > 0)
			})),
			taxes: form.taxLines.map((tax) => ({
				name: tax.name.trim(),
				rate: tax.rate.trim() || null,
				amount: tax.amount.trim()
			})),
			subtotal: form.subtotal.trim(),
			discount_total: form.discount_total.trim() || null,
			tax_total: form.tax_total.trim(),
			tip: form.tip.trim() === '' ? null : form.tip.trim(),
			total: form.total.trim(),
			payment_method: form.payment_method as PaymentMethod,
			collection_ids: [...form.collection_ids]
		};
	}

	function addLine() {
		form.lines.push({
			description: '',
			original_description: '',
			quantity: '',
			unit_price: '',
			total_price: '',
			spending_category: 'unknown',
			category_source: 'unknown',
			manualChoice: false,
			remember_category: false,
			suggestion: null,
			lookupVersion: 0,
			tags: []
		});
	}

	function clearHistoryCategory(line: EditorLine) {
		if (!line.manualChoice && line.category_source === 'history') {
			line.spending_category = 'unknown';
			line.category_source = 'unknown';
		}
		line.suggestion = null;
		line.lookupVersion++;
	}

	function changeDescription(line: EditorLine, description: string) {
		clearHistoryCategory(line);
		line.remember_category = false;
		line.description = description;
	}

	function changeMerchant(merchant: string) {
		merchantVersion++;
		form.merchant_name = merchant;
		for (const line of form.lines) {
			clearHistoryCategory(line);
			line.remember_category = false;
		}
	}

	async function lookupCategory(line: EditorLine) {
		const description = line.description.trim();
		const merchant = form.merchant_name.trim();
		if (!description || !merchant) return;
		const version = ++line.lookupVersion;
		const currentMerchantVersion = merchantVersion;
		const currentReceiptId = receipt.id;
		try {
			const result = await api.getCategorySuggestion(merchant, description);
			if (
				line.lookupVersion !== version ||
				merchantVersion !== currentMerchantVersion ||
				receipt.id !== currentReceiptId ||
				!form.lines.includes(line) ||
				line.description.trim() !== description ||
				form.merchant_name.trim() !== merchant
			)
				return;
			if (result.source !== 'history' || !result.category) return;
			if (line.manualChoice) {
				line.suggestion = result.category === line.spending_category ? null : result.category;
			} else {
				line.spending_category = result.category;
				line.category_source = 'history';
				line.suggestion = null;
			}
		} catch {
			// History is optional; an unavailable lookup must not block receipt review.
		}
	}

	function selectCategory(line: EditorLine, category: string) {
		line.spending_category = category;
		line.category_source = 'user';
		line.manualChoice = true;
		line.remember_category = true;
		line.suggestion = null;
	}

	function removeLineSources() {
		form.overlap_decisions = form.overlap_decisions.map(unknownOverlap);
		form.field_sources = Object.fromEntries(
			Object.entries(form.field_sources).filter(([path]) => !/^line_items(?:\.|\[|$)/.test(path))
		);
	}
	function moveLine(index: number, direction: number) {
		const other = index + direction;
		if (other < 0 || other >= form.lines.length) return;
		[form.lines[index], form.lines[other]] = [form.lines[other], form.lines[index]];
		removeLineSources();
	}
	function removeLine(index: number) {
		removeLineSources();
		form.lines.splice(index, 1);
	}

	function addTax() {
		form.taxLines.push({ name: '', rate: '', amount: '' });
	}

	function removeTax(index: number) {
		form.field_sources = Object.fromEntries(
			Object.entries(form.field_sources).filter(([path]) => !/^taxes(?:\.|\[|$)/.test(path))
		);
		form.taxLines.splice(index, 1);
	}

	const labelClass = 'mb-1 block text-xs font-medium text-on-surface-variant';
	const inputClass =
		'w-full rounded-lg border border-outline-variant bg-surface-container-lowest px-3 py-2.5 text-sm';
</script>

<fieldset disabled={busy} class="space-y-6">
	<!-- Details -->
	<div class="rounded-xl border border-outline-variant bg-surface-container-low p-4">
		<h2 class="mb-3 text-sm font-semibold uppercase tracking-wide text-on-surface-variant">
			{$t('editor.details')} -
			<label class={labelClass} for="re-confidence"
				>{$t('editor.confidence')}: {form.confidence}</label
			>
		</h2>
		<div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
			<div class="sm:col-span-2">
				<label class={labelClass} for="re-merchant">{$t('editor.merchant')}</label>
				<input
					id="re-merchant"
					class={inputClass}
					value={form.merchant_name}
					oninput={(e) => changeMerchant((e.currentTarget as HTMLInputElement).value)}
					onblur={() => form.lines.forEach((line) => void lookupCategory(line))}
				/>
			</div>
			<div class="sm:col-span-2">
				<label class={labelClass} for="re-address">{$t('editor.address')}</label>
				<input
					id="re-address"
					class={inputClass}
					value={form.merchant_address}
					oninput={(e) => (form.merchant_address = (e.currentTarget as HTMLInputElement).value)}
				/>
			</div>
			<div>
				<label class={labelClass} for="re-number">{$t('editor.number')}</label>
				<input
					id="re-number"
					class={inputClass}
					value={form.receipt_number}
					oninput={(e) => (form.receipt_number = (e.currentTarget as HTMLInputElement).value)}
				/>
			</div>
			<div>
				<label class={labelClass} for="re-date">{$t('editor.date')}</label>
				<input
					id="re-date"
					type="date"
					class={inputClass}
					value={form.date}
					oninput={(e) => (form.date = (e.currentTarget as HTMLInputElement).value)}
				/>
			</div>
			<div>
				<label class={labelClass} for="re-time">{$t('editor.time')}</label>
				<!-- Native time inputs can sanitize microseconds; retain a text control
				     throughout editing when the saved value has sub-minute precision. -->
				<input
					id="re-time"
					type={receipt.time && !/^\d{2}:\d{2}(?::00(?:\.0+)?)?$/.test(receipt.time)
						? 'text'
						: 'time'}
					class={inputClass}
					value={form.time}
					oninput={(e) => (form.time = (e.currentTarget as HTMLInputElement).value)}
				/>
			</div>
			<div>
				<label class={labelClass} for="re-currency">{$t('editor.currency')}</label>
				<input
					id="re-currency"
					class={inputClass}
					placeholder="EUR"
					value={form.currency}
					oninput={(e) => (form.currency = (e.currentTarget as HTMLInputElement).value)}
				/>
			</div>
			<div>
				<label class={labelClass} for="re-category">{$t('editor.category')}</label>
				<select
					id="re-category"
					class={inputClass}
					value={form.category}
					onchange={(e) => (form.category = (e.currentTarget as HTMLSelectElement).value)}
				>
					{#each CATEGORIES as category (category)}
						<option value={category}>{$t(`categories.${category}`)}</option>
					{/each}
				</select>
			</div>
			<div>
				<label class={labelClass} for="re-payment">{$t('editor.paymentMethod')}</label>
				<select
					id="re-payment"
					class={inputClass}
					value={form.payment_method}
					onchange={(e) => (form.payment_method = (e.currentTarget as HTMLSelectElement).value)}
				>
					{#each PAYMENT_METHODS as method (method)}
						<option value={method}>{$t(`payment.${method}`)}</option>
					{/each}
				</select>
			</div>
			<div class="sm:col-span-2">
				<span class={labelClass}>{$t('editor.collections')}</span>
				<CollectionPicker
					disabled={busy}
					options={collections.data ?? []}
					mode="multi"
					value={form.collection_ids}
					allowCreate
					placeholder={$t('collections.searchPlaceholder')}
					onchange={(v) => (form.collection_ids = v as string[])}
					oncreate={async (name) => {
						if (!isCurrentView()) return;
						const mutation = {};
						const target = form;
						const receiptId = receipt.id;
						const isCurrentForm = () =>
							isCurrentView() && form === target && receipt.id === receiptId;
						pendingCollections.add(mutation);
						try {
							const created = await api.createCollection({ name });
							if (!isCurrentForm()) return;
							form.collection_ids = [...form.collection_ids, created.id];
							await queryClient.invalidateQueries({ queryKey: queryKeys.collections() });
						} catch {
							if (isCurrentForm()) snackbar.notify('error', translate('collections.createFailed'));
						} finally {
							if (isCurrentView()) pendingCollections.delete(mutation);
						}
					}}
				/>
			</div>
		</div>
	</div>

	<!-- Amounts -->
	<div class="rounded-xl border border-outline-variant bg-surface-container-low p-4">
		<h2 class="mb-3 text-sm font-semibold uppercase tracking-wide text-on-surface-variant">
			{$t('editor.amounts')}
		</h2>
		<div class="grid grid-cols-2 gap-3 sm:grid-cols-3">
			{#each [['subtotal', 'editor.subtotal'], ['discount_total', 'editor.discount'], ['tax_total', 'editor.taxTotal'], ['tip', 'editor.tip'], ['total', 'editor.total']] as [field, labelKey] (field)}
				<div class={field === 'total' ? 'col-span-2 sm:col-span-3' : ''}>
					<label class={labelClass} for="re-{field}">{$t(labelKey)}</label>
					<input
						id="re-{field}"
						inputmode="decimal"
						class="{inputClass} {field === 'total' ? 'font-semibold' : ''}"
						value={form[field as 'subtotal' | 'discount_total' | 'tax_total' | 'tip' | 'total']}
						oninput={(e) => {
							const target = e.currentTarget as HTMLInputElement;
							form[field as 'subtotal' | 'discount_total' | 'tax_total' | 'tip' | 'total'] =
								target.value;
						}}
					/>
				</div>
			{/each}
		</div>
	</div>

	<AmountPolicyEditor
		value={form.amount_policy}
		disabled={busy}
		onChange={(value) => (form.amount_policy = value)}
	/>
	<details class="rounded-xl border border-outline-variant p-4">
		<summary>{$t('review.sourceDecisions')}</summary>
		{#each Object.keys(form.field_sources) as path (path)}<p>
				{path}
				{#if onSource}<button type="button" onclick={() => onSource?.(form.field_sources[path])}
						>{$t('review.showSource')}</button
					>{/if}
				<button
					type="button"
					onclick={() => (form.field_sources = { ...form.field_sources, [path]: [] })}
					>{$t('review.markUnknown')}</button
				>
			</p>{/each}
	</details>
	{#each form.overlap_decisions as decision, index (index)}
		<div class="rounded-xl border border-outline-variant p-4">
			<p>{$t('review.overlapNumber', { values: { number: index + 1 } })}</p>
			{#if hasOverlapRows(decision, form.lines.length, form.overlap_decisions)}
				<p>
					{$t('review.overlapRows', {
						values: {
							first: (decision.first_index as number) + 1,
							second: (decision.second_index as number) + 1
						}
					})}
				</p>
			{:else}<p>{$t('review.overlapUnknown')}</p>{/if}
			<label
				><input
					type="checkbox"
					disabled={!hasOverlapRows(decision, form.lines.length, form.overlap_decisions)}
					checked={decision.verified === true}
					onchange={(e) => {
						if (hasOverlapRows(decision, form.lines.length, form.overlap_decisions))
							form.overlap_decisions[index] = { ...decision, verified: e.currentTarget.checked };
					}}
				/>
				{$t('review.reviewedOverlap')}</label
			>
		</div>
	{/each}
	<!-- Line items -->
	<div class="rounded-xl border border-outline-variant bg-surface-container-low p-4">
		<div class="mb-3 flex items-center justify-between">
			<h2 class="text-sm font-semibold uppercase tracking-wide text-on-surface-variant">
				{$t('editor.lineItems')}
			</h2>
			<button
				type="button"
				class="flex items-center gap-1.5 rounded-lg bg-secondary-container px-3 py-2 text-sm font-medium text-on-secondary-container hover:opacity-90"
				onclick={addLine}
			>
				<Icon icon="plus" />
				{$t('editor.addItem')}
			</button>
		</div>
		{#if form.lines.length === 0}
			<p class="text-sm text-on-surface-variant">{$t('editor.noLineItems')}</p>
		{:else}
			<div class="space-y-3">
				{#each form.lines as line, index (line)}
					<div class="rounded-lg border border-outline-variant bg-surface-container-lowest p-3">
						<div class="flex items-start gap-2">
							<input
								class="{inputClass} flex-1"
								id={`li-${index}-description`}
								aria-label={$t('editor.description')}
								placeholder={$t('editor.description')}
								value={line.description}
								oninput={(e) =>
									changeDescription(line, (e.currentTarget as HTMLInputElement).value)}
								onblur={() => void lookupCategory(line)}
							/>
							<button
								type="button"
								class="rounded-lg p-2 text-on-surface-variant hover:bg-error-container hover:text-on-error-container"
								aria-label={$t('common.remove')}
								onclick={() => removeLine(index)}
							>
								<Icon icon="trash" />
							</button>
							<button
								type="button"
								aria-label={$t('editor.moveItemUp')}
								disabled={index === 0 || busy}
								onclick={() => moveLine(index, -1)}>↑</button
							>
							<button
								type="button"
								aria-label={$t('editor.moveItemDown')}
								disabled={index === form.lines.length - 1 || busy}
								onclick={() => moveLine(index, 1)}>↓</button
							>
						</div>
						{#if line.original_description && line.original_description !== line.description}
							<p class="mt-1 text-xs text-on-surface-variant">
								{$t('editor.originalDescription')}: {line.original_description}
							</p>
						{/if}
						<div class="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
							<div>
								<label class={labelClass} for="li-{index}-qty">{$t('editor.quantity')}</label>
								<input
									id="li-{index}-qty"
									inputmode="decimal"
									class={inputClass}
									value={line.quantity}
									oninput={(e) => (line.quantity = (e.currentTarget as HTMLInputElement).value)}
								/>
							</div>
							<div>
								<label class={labelClass} for="li-{index}-unit">{$t('editor.unitPrice')}</label>
								<input
									id="li-{index}-unit"
									inputmode="decimal"
									class={inputClass}
									value={line.unit_price}
									oninput={(e) => (line.unit_price = (e.currentTarget as HTMLInputElement).value)}
								/>
							</div>
							<div>
								<label class={labelClass} for="li-{index}-total">{$t('editor.totalPrice')}</label>
								<input
									id="li-{index}-total"
									inputmode="decimal"
									class={inputClass}
									value={line.total_price}
									oninput={(e) => (line.total_price = (e.currentTarget as HTMLInputElement).value)}
								/>
							</div>
							<div class="col-span-2 sm:col-span-4">
								<label class={labelClass} for="li-{index}-category"
									>{$t('editor.spendingCategory')}</label
								>
								<div class="flex flex-wrap items-center gap-2">
									<select
										id="li-{index}-category"
										class="{inputClass} flex-1"
										value={line.spending_category}
										onchange={(e) =>
											selectCategory(line, (e.currentTarget as HTMLSelectElement).value)}
									>
										{#if !spendingCategories.some((option) => option.code === line.spending_category)}
											<option value={line.spending_category}
												>{$t(`spendingCategories.${line.spending_category}`)}</option
											>
										{/if}
										{#each spendingCategories as option (option.code)}
											<option value={option.code}>{$t(`spendingCategories.${option.code}`)}</option>
										{/each}
									</select>
									<button
										type="button"
										class="rounded-lg border border-outline-variant px-3 py-2 text-xs hover:bg-surface-container-high"
										disabled={line.spending_category === 'unknown' || line.remember_category}
										onclick={() => selectCategory(line, line.spending_category)}
										>{$t('editor.rememberCategory')}</button
									>
								</div>
								<p class="mt-1 text-xs text-on-surface-variant">
									{$t(`editor.categorySource.${line.category_source}`)}
								</p>
								{#if line.suggestion}
									<button
										type="button"
										class="mt-1 text-xs text-primary underline"
										onclick={() => selectCategory(line, line.suggestion ?? 'unknown')}
									>
										{$t('editor.historySuggestion', {
											values: { category: $t(`spendingCategories.${line.suggestion}`) }
										})}
									</button>
								{/if}
							</div>
							<div class="col-span-2 sm:col-span-4">
								<details>
									<summary class="cursor-pointer text-xs text-on-surface-variant"
										>{$t('editor.optionalTags')}</summary
									>
									<div class="mt-2">
										<TagEditor
											disabled={busy}
											value={line.tags}
											{tagOptions}
											id={`li-${index}`}
											onPendingChange={(pending) => tagPending(line, pending)}
										/>
									</div>
								</details>
							</div>
						</div>
					</div>
				{/each}
			</div>
		{/if}
	</div>

	<!-- Taxes -->
	<div class="rounded-xl border border-outline-variant bg-surface-container-low p-4">
		<div class="mb-3 flex items-center justify-between">
			<h2 class="text-sm font-semibold uppercase tracking-wide text-on-surface-variant">
				{$t('editor.taxes')}
			</h2>
			<button
				type="button"
				class="flex items-center gap-1.5 rounded-lg bg-secondary-container px-3 py-2 text-sm font-medium text-on-secondary-container hover:opacity-90"
				onclick={addTax}
			>
				<Icon icon="plus" />
				{$t('editor.addTax')}
			</button>
		</div>
		{#if form.taxLines.length === 0}
			<p class="text-sm text-on-surface-variant">{$t('editor.noTaxes')}</p>
		{:else}
			<div class="space-y-2">
				{#each form.taxLines as tax, index (index)}
					<div class="grid grid-cols-2 items-end gap-2 sm:grid-cols-[1fr_8rem_10rem_2.5rem]">
						<div>
							<label class={labelClass} for="tax-{index}-name">{$t('editor.taxName')}</label>
							<input
								id="tax-{index}-name"
								class={inputClass}
								value={tax.name}
								oninput={(e) => (tax.name = (e.currentTarget as HTMLInputElement).value)}
							/>
						</div>
						<div>
							<label class={labelClass} for="tax-{index}-rate">{$t('editor.taxRate')}</label>
							<input
								id="tax-{index}-rate"
								inputmode="decimal"
								class={inputClass}
								value={tax.rate}
								oninput={(e) => (tax.rate = (e.currentTarget as HTMLInputElement).value)}
							/>
						</div>
						<div>
							<label class={labelClass} for="tax-{index}-amount">{$t('editor.amount')}</label>
							<input
								id="tax-{index}-amount"
								inputmode="decimal"
								class={inputClass}
								value={tax.amount}
								oninput={(e) => (tax.amount = (e.currentTarget as HTMLInputElement).value)}
							/>
						</div>
						<button
							type="button"
							class="rounded-lg p-2.5 text-on-surface-variant hover:bg-error-container hover:text-on-error-container"
							aria-label={$t('common.remove')}
							onclick={() => removeTax(index)}
						>
							<Icon icon="trash" />
						</button>
					</div>
				{/each}
			</div>
		{/if}
	</div>

	<!-- Action bar -->
	{#if hasActions && dirty}
		<div
			class="sticky bottom-20 md:bottom-0 space-y-3 rounded-xl border border-outline-variant bg-surface-container p-4 shadow-elevation-2"
		>
			{#if errors.length > 0}
				<ul class="space-y-1" role="alert">
					{#each errors as error (error)}
						<li class="flex items-center gap-2 text-sm text-error">
							<Icon icon="alert" />
							{$t(error)}
						</li>
					{/each}
				</ul>
			{/if}
			<div class="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
				{#if dirty && onSave}
					<button
						type="button"
						class="rounded-lg border border-outline-variant px-4 py-2.5 text-sm font-medium text-on-surface hover:bg-surface-container-high disabled:opacity-50"
						disabled={busy || childPending || !valid}
						onclick={() => {
							if (!busy && !childPending && valid) onSave(buildWrite());
						}}
					>
						{$t('common.save')}
					</button>
				{/if}
				{#if onSaveAndVerify !== undefined}
					<button
						type="button"
						class="flex items-center justify-center gap-2 rounded-lg bg-action px-4 py-2.5 text-sm font-medium text-on-primary hover:opacity-90 disabled:opacity-50"
						disabled={busy || childPending || !valid}
						onclick={() => {
							if (!busy && !childPending && valid) onSaveAndVerify(buildWrite());
						}}
					>
						<Icon icon="check" />
						{$t('editor.saveAndVerify')}
					</button>
				{/if}
			</div>
		</div>
	{/if}
</fieldset>
