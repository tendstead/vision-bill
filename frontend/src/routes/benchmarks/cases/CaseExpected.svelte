<script lang="ts">
	import {
		collidingProductTagLabels,
		hasCustomTagCollision,
		referenceTagText,
		productTagValue
	} from '$lib/i18n/product-tag-label';
	import { locale, t } from '$lib/i18n';
	import messages from '$lib/i18n/en.json';
	import type { Category, DraftLineItem, ExtractionDraft, PaymentMethod } from '$lib/types';
	let { expected = $bindable() }: { expected: ExtractionDraft } = $props();
	let tagInput = $state.raw<{ line: DraftLineItem; text: string } | null>(null);
	const headers = [
		['merchant_name', 'merchant'],
		['merchant_address', 'address'],
		['receipt_number', 'number'],
		['date', 'date'],
		['time', 'time'],
		['currency', 'currency']
	] as const;
	const amounts = [
		['subtotal', 'subtotal'],
		['discount_total', 'discount'],
		['tax_total', 'taxTotal'],
		['tip', 'tip'],
		['total', 'total']
	] as const;
	const categories = Object.keys(messages.categories) as Category[];
	const spendingCategories = Object.keys(messages.spendingCategories);
	const payments: PaymentMethod[] = [
		'cash',
		'credit_card',
		'debit_card',
		'mobile_payment',
		'check',
		'other',
		'unknown'
	];
	const lineFields = [
		['description', 'description'],
		['quantity', 'quantity'],
		['unit_price', 'unitPrice'],
		['total_price', 'totalPrice']
	] as const;
	const taxFields = [
		['name', 'taxName'],
		['rate', 'taxRate'],
		['amount', 'amount']
	] as const;
	const nullableText = (value: string) => (value.trim() ? value : null);
	const decimal = (value: string) => value.trim().replace(',', '.') || null;
</script>

<fieldset class="case-panel">
	<legend>{$t('editor.details')}</legend>
	<div class="field-grid">
		{#each headers as [key, label] (key)}
			<label
				>{$t('editor.' + label)}
				<input
					type={key === 'date' ? 'date' : 'text'}
					placeholder={key === 'time' ? 'HH:MM[:SS]' : undefined}
					maxlength={key === 'currency' ? 3 : undefined}
					pattern={key === 'currency' ? '[A-Z]{3}' : undefined}
					value={expected[key] ?? ''}
					oninput={(e) => {
						const value = e.currentTarget.value;
						expected[key] = value.trim()
							? key === 'currency'
								? value.toUpperCase()
								: value
							: null;
					}}
				/>
			</label>
		{/each}
		<label
			>{$t('editor.paymentMethod')}
			<select
				value={expected.payment_method ?? ''}
				onchange={(e) =>
					(expected.payment_method = (e.currentTarget.value || null) as PaymentMethod | null)}
			>
				<option value="">{$t('benchmarkCase.unspecified')}</option>
				{#each payments as payment (payment)}<option value={payment}
						>{$t('payment.' + payment)}</option
					>{/each}
			</select>
		</label>
		<label
			>{$t('editor.category')}
			<select
				value={expected.category ?? ''}
				onchange={(e) => (expected.category = (e.currentTarget.value || null) as Category | null)}
			>
				<option value="">{$t('benchmarkCase.unscored')}</option>
				{#each categories as category (category)}<option value={category}
						>{$t('categories.' + category)}</option
					>{/each}
			</select>
		</label>
	</div>
</fieldset>
<fieldset class="case-panel">
	<legend>{$t('editor.amounts')}</legend>
	<div class="field-grid">
		{#each amounts as [key, label] (key)}
			<label
				>{$t('editor.' + label)}<input
					inputmode="decimal"
					value={expected[key] ?? ''}
					pattern={key === 'subtotal' || key === 'total'
						? '-?[0-9]+([.][0-9]+)?'
						: '[0-9]+([.][0-9]+)?'}
					oninput={(e) => (expected[key] = decimal(e.currentTarget.value))}
				/></label
			>
		{/each}
	</div>
</fieldset>
<fieldset class="case-panel space-y-3">
	<legend>{$t('editor.lineItems')}</legend>
	{#each expected.line_items ?? [] as line, index (line)}
		<div class="field-grid rounded-lg border border-outline-variant p-3">
			{#each lineFields as [key, label] (key)}
				<label
					>{$t('editor.' + label)}<input
						inputmode={key === 'description' ? 'text' : 'decimal'}
						value={line[key] ?? ''}
						oninput={(e) =>
							(line[key] =
								key === 'description'
									? nullableText(e.currentTarget.value)
									: decimal(e.currentTarget.value))}
					/></label
				>
			{/each}
			<details
				class="space-y-2 sm:col-span-2"
				open={line.spending_category != null || line.tags != null}
			>
				<summary class="cursor-pointer text-sm font-medium">{$t('benchmarkCase.labels')}</summary>
				<label
					>{$t('editor.spendingCategory')}
					<select
						value={line.spending_category ?? ''}
						onchange={(e) => (line.spending_category = e.currentTarget.value || null)}
					>
						<option value="">{$t('benchmarkCase.unscored')}</option>
						{#each spendingCategories as category (category)}<option value={category}
								>{$t('spendingCategories.' + category)}</option
							>{/each}
					</select>
				</label>
				<label class="case-check"
					><input
						type="checkbox"
						checked={line.tags != null}
						onchange={(e) => (line.tags = e.currentTarget.checked ? [] : null)}
					/>{$t('benchmarkCase.scoreTags')}</label
				>
				{#if line.tags != null}
					{@const collisions = collidingProductTagLabels(line.tags, $locale)}
					{@const lines = line.tags.map((tag) => referenceTagText(tag, $locale, collisions))}
					<label
						>{$t('benchmarkCase.tags')}<textarea
							rows="3"
							value={tagInput?.line === line ? tagInput.text : lines.join('\n')}
							oninput={(e) => {
								const previous = line.tags ?? [];
								const displayed = tagInput?.line === line ? tagInput.text.split('\n') : lines;
								tagInput = { line, text: e.currentTarget.value };
								line.tags = e.currentTarget.value.split('\n').map((text, index) => {
									const match = displayed[index] === text ? index : displayed.indexOf(text);
									return match >= 0 ? previous[match] : productTagValue(text, $locale);
								});
							}}
							onblur={() => {
								tagInput = null;
								line.tags = (line.tags ?? []).map((tag) => tag.trim()).filter(Boolean);
							}}></textarea></label
					>
					{#each new Set(line.tags.filter( (tag) => hasCustomTagCollision(tag, collisions) )) as tag (tag)}
						<p class="text-sm text-on-surface-variant">
							{$t('editor.customTagName', { values: { tag } })}
						</p>
					{/each}
					<p class="text-sm text-on-surface-variant">{$t('benchmarkCase.tagsHint')}</p>
				{/if}
			</details>
			<button
				type="button"
				class="case-secondary"
				onclick={() => {
					const next = expected.line_items!.filter((_, i) => i !== index);
					expected.line_items = next.length ? next : null;
				}}>{$t('common.remove')}</button
			>
		</div>
	{/each}
	{#if !expected.line_items?.length}
		<label class="case-check"
			><input
				type="checkbox"
				checked={Array.isArray(expected.line_items)}
				onchange={(e) => (expected.line_items = e.currentTarget.checked ? [] : null)}
			/>{$t('benchmarkCase.emptyItems')}</label
		>
	{/if}
	<button
		type="button"
		class="case-secondary"
		onclick={() => (expected.line_items = [...(expected.line_items ?? []), {}])}
		>{$t('editor.addItem')}</button
	>
</fieldset>
<fieldset class="case-panel space-y-3">
	<legend>{$t('editor.taxes')}</legend>
	{#if expected.taxes?.length}<p class="text-sm text-on-surface-variant">
			{$t('editor.taxRateHint')}
		</p>{/if}
	{#each expected.taxes ?? [] as tax, index (tax)}
		<div class="field-grid rounded-lg border border-outline-variant p-3">
			{#each taxFields as [key, label] (key)}
				<label
					>{$t('editor.' + label)}<input
						inputmode={key === 'name' ? 'text' : 'decimal'}
						value={tax[key] ?? ''}
						oninput={(e) =>
							(tax[key] =
								key === 'name'
									? nullableText(e.currentTarget.value)
									: decimal(e.currentTarget.value))}
					/></label
				>
			{/each}
			<button
				type="button"
				class="case-secondary"
				onclick={() => {
					const next = expected.taxes!.filter((_, i) => i !== index);
					expected.taxes = next.length ? next : null;
				}}>{$t('common.remove')}</button
			>
		</div>
	{/each}
	{#if !expected.taxes?.length}
		<label class="case-check"
			><input
				type="checkbox"
				checked={Array.isArray(expected.taxes)}
				onchange={(e) => (expected.taxes = e.currentTarget.checked ? [] : null)}
			/>{$t('benchmarkCase.emptyTaxes')}</label
		>
	{/if}
	<button
		type="button"
		class="case-secondary"
		onclick={() => (expected.taxes = [...(expected.taxes ?? []), {}])}>{$t('editor.addTax')}</button
	>
</fieldset>
