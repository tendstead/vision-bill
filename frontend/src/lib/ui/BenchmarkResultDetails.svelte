<script lang="ts">
	import type { IndividualResult } from '$lib/benchmark-experiments';
	import { t } from '$lib/i18n';
	let { results, names = {} }: { results: IndividualResult[]; names?: Record<string, string> } =
		$props();
	const percent = (value: number | null | undefined) =>
		value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(1)}%`;
	const editorLabels: Record<string, string> = {
		merchant_name: 'merchant',
		merchant_address: 'address',
		receipt_number: 'number',
		payment_method: 'paymentMethod',
		discount_total: 'discount',
		tax_total: 'taxTotal',
		line_items: 'lineItems'
	};
	const json = (value: unknown) => JSON.stringify(value, null, 2);
</script>

<div class="mt-4 space-y-3">
	{#each results as result (`${result.model_id}:${result.receipt_id}`)}
		{@const enrichment = result.metadata?.classification_enrichment}
		<details class="rounded-xl border border-outline-variant p-4">
			<summary class="cursor-pointer text-sm">
				<strong>{result.model_id}</strong> · {names[result.receipt_id] ?? result.receipt_id} · {result.success
					? $t('benchmarkResults.extraction', {
							values: { score: percent(result.scores?.normalized) }
						})
					: $t('benchmarkResults.failed')}
			</summary>
			{#if result.error}<p class="mt-3 text-sm text-error">
					{result.error === 'ocr_gpu_out_of_memory'
						? $t('settings.dedicatedOcr.reasons.ocr_gpu_out_of_memory')
						: result.error}
				</p>{/if}
			{#if result.version === '4' || result.version === '5'}
				<p class="mt-3 text-sm">
					{$t('benchmarkResults.confirmedFields')} · {result.scored_fields
						.map((field) => $t('editor.' + (editorLabels[field] ?? field), { default: field }))
						.join(', ')}
				</p>
				<dl class="mt-3 grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-3">
					{@render diagnostic(
						result.version === '5' ? 'confirmedCorrectReceipts' : 'fullyCorrectReceipt',
						result.full_receipt_correct
					)}
					{@render diagnostic('exactAmounts', result.exact_amounts)}
					{@render diagnostic('exactItemOrder', result.exact_item_order)}
					{@render diagnostic('exactItemCount', result.exact_item_count)}
					{@render diagnostic('itemMatches', result.item_tp)}
					{@render diagnostic('itemExtra', result.item_fp)}
					{@render diagnostic('itemMissing', result.item_fn)}
					{@render diagnostic('itemPrecision', percent(result.precision))}
					{@render diagnostic('itemRecall', percent(result.recall))}
					{#if result.item_matches?.length > 0}
						<div class="sm:col-span-2 lg:col-span-3">
							<dt class="font-medium">{$t('benchmarkResults.itemPositions')}</dt>
							<dd>
								{result.item_matches
									.map(([expected, actual]) => `${expected + 1} → ${actual + 1}`)
									.join(', ')}
							</dd>
						</div>
					{/if}
				</dl>
			{/if}
			{#if enrichment || result.success}
				<p class="mt-3 text-sm text-on-surface-variant">
					{$t(`benchmarkResults.enrichment.${enrichment?.state ?? 'not_recorded'}`)}
					{#if enrichment?.missing_fields.length}
						{$t('benchmarkResults.missingLabels', {
							values: { fields: enrichment.missing_fields.join(', ') }
						})}
					{/if}
				</p>
			{/if}
			{#if result.success}
				<p class="mt-3 text-sm">
					{#if result.version !== '5'}
						{$t('benchmarkResults.strict', { values: { score: percent(result.scores?.strict) } })} ·
					{/if}
					{#if result.version === '5' && result.scores?.classification == null}
						{$t('benchmarkResults.classificationUnscored')}
					{:else}
						{$t('benchmarkResults.classificationScore', {
							values: { score: percent(result.scores?.classification) }
						})}
					{/if}
					{#if result.rows}
						· {$t('benchmarkResults.legacyMissingRows', { values: { count: result.rows.missing } })}
						· {$t('benchmarkResults.legacyExtraRows', { values: { count: result.rows.extra } })}
					{/if}
				</p>
				<div class="mt-3 overflow-x-auto">
					<table class="w-full text-left text-sm">
						<thead>
							<tr>
								<th scope="col" class="p-2">{$t('benchmarkResults.field')}</th>
								<th scope="col" class="p-2">{$t('benchmarkResults.reference')}</th>
								<th scope="col" class="p-2">{$t('benchmarkResults.extracted')}</th>
								<th scope="col" class="p-2">{$t('benchmarkResults.equivalent')}</th>
							</tr>
						</thead>
						<tbody>
							{#each result.differences ?? [] as difference, i (`${difference.path}:${i}`)}
								<tr class="border-t border-outline-variant">
									<td class="p-2">{difference.path}</td>
									<td class="p-2"
										><pre class="whitespace-pre-wrap">{json(difference.expected)}</pre></td
									>
									<td class="p-2"
										><pre class="whitespace-pre-wrap">{json(difference.actual)}</pre></td
									>
									<td class="p-2"
										>{$t(
											difference.equivalent ? 'benchmarkResults.yes' : 'benchmarkResults.no'
										)}</td
									>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
			<details class="mt-3">
				<summary class="cursor-pointer text-sm">{$t('benchmarkResults.processingDetails')}</summary>
				<pre class="mt-2 overflow-x-auto whitespace-pre-wrap text-xs">{json(result.metadata)}</pre>
			</details>
			{#if result.success}
				<details class="mt-3">
					<summary class="cursor-pointer text-sm">{$t('benchmarkResults.receiptJson')}</summary>
					<div class="mt-2 grid gap-3 sm:grid-cols-2">
						<pre class="overflow-x-auto whitespace-pre-wrap text-xs">{json(result.expected)}</pre>
						<pre class="overflow-x-auto whitespace-pre-wrap text-xs">{json(result.actual)}</pre>
					</div>
				</details>
			{/if}
		</details>
	{/each}
</div>

{#snippet diagnostic(
	label:
		| 'fullyCorrectReceipt'
		| 'confirmedCorrectReceipts'
		| 'exactAmounts'
		| 'exactItemOrder'
		| 'exactItemCount'
		| 'itemMatches'
		| 'itemExtra'
		| 'itemMissing'
		| 'itemPrecision'
		| 'itemRecall',
	value: boolean | number | string | null
)}
	<div>
		<dt class="font-medium">{$t(`benchmarkResults.${label}`)}</dt>
		<dd>
			{typeof value === 'boolean'
				? $t(value ? 'benchmarkResults.yes' : 'benchmarkResults.no')
				: (value ?? $t('benchmarkResults.unavailable'))}
		</dd>
	</div>
{/snippet}
