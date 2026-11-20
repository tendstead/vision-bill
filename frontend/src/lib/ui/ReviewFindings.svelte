<script lang="ts">
	import { t } from '$lib/i18n';
	import { diagnosticLabel } from '$lib/diagnostics';
	import type { ReviewFinding, SourceRef } from '$lib/types';
	let {
		findings,
		revision,
		onChange,
		onSource,
		onField,
		editable = true
	}: {
		findings: ReviewFinding[];
		revision: number | string;
		onChange?: (value: { finding_id: string; comment: string }[]) => void;
		onSource?: (sources: SourceRef[]) => void;
		editable?: boolean;
		onField?: (path: string) => void;
	} = $props();
	let comments = $state<Record<string, string>>({});
	let identity = $derived(JSON.stringify([revision, findings, editable]));
	$effect(() => {
		void identity;
		comments = {};
	});
	$effect(() => {
		onChange?.(
			findings
				.filter((f) => f.severity === 'review' && comments[f.id]?.trim())
				.map((f) => ({ finding_id: f.id, comment: comments[f.id].trim() }))
		);
	});
</script>

<div class="my-4 space-y-3" aria-label={$t('review.findings')}>
	{#each findings as finding (finding.id)}
		<article class="rounded-xl border border-outline-variant p-3" data-severity={finding.severity}>
			<p class="font-medium">
				{diagnosticLabel($t, 'severity', finding.severity)}: {diagnosticLabel(
					$t,
					'checks',
					finding.code
				)}
			</p>
			{#if onField && /^(merchant_name|merchant_address|receipt_number|date|time|currency|category|payment_method|subtotal|discount_total|tax_total|tip|total|line_items[.[]\d+\]?\.(description|quantity|unit_price|total_price|spending_category)|taxes[.[]\d+\]?\.(name|rate|amount))$/.test(finding.field_path)}<button
					type="button"
					class="text-sm text-primary underline"
					onclick={() => onField?.(finding.field_path)}>{$t('receiptWorkspace.field')}</button
				>{/if}
			<details class="text-xs text-on-surface-variant">
				<summary>{$t('editor.details')}</summary>{finding.code} · {finding.field_path}
			</details>
			{#each Object.entries(finding.comparison) as [key, value] (key)}<p class="text-sm">
					{diagnosticLabel($t, 'comparison', key)}: {value ?? '—'}
				</p>{/each}
			{#if finding.source_refs.length && onSource}<button
					type="button"
					class="text-primary underline"
					onclick={() => onSource?.(finding.source_refs)}>{$t('review.showSource')}</button
				>{/if}
			{#if finding.severity === 'review' && editable}
				<label class="mt-2 block text-sm"
					>{$t('review.comment')}: {diagnosticLabel($t, 'checks', finding.code)}<textarea
						class="block w-full rounded border p-2"
						bind:value={comments[finding.id]}></textarea></label
				>
			{/if}
		</article>
	{/each}
</div>
