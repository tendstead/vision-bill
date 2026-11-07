<script lang="ts">
	import type { IndividualResult } from '$lib/benchmark-experiments';
	let { results }: { results: IndividualResult[] } = $props();
	const percent = (value: number | null | undefined) =>
		value == null ? '—' : `${(value * 100).toFixed(1)}%`;
	const json = (value: unknown) => JSON.stringify(value, null, 2);
</script>

<div class="mt-4 space-y-3">
	{#each results as result (`${result.model_id}:${result.receipt_id}`)}
		<details class="rounded-xl border border-outline-variant p-4">
			<summary class="cursor-pointer text-sm"
				><strong>{result.model_id}</strong> · {result.receipt_id} · {result.success
					? `Extraction ${percent(result.scores?.normalized)}`
					: 'Failed'}</summary
			>
			{#if result.error}<p class="mt-3 text-sm text-error">{result.error}</p>{/if}
			{#if result.success}
				<p class="mt-3 text-sm">
					Strict: {percent(result.scores?.strict)} · Classification: {percent(
						result.scores?.classification
					)} · Missing rows: {result.rows?.missing ?? 0} · Extra rows: {result.rows?.extra ?? 0}
				</p>
				<div class="mt-3 overflow-x-auto">
					<table class="w-full text-left text-sm">
						<thead
							><tr
								><th class="p-2">Field</th><th class="p-2">Verified</th><th class="p-2"
									>Extracted</th
								><th class="p-2">Equivalent value</th></tr
							></thead
						><tbody>
							{#each result.differences ?? [] as difference, i (`${difference.path}:${i}`)}<tr
									class="border-t border-outline-variant"
									><td class="p-2">{difference.path}</td><td class="p-2"
										><pre class="whitespace-pre-wrap">{json(difference.expected)}</pre></td
									><td class="p-2"
										><pre class="whitespace-pre-wrap">{json(difference.actual)}</pre></td
									><td class="p-2">{difference.equivalent ? 'Yes' : 'No'}</td></tr
								>{/each}
						</tbody>
					</table>
				</div>
			{/if}
			<details class="mt-3">
				<summary class="cursor-pointer text-sm">Processing details</summary>
				<pre class="mt-2 overflow-x-auto whitespace-pre-wrap text-xs">{json(result.metadata)}</pre>
			</details>
			{#if result.success}<details class="mt-3">
					<summary class="cursor-pointer text-sm">Verified and extracted JSON</summary>
					<div class="mt-2 grid gap-3 sm:grid-cols-2">
						<pre class="overflow-x-auto whitespace-pre-wrap text-xs">{json(result.expected)}</pre>
						<pre class="overflow-x-auto whitespace-pre-wrap text-xs">{json(result.actual)}</pre>
					</div>
				</details>{/if}
		</details>
	{/each}
</div>
