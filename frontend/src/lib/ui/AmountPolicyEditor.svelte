<script lang="ts">
	import { t } from '$lib/i18n';
	import type { AmountPolicy } from '$lib/types';
	let {
		value,
		onChange,
		disabled = false
	}: {
		value: AmountPolicy;
		onChange: (value: AmountPolicy) => void;
		disabled?: boolean;
	} = $props();
	const choices = {
		tax_basis: ['unknown', 'gross', 'net'],
		discount_treatment: ['unknown', 'included_in_lines', 'subtract'],
		tip_treatment: ['unknown', 'included', 'add', 'outside_receipt', 'none'],
		subtotal_origin: ['unknown', 'printed', 'printed_net', 'derived_net', 'derived_gross'],
		tax_information: ['missing', 'present', 'explicit_zero']
	};
</script>

<fieldset class="my-4 grid gap-3 rounded-xl border border-outline-variant p-4" {disabled}>
	<legend>{$t('review.amountPolicy')}</legend>
	{#each Object.entries(choices) as [key, options] (key)}
		<label
			>{$t(`review.policy.${key}`)}<select
				class="ml-2 rounded border p-2"
				value={value[key as keyof AmountPolicy]}
				onchange={(e) => onChange({ ...value, [key]: e.currentTarget.value })}
				>{#each options as option (option)}<option value={option}
						>{$t(`review.policyValue.${option}`)}</option
					>{/each}</select
			></label
		>
	{/each}
</fieldset>
