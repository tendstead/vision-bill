<script lang="ts">
	import { t } from '$lib/i18n';
	import { theme, isTheme, setTheme } from '$lib/theme';
</script>

<label class="flex flex-wrap items-center gap-2 text-sm font-medium">
	<span>{$t('theme.label')}</span>
	<select
		class="min-h-10 rounded-lg border border-outline bg-surface px-3 py-2 text-on-surface"
		value={$theme}
		onchange={(event) => {
			const value = event.currentTarget.value;
			if (isTheme(value)) setTheme(value);
		}}
	>
		<option value="light">{$t('theme.light')}</option>
		<option value="black">{$t('theme.black')}</option>
		<option value="dark">{$t('theme.purple')}</option>
	</select>
</label>
