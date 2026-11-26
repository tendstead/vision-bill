<script lang="ts">
	import { t } from '$lib/i18n';
	let {
		busy,
		oncreate
	}: { busy: boolean; oncreate: (name: string, photos: File[]) => Promise<void> } = $props();
	let name = $state('');
	let photos = $state<File[]>([]);
	let error = $state('');
	function append(input: HTMLInputElement) {
		const additions = Array.from(input.files ?? []);
		input.value = '';
		if (
			photos.length + additions.length > 10 ||
			additions.some((file) => file.size > 20 * 1024 * 1024)
		) {
			error = 'submission.limits';
			return;
		}
		error = '';
		photos = [...photos, ...additions];
	}
	function move(index: number, delta: number) {
		const next = photos.slice();
		[next[index], next[index + delta]] = [next[index + delta], next[index]];
		photos = next;
	}
</script>

<form
	class="case-panel space-y-4"
	onsubmit={(e) => {
		e.preventDefault();
		if (!busy && name.trim() && photos.length) void oncreate(name.trim(), photos);
	}}
>
	<fieldset disabled={busy} class="space-y-4">
		<legend class="text-xl font-semibold">{$t('benchmarkCase.new')}</legend>
		<label>{$t('benchmarkCase.name')}<input required maxlength="160" bind:value={name} /></label>
		<label
			>{$t('submission.photos')}<input
				type="file"
				accept="image/*,.heic,.heif,.avif"
				multiple
				onchange={(e) => append(e.currentTarget)}
			/></label
		>
		<p class="text-sm text-on-surface-variant">{$t('submission.captureHint')}</p>
		{#if photos.length}
			<ol class="space-y-2">
				{#each photos as photo, index (index)}
					<li
						class="flex flex-wrap items-center gap-2 rounded-lg border border-outline-variant p-2"
					>
						<span class="min-w-0 flex-1 break-all">{index + 1}. {photo.name}</span>
						<button
							type="button"
							class="case-secondary"
							aria-label={$t('submission.moveUp')}
							disabled={index === 0}
							onclick={() => move(index, -1)}>↑</button
						>
						<button
							type="button"
							class="case-secondary"
							aria-label={$t('submission.moveDown')}
							disabled={index === photos.length - 1}
							onclick={() => move(index, 1)}>↓</button
						>
						<button
							type="button"
							class="case-secondary"
							onclick={() => (photos = photos.filter((_, i) => i !== index))}
							>{$t('common.remove')}</button
						>
					</li>
				{/each}
			</ol>
		{/if}
		{#if error}<p class="text-error" role="alert">{$t(error)}</p>{/if}
		<button class="case-primary" disabled={!name.trim() || !photos.length}
			>{$t('benchmarkCase.create')}</button
		>
	</fieldset>
</form>
