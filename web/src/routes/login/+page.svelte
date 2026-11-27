<script lang="ts">
	import { createQuery } from '@tanstack/svelte-query';
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { api } from '$lib/api/client';
	import { signIn, signUp } from '$lib/auth';
	import { t, translate } from '$lib/i18n';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';

	type Mode = 'login' | 'register';

	let mode = $state<Mode>('login');
	let username = $state('');
	let password = $state('');
	let busy = $state(false);
	let error = $state<string | null>(null);

	const uiConfig = createQuery(
		() => ({ queryKey: queryKeys.uiConfig(), queryFn: () => api.getUiConfig() }),
		() => queryClient
	);

	let registrationOpen = $derived(uiConfig.data?.registration_open ?? false);

	function switchMode(next: Mode) {
		mode = next;
		error = null;
	}

	async function submit() {
		if (busy) return;
		busy = true;
		error = null;
		try {
			if (mode === 'login') {
				await signIn(username, password);
			} else {
				await signUp(username, password);
			}
			void goto(resolve('/'));
		} catch (err) {
			error = err instanceof Error ? err.message : translate('login.error');
		} finally {
			busy = false;
		}
	}
</script>

<svelte:head>
	<title>{$t('login.title')} – {$t('app.name')}</title>
</svelte:head>

<section
	class="w-full max-w-md rounded-2xl border border-outline-variant bg-surface-container-low p-8 shadow-elevation-1"
>
	<h1 class="text-2xl font-semibold">{$t('login.title')}</h1>
	<p class="mt-1 text-sm text-on-surface-variant">{$t('app.name')}</p>

	<form
		class="mt-6 flex flex-col gap-4"
		onsubmit={(event) => {
			event.preventDefault();
			void submit();
		}}
	>
		<div>
			<label for="username" class="text-sm font-medium">{$t('login.username')}</label>
			<input
				id="username"
				type="text"
				autocomplete="username"
				bind:value={username}
				required
				class="mt-1 w-full rounded-lg border border-outline-variant bg-surface px-3 py-2 text-sm"
			/>
		</div>
		<div>
			<label for="password" class="text-sm font-medium">{$t('login.password')}</label>
			<input
				id="password"
				type="password"
				autocomplete={mode === 'login' ? 'current-password' : 'new-password'}
				bind:value={password}
				required
				class="mt-1 w-full rounded-lg border border-outline-variant bg-surface px-3 py-2 text-sm"
			/>
		</div>

		{#if error}
			<p
				role="alert"
				class="rounded-lg bg-error-container px-3 py-2 text-sm text-on-error-container"
			>
				{error}
			</p>
		{/if}

		<button
			type="submit"
			disabled={busy}
			class="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-on-primary disabled:opacity-60"
		>
			{#if busy}
				{$t(mode === 'login' ? 'login.signingIn' : 'login.creating')}
			{:else}
				{$t(mode === 'login' ? 'login.signIn' : 'login.createAccount')}
			{/if}
		</button>
	</form>

	{#if mode === 'login' && registrationOpen}
		<p class="mt-4 text-center text-sm text-on-surface-variant">
			{$t('login.noAccount')}
			<button
				type="button"
				class="font-medium text-primary hover:underline"
				onclick={() => switchMode('register')}
			>
				{$t('login.switchToRegister')}
			</button>
		</p>
	{:else if mode === 'register'}
		<p class="mt-4 text-center text-sm text-on-surface-variant">
			{$t('login.haveAccount')}
			<button
				type="button"
				class="font-medium text-primary hover:underline"
				onclick={() => switchMode('login')}
			>
				{$t('login.switchToLogin')}
			</button>
		</p>
	{/if}
</section>
