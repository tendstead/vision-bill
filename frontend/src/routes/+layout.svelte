<script lang="ts">
	import type { Snippet } from 'svelte';
	import { onMount } from 'svelte';
	import { QueryClientProvider, createQuery } from '@tanstack/svelte-query';
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import { goto } from '$app/navigation';
	import '@tendstead/ui/theme.css';
	import '../app.css';
	import { AppShell, type Destination } from '@tendstead/ui';
	import {
		LayoutDashboard,
		Upload,
		Folder,
		ReceiptText,
		Search,
		ChartNoAxesCombined,
		Clock,
		FlaskConical,
		Settings
	} from '@lucide/svelte';
	import { api } from '$lib/api/client';
	import { sessionGeneration } from '$lib/session-lifetime';
	import { initSession, session, signOut, watchSession } from '$lib/auth';
	import { LOCALES, locale, setLocale, t } from '$lib/i18n';
	import { queryClient } from '$lib/query/client';
	import { queryKeys } from '$lib/query/keys';
	import { colorToStyle } from '$lib/collection-colors';
	import Snackbar from '$lib/ui/Snackbar.svelte';
	import Icon from '$lib/ui/Icon.svelte';
	import AppearanceSelect from '$lib/ui/AppearanceSelect.svelte';

	let { children }: { children: Snippet } = $props();

	function onLocaleChange(event: Event) {
		const select = event.currentTarget as HTMLSelectElement;
		const value = select.value;
		if (value === 'en' || value === 'de') {
			void setLocale(value);
		}
	}

	async function handleSignOut() {
		await signOut();
	}

	// Resolved user for rendering; null while loading or signed out.
	let currentUser = $derived.by(() => {
		const s = $session;
		return s === 'loading' || s === null ? null : s;
	});

	let activePath = $derived(page.url.pathname);

	// Populate the session once on start (the HttpOnly cookie is the source of
	// truth; this is just the client's view of it).
	onMount(() => {
		const stopWatching = watchSession();
		void initSession();
		return stopWatching;
	});

	// Auth guard: signed out -> the login page; signed in on /login -> home.
	$effect(() => {
		if ($session === null && activePath !== '/login') void goto(resolve('/login'));
		if (currentUser !== null && activePath === '/login') void goto(resolve('/'));
	});

	const queue = createQuery(
		() => ({
			queryKey: [
				...queryKeys.images({
					limit: 51,
					status: ['pending', 'processing', 'failed', 'timed_out', 'unreadable', 'needs_review']
				}),
				$sessionGeneration
			],
			queryFn: () =>
				api.listImages({
					limit: 51,
					status: ['pending', 'processing', 'failed', 'timed_out', 'unreadable', 'needs_review']
				}),
			staleTime: 30_000,
			refetchInterval: 30_000,
			enabled: currentUser !== null
		}),
		() => queryClient
	);

	let queueCount = $derived(currentUser ? (queue.data?.length ?? 0) : 0);
	let queueLabel = $derived(queueCount > 50 ? '50+' : String(queueCount));

	// The caller's active (auto-capturing) collection, if any — drives the
	// header status dot and is one of the three surfaces kept in sync by
	// invalidating collections()/collectionsActive() on activate/deactivate.
	const active = createQuery(
		() => ({
			queryKey: [...queryKeys.collectionsActive(), $sessionGeneration],
			queryFn: () => api.getActiveCollection(),
			enabled: currentUser !== null
		}),
		() => queryClient
	);
	let activeCollection = $derived(currentUser ? (active.data ?? null) : null);

	let destinations: Destination[] = $derived([
		{
			id: 'dashboard',
			href: resolve('/'),
			label: $t('nav.dashboard'),
			icon: LayoutDashboard,
			match: 'exact'
		},
		{ id: 'upload', href: resolve('/upload'), label: $t('nav.upload'), icon: Upload },
		{
			id: 'collections',
			href: resolve('/collections'),
			label: $t('nav.collections'),
			icon: Folder
		},
		{ id: 'receipts', href: resolve('/receipts'), label: $t('nav.receipts'), icon: ReceiptText },
		{ id: 'search', href: resolve('/search'), label: $t('nav.search'), icon: Search },
		{
			id: 'statistics',
			href: resolve('/statistics'),
			label: $t('nav.statistics'),
			icon: ChartNoAxesCombined
		},
		{
			id: 'queue',
			href: resolve('/queue'),
			label: $t('nav.queue'),
			icon: Clock,
			badge: queueCount ? queueLabel : undefined
		},
		{
			id: 'benchmarks',
			href: resolve('/benchmarks/results'),
			label: $t('nav.benchmarks'),
			icon: FlaskConical,
			permitted: currentUser?.is_admin === true
		},
		{
			id: 'settings',
			href: resolve('/settings'),
			label: $t('nav.settings'),
			icon: Settings,
			permitted: currentUser?.is_admin === true
		}
	]);
</script>

<QueryClientProvider client={queryClient}>
	{#if $session === 'loading'}
		<div class="flex min-h-[calc(100vh-4rem)] items-center justify-center p-8">
			<p class="text-sm text-on-surface-variant">{$t('common.loading')}</p>
		</div>
	{:else if currentUser === null}
		{#if activePath === '/login'}
			<div class="flex min-h-screen flex-col items-center justify-center gap-6 p-4">
				<div class="flex flex-wrap items-center justify-center gap-4">
					<AppearanceSelect />
					<label class="flex items-center gap-2 text-sm"
						>{$t('language.label')}
						<select
							class="rounded-lg border border-outline bg-surface p-2"
							value={$locale}
							onchange={onLocaleChange}
						>
							{#each LOCALES as code (code)}<option value={code}>{code.toUpperCase()}</option
								>{/each}
						</select>
					</label>
				</div>
				{@render children()}
			</div>
		{/if}
	{:else}
		{#key $sessionGeneration}
			<AppShell
				brand={$t('app.name')}
				appLabel={$t('nav.receipts')}
				pageTitle={$t('app.name')}
				pathname={activePath}
				{destinations}
				mobilePrimaryIds={['dashboard', 'upload', 'receipts']}
				labels={{
					menu: $t('nav.menu'),
					closeMenu: $t('nav.closeMenu'),
					allDestinations: $t('nav.allDestinations'),
					skipToContent: $t('nav.skipToContent'),
					sidebar: $t('nav.sidebar'),
					mainNavigation: $t('nav.mainNavigation'),
					mobileNavigation: $t('nav.mobileNavigation'),
					allNavigation: $t('nav.allNavigation'),
					connectedServices: $t('nav.connectedServices'),
					sidebarNote: $t('nav.sidebarNote')
				}}
			>
				{#snippet pageActions()}
					{#if activeCollection}
						<a
							href={resolve('/collections/[id]', { id: activeCollection.id })}
							title={$t('collections.capturing')}
							class="flex min-w-0 items-center gap-2 rounded-full px-3 py-2 text-sm hover:bg-surface-container-high"
						>
							<span class="size-2.5 rounded-full" style={colorToStyle(activeCollection.color)}
							></span>
							<span class="max-w-32 truncate">{activeCollection.name}</span>
						</a>
					{/if}
					{#if queueCount > 0}
						<a
							href={resolve('/queue')}
							title={$t('nav.queue')}
							class="flex items-center gap-1 rounded-full bg-tertiary-container px-3 py-1 text-sm font-medium text-on-tertiary-container"
						>
							<Icon icon="queue" />{queueLabel}
						</a>
					{/if}
				{/snippet}
				{#snippet account()}
					<div class="flex flex-col gap-3 text-sm">
						<span class="break-words text-on-surface-variant"
							>{$t('nav.signedInAs', { values: { name: currentUser.username } })}</span
						>
						<div class="flex flex-wrap items-center gap-2">
							<label class="sr-only" for="locale-select">{$t('language.label')}</label>
							<select
								id="locale-select"
								class="rounded-lg border border-outline-variant bg-surface px-2 py-2 text-sm"
								value={$locale}
								onchange={onLocaleChange}
							>
								{#each LOCALES as code (code)}<option value={code}>{code.toUpperCase()}</option
									>{/each}
							</select>
							<AppearanceSelect />
							<button
								type="button"
								class="rounded-lg border border-outline-variant px-3 py-2 text-sm font-medium hover:bg-surface-container-high"
								onclick={handleSignOut}>{$t('nav.signOut')}</button
							>
						</div>
					</div>
				{/snippet}
				{@render children()}
			</AppShell>
		{/key}
	{/if}

	<Snackbar />
</QueryClientProvider>
