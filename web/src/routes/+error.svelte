<script lang="ts">
	import { page } from '$app/state';
	import { resolve } from '$app/paths';

	let status = $derived(page.status);
	let notFound = $derived(status === 404);
	let title = $derived(notFound ? 'Page not found' : 'Something went wrong');
	let body = $derived(
		notFound
			? 'The page you requested does not exist or is no longer available.'
			: 'We could not load this page. Please try again.'
	);
</script>

<svelte:head>
	<title>{status} · Vision Bill</title>
</svelte:head>

<main class="flex min-h-screen items-center justify-center bg-surface p-6 text-on-surface">
	<section
		class="w-full max-w-md rounded-xl border border-outline-variant bg-surface-container-low p-8 text-center"
	>
		<p class="text-sm font-semibold text-primary">{status}</p>
		<h1 class="mt-2 text-2xl font-semibold">{title}</h1>
		<p class="mt-3 text-sm text-on-surface-variant">{body}</p>
		<a
			href={resolve('/')}
			class="mt-6 inline-block rounded-lg bg-primary px-4 py-2.5 text-sm font-medium text-on-primary hover:opacity-90"
		>
			Return to dashboard
		</a>
	</section>
</main>
