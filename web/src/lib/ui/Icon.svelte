<script lang="ts">
	export type IconName =
		| 'dashboard'
		| 'queue'
		| 'upload'
		| 'receipts'
		| 'search'
		| 'settings'
		| 'theme-light'
		| 'theme-dark'
		| 'close'
		| 'camera'
		| 'trash'
		| 'check'
		| 'refresh'
		| 'plus'
		| 'folder'
		| 'alert'
		| 'edit'
		| 'remove';

	let { icon }: { icon: IconName } = $props();
</script>

<svg
	class="size-5 shrink-0"
	viewBox="0 0 24 24"
	fill="none"
	stroke="currentColor"
	stroke-width="2"
	stroke-linecap="round"
	stroke-linejoin="round"
	aria-hidden="true"
>
	{#if icon === 'dashboard'}
		<rect x="3" y="3" width="7" height="9" rx="1" />
		<rect x="14" y="3" width="7" height="5" rx="1" />
		<rect x="14" y="12" width="7" height="9" rx="1" />
		<rect x="3" y="16" width="7" height="5" rx="1" />
	{:else if icon === 'queue'}
		<circle cx="12" cy="12" r="9" />
		<path d="M12 7v5l3 3" />
	{:else if icon === 'upload'}
		<path d="M12 16V4" />
		<path d="m6 10 6-6 6 6" />
		<path d="M4 20h16" />
	{:else if icon === 'receipts'}
		<path d="M6 3h12v18l-3-2-3 2-3-2-3 2Z" />
		<path d="M9 8h6" />
		<path d="M9 12h6" />
	{:else if icon === 'search'}
		<circle cx="11" cy="11" r="7" />
		<path d="m20 20-3.5-3.5" />
	{:else if icon === 'settings'}
		<path d="M4 6h16" />
		<path d="M4 12h16" />
		<path d="M4 18h16" />
		<circle cx="15" cy="6" r="2" />
		<circle cx="8" cy="12" r="2" />
		<circle cx="17" cy="18" r="2" />
	{:else if icon === 'theme-light'}
		<circle cx="12" cy="12" r="4" />
		<path d="M12 2v2" />
		<path d="M12 20v2" />
		<path d="m4.9 4.9 1.4 1.4" />
		<path d="m17.7 17.7 1.4 1.4" />
		<path d="M2 12h2" />
		<path d="M20 12h2" />
		<path d="m4.9 19.1 1.4-1.4" />
		<path d="m17.7 6.3 1.4-1.4" />
	{:else if icon === 'theme-dark'}
		<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" />
	{:else if icon === 'close'}
		<path d="m6 6 12 12" />
		<path d="M18 6 6 18" />
	{:else if icon === 'camera'}
		<path d="M4 8h3l2-3h6l2 3h3v12H4Z" />
		<circle cx="12" cy="13" r="3" />
	{:else if icon === 'trash'}
		<path d="M4 7h16" />
		<path d="M10 11v6" />
		<path d="M14 11v6" />
		<path d="M6 7l1 13h10l1-13" />
		<path d="M9 7V4h6v3" />
	{:else if icon === 'check'}
		<path d="m5 13 4 4L19 7" />
	{:else if icon === 'refresh'}
		<path d="M20 11a8 8 0 1 0-2.3 6.3" />
		<path d="M20 4v7h-7" />
	{:else if icon === 'plus'}
		<path d="M12 5v14" />
		<path d="M5 12h14" />
	{:else if icon === 'alert'}
		<path d="M12 3 2 21h20Z" />
		<path d="M12 10v5" />
		<path d="M12 18h.01" />
	{:else if icon === 'folder'}
		<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z" />
	{:else if icon === 'edit'}
		<path d="M12 20h9" />
		<path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
	{:else if icon === 'remove'}
		<circle cx="12" cy="12" r="9" />
		<path d="M8 12h8" />
	{/if}
</svg>
