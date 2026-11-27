import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

export default defineConfig({
	plugins: [sveltekit()],
	// Component tests render client-side: resolve the `browser` build of svelte
	// (without this, vitest resolves the SSR entry and `mount` is unavailable).
	resolve: {
		conditions: ['browser']
	},
	test: {
		environment: 'jsdom',
		include: ['src/**/*.{test,spec}.{ts,js}']
	}
});
