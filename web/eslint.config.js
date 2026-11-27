import js from '@eslint/js';
import globals from 'globals';
import prettier from 'eslint-config-prettier';
import svelte from 'eslint-plugin-svelte';
import svelteParser from 'svelte-eslint-parser';
import tseslint from 'typescript-eslint';

const svelteRunesGlobals = {
	$state: 'readonly',
	$derived: 'readonly',
	$effect: 'readonly',
	$inspect: 'readonly',
	$bindable: 'readonly'
};

export default [
	{
		ignores: [
			'.svelte-kit/**',
			'build/**',
			'out/**',
			'dist/**',
			'node_modules/**',
			'playwright-report/**',
			'test-results/**'
		]
	},
	js.configs.recommended,
	...tseslint.configs.recommended,
	...svelte.configs['flat/recommended'],
	{
		files: ['**/*.svelte', '**/*.svelte.js', '**/*.svelte.ts'],
		languageOptions: {
			parser: svelteParser,
			parserOptions: {
				parser: tseslint.parser
			},
			globals: { ...globals.browser, ...svelteRunesGlobals }
		},
		rules: {
			'@typescript-eslint/no-explicit-any': 'off'
		}
	},
	prettier,
	{
		files: ['**/*.test.ts'],
		languageOptions: {
			globals: {
				describe: 'readonly',
				it: 'readonly',
				expect: 'readonly',
				vi: 'readonly',
				beforeEach: 'readonly',
				afterEach: 'readonly'
			}
		}
	}
];
