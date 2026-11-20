import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, expect, it, vi } from 'vitest';
import { api, ApiError } from '$lib/api/client';
import { signIn, signUp } from '$lib/auth';
import { queryClient } from '$lib/query/client';
import Login from './+page.svelte';
vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));
vi.mock('$lib/auth', () => ({ signIn: vi.fn(), signUp: vi.fn() }));
afterEach(() => {
	cleanup();
	queryClient.clear();
	vi.restoreAllMocks();
});
it('names registration and associates rejected credentials with inline feedback', async () => {
	vi.spyOn(api, 'getUiConfig').mockResolvedValue({ registration_open: true } as never);
	vi.mocked(signUp).mockRejectedValue(new ApiError(409, 'Username already exists'));
	render(Login);
	await fireEvent.click(await screen.findByRole('button', { name: 'Create one' }));
	expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Create account');
	await waitFor(() => expect(document.title).toContain('Create account'));
	await fireEvent.input(screen.getByLabelText('Username'), { target: { value: 'alice' } });
	await fireEvent.input(screen.getByLabelText('Password'), { target: { value: 'password' } });
	await fireEvent.submit(screen.getByRole('button', { name: 'Create account' }).closest('form')!);
	expect((await screen.findByRole('alert')).id).toBe('login-error');
	expect(screen.getByLabelText('Username').getAttribute('aria-describedby')).toBe('login-error');
	expect(screen.getByLabelText('Password').getAttribute('aria-invalid')).toBeNull();
});

it.each([
	['login', 401, true, true],
	['login', 0, false, false],
	['register', 403, false, false],
	['register', 409, true, false]
] as const)(
	'targets fields only for %s status %s',
	async (mode, status, usernameInvalid, passwordInvalid) => {
		vi.spyOn(api, 'getUiConfig').mockResolvedValue({ registration_open: true } as never);
		vi.mocked(mode === 'login' ? signIn : signUp).mockRejectedValue(
			new ApiError(status, 'Request failed')
		);
		render(Login);
		if (mode === 'register')
			await fireEvent.click(await screen.findByRole('button', { name: 'Create one' }));
		await fireEvent.submit(document.querySelector('form')!);
		await screen.findByRole('alert');
		for (const [label, invalid] of [
			['Username', usernameInvalid],
			['Password', passwordInvalid]
		] as const) {
			expect(screen.getByLabelText(label).getAttribute('aria-invalid')).toBe(
				invalid ? 'true' : null
			);
			expect(screen.getByLabelText(label).getAttribute('aria-describedby')).toBe(
				invalid ? 'login-error' : null
			);
		}
	}
);
