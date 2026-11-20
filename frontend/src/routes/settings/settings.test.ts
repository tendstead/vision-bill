import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { api } from '$lib/api/client';
import { queryClient } from '$lib/query/client';
import { session, signIn } from '$lib/auth';
import { queryKeys } from '$lib/query/keys';
import type { SettingsView } from '$lib/types';
import Settings from './+page.svelte';
vi.mock('$app/paths', () => ({ resolve: (p: string) => p }));
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));
beforeEach(() => {
	vi.spyOn(api, 'getDedicatedOcr').mockResolvedValue({ state: 'unavailable' });
});
afterEach(() => {
	cleanup();
	queryClient.clear();
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
});
it.each([true, false])(
	'discovers appropriate models with OCR enabled=%s, preserving typed selection and distinct limits',
	async (ocr) => {
		session.set({ id: 'admin', username: 'admin', is_admin: true, can_see_all: true });
		vi.spyOn(api, 'getSettings').mockResolvedValue({
			llm: { provider: 'ollama', host: 'local', model_name: 'configured', temperature: 0 },
			allow_registration: true,
			sources: {},
			restart_required: false
		});
		vi.spyOn(api, 'getUiConfig').mockResolvedValue({
			bypass_review_default: false,
			registration_open: true,
			ocr_enabled: ocr,
			formatter_max_output_tokens: 8192
		});
		vi.spyOn(api, 'listModels').mockResolvedValue([
			{ id: 'formatter', context_window_tokens: 32768 }
		]);
		render(Settings);
		await waitFor(() => expect(api.listModels).toHaveBeenCalledWith(ocr ? 'text' : 'vision'));
		expect(document.querySelector('option[value="anthropic"]')).toBeNull();
		expect(document.querySelector('option[value="ollama"]')).toBeTruthy();
		expect(document.querySelector('option[value="openai"]')).toBeTruthy();
		const model = screen.getByLabelText(/Default model/);
		await fireEvent.input(model, { target: { value: 'custom-model' } });
		await queryClient.invalidateQueries({ queryKey: ['models'] });
		expect((model as HTMLInputElement).value).toBe('custom-model');
		expect(model.getAttribute('list')).toBe('formatter-models');
		expect(screen.getByText(/8192/)).toBeTruthy();
		expect(document.querySelector('option[value="formatter"]')?.textContent).toContain('32768');
	}
);

it('does not repopulate the settings cache when the actor changes after response parsing', async () => {
	const admin = { id: 'admin', username: 'admin', is_admin: true, can_see_all: true };
	session.set(admin);
	const settings: SettingsView = {
		llm: { provider: 'ollama', host: 'local', model_name: 'original', temperature: 0 },
		allow_registration: true,
		sources: {},
		restart_required: false
	};
	vi.spyOn(api, 'getSettings').mockResolvedValue(settings);
	vi.spyOn(api, 'getUiConfig').mockResolvedValue({
		bypass_review_default: false,
		registration_open: true
	});
	vi.spyOn(api, 'listModels').mockResolvedValue([]);
	vi.spyOn(api, 'login').mockResolvedValue({ ...admin, id: 'other-admin', username: 'other' });

	let resolveBody!: (settings: SettingsView) => void;
	const body = new Promise<SettingsView>((resolve) => (resolveBody = resolve));
	const json = vi.fn(() => body);
	vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json }));
	render(Settings);
	await screen.findByLabelText(/Default model/);
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	await waitFor(() => expect(json).toHaveBeenCalledOnce());

	resolveBody({ ...settings, llm: { ...settings.llm, model_name: 'origin save' } });
	// The already-queued client continuation validates A. The UI's continuation
	// follows this microtask and must observe the session switch.
	await new Promise<void>((resolve, reject) => {
		queueMicrotask(() => void signIn('other', 'secret').then(() => resolve(), reject));
	});
	expect(queryClient.getQueryData(queryKeys.settings())).toBeUndefined();
	expect(screen.queryByText('Settings saved.')).toBeNull();
});

it('lets an admin explicitly save environment defaults and keeps restart guidance visible', async () => {
	session.set({ id: 'admin', username: 'admin', is_admin: true, can_see_all: true });
	const initial: SettingsView = {
		llm: { provider: 'ollama', host: 'http://local', model_name: 'original', temperature: 0 },
		allow_registration: true,
		sources: {
			'llm.provider': 'environment',
			'llm.host': 'environment',
			'llm.model_name': 'environment',
			'llm.temperature': 'environment'
		},
		restart_required: false
	};
	vi.spyOn(api, 'getSettings').mockResolvedValue(initial);
	vi.spyOn(api, 'getUiConfig').mockResolvedValue({
		bypass_review_default: false,
		registration_open: true
	});
	vi.spyOn(api, 'listModels').mockResolvedValue([]);
	vi.spyOn(api, 'updateSettings').mockResolvedValue({ ...initial, restart_required: true });
	render(Settings);
	const model = await screen.findByLabelText(/Default model/);
	expect((model as HTMLInputElement).disabled).toBe(false);
	for (const field of document.querySelectorAll('form input, form select'))
		expect((field as HTMLInputElement).disabled).toBe(false);
	await fireEvent.input(model, { target: { value: 'edited-model' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	expect(await screen.findByText(/Settings saved/)).toBeTruthy();
	expect(api.updateSettings).toHaveBeenCalledWith(
		expect.objectContaining({ changed_paths: ['llm.model_name'] })
	);
	await waitFor(() => expect(api.getDedicatedOcr).toHaveBeenCalledTimes(2));
	expect(screen.getByText(/Ask your operator to restart the backend/)).toBeTruthy();
});

it('adopts a write-only bearer key and shows the saved connection result', async () => {
	session.set({ id: 'admin', username: 'admin', is_admin: true, can_see_all: true });
	const initial = {
		llm: {
			provider: 'openai',
			host: 'https://llm.example/v1',
			model_name: 'formatter',
			temperature: 0
		},
		allow_registration: true,
		sources: {},
		restart_required: false,
		llm_authentication: { mode: 'bearer', source: 'environment', status: 'configured' }
	} as const;
	const saved = {
		...initial,
		llm_authentication: { ...initial.llm_authentication, source: 'settings' },
		restart_required: true
	};
	vi.spyOn(api, 'getSettings').mockResolvedValue(initial);
	vi.spyOn(api, 'getUiConfig').mockResolvedValue({
		bypass_review_default: false,
		registration_open: true
	});
	vi.spyOn(api, 'listModels').mockResolvedValue([]);
	const fetch = vi.fn().mockImplementation((path: string) =>
		Promise.resolve({
			ok: true,
			status: 200,
			json: async () =>
				path.endsWith('/connection-test') ? { status: 'authentication_failed' } : saved
		})
	);
	vi.stubGlobal('fetch', fetch);
	render(Settings);
	const mode = await screen.findByLabelText('Authentication');
	expect(screen.getByText(/Environment default/)).toBeTruthy();
	await fireEvent.change(mode, { target: { value: 'bearer' } });
	const key = screen.getByLabelText('New API key or access token');
	expect(key.getAttribute('type')).toBe('password');
	await fireEvent.input(key, { target: { value: 'new-private-token' } });
	await fireEvent.click(screen.getByRole('button', { name: 'Save' }));
	await screen.findByText(/Settings saved/);
	const update = JSON.parse(fetch.mock.calls[0][1].body);
	expect(update.llm_authentication).toEqual({ mode: 'bearer', api_key: 'new-private-token' });
	expect(update.changed_paths).toEqual([]);
	expect(screen.queryByLabelText('New API key or access token')).toBeNull();
	await fireEvent.click(screen.getByRole('button', { name: 'Test saved connection' }));
	expect(await screen.findByText('The endpoint rejected the configured credentials.')).toBeTruthy();
	expect(JSON.parse(fetch.mock.calls[1][1].body)).toEqual({ host: initial.llm.host });
});
