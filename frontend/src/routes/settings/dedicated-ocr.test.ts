import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { api } from '$lib/api/client';
import { session } from '$lib/auth';
import { setLocale } from '$lib/i18n';
import { queryClient } from '$lib/query/client';
import Settings from './+page.svelte';

vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
vi.mock('$app/navigation', () => ({ goto: vi.fn() }));

const admin = { id: 'admin', username: 'admin', is_admin: true, can_see_all: true };
const directVision = {
	bypass_review_default: false,
	registration_open: true,
	ocr_enabled: false,
	formatter_max_output_tokens: 32768
};

beforeEach(() => {
	setLocale('en');
	session.set(admin);
	vi.spyOn(api, 'getSettings').mockResolvedValue({
		llm: { provider: 'ollama', host: 'local', model_name: 'configured', temperature: 0 },
		allow_registration: true,
		sources: {},
		restart_required: false
	});
	vi.spyOn(api, 'getUiConfig').mockResolvedValue(directVision);
	vi.spyOn(api, 'listModels').mockResolvedValue([]);
});

afterEach(() => {
	cleanup();
	queryClient.clear();
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
});

function status(
	state: string,
	reason: string | null = null,
	httpStatus = 200,
	installed = false,
	shared_gpu = true
) {
	return Response.json({ state, reason, installed, shared_gpu }, { status: httpStatus });
}

function handoffConfirmation() {
	return screen.findByRole('checkbox', {
		name: 'I confirm the LLM endpoint is dedicated to Vision Bill and idle, with eager model loading turned off.'
	});
}

it('can save separate GPUs before activating OCR without confirming LLM handoff', async () => {
	const fetchStatus = vi
		.fn()
		.mockResolvedValueOnce(status('action_required', 'llm_lifecycle_required'))
		.mockResolvedValueOnce(status('disabled', null, 200, false, false))
		.mockResolvedValueOnce(status('requested', null, 202, false, false));
	vi.stubGlobal('fetch', fetchStatus);

	render(Settings);
	const shared = await screen.findByRole('checkbox', { name: 'LLM and OCR share a GPU' });
	expect(shared).toHaveProperty('checked', true);
	const enable = screen.getByRole('switch', { name: 'Dedicated OCR' });
	await fireEvent.click(shared);
	expect(enable).toHaveProperty('disabled', true);
	expect(screen.queryByRole('checkbox', { name: /I confirm the LLM endpoint/ })).toBeNull();
	await fireEvent.click(screen.getByRole('button', { name: 'Save GPU configuration' }));

	await waitFor(() => expect(enable).toHaveProperty('disabled', false));
	expect(shared).toHaveProperty('checked', false);
	const [url, options] = fetchStatus.mock.calls[1];
	expect(url).toBe('/api/v1/system/dedicated-ocr/gpu-topology');
	expect(options.method).toBe('PUT');
	expect(new Headers(options.headers).get('X-Vision-Bill-Expected-User')).toBe('admin');
	expect(JSON.parse(options.body)).toEqual({ shared_gpu: false, confirm_handoff: false });
	await fireEvent.click(enable);
	await screen.findByText('Dedicated OCR setup is queued.');
	expect(fetchStatus.mock.calls[2][1].body).toBe('{"enabled":true,"confirm_handoff":false}');
});

it('requires a fresh handoff confirmation before using a shared GPU with installed models', async () => {
	const fetchStatus = vi
		.fn()
		.mockResolvedValueOnce(status('disabled', null, 200, true, false))
		.mockResolvedValueOnce(status('disabled', null, 200, true));
	vi.stubGlobal('fetch', fetchStatus);

	render(Settings);
	const shared = await screen.findByRole('checkbox', { name: 'LLM and OCR share a GPU' });
	expect(shared).toHaveProperty('checked', false);
	await fireEvent.click(shared);
	const save = screen.getByRole('button', { name: 'Save GPU configuration' });
	expect(save).toHaveProperty('disabled', true);
	await fireEvent.click(await handoffConfirmation());
	await fireEvent.click(shared);
	await fireEvent.click(shared);
	expect(save).toHaveProperty('disabled', true);
	await fireEvent.click(save);
	expect(fetchStatus).toHaveBeenCalledOnce();
	await fireEvent.click(await handoffConfirmation());
	await fireEvent.click(save);

	await waitFor(() =>
		expect(screen.getByRole('switch', { name: 'Dedicated OCR' })).toHaveProperty('disabled', false)
	);
	expect(JSON.parse(fetchStatus.mock.calls[1][1].body)).toEqual({
		shared_gpu: true,
		confirm_handoff: true
	});
	expect(screen.queryByRole('checkbox', { name: /I confirm the LLM endpoint/ })).toBeNull();
});

it('explains a rejected topology change without applying the unsaved choice', async () => {
	const fetchStatus = vi
		.fn()
		.mockResolvedValueOnce(status('active', null, 200, true))
		.mockResolvedValueOnce(
			Response.json({ detail: 'llm_request_recovery_required' }, { status: 409 })
		)
		.mockResolvedValueOnce(status('active', null, 200, true));
	vi.stubGlobal('fetch', fetchStatus);

	render(Settings);
	const shared = await screen.findByRole('checkbox', { name: 'LLM and OCR share a GPU' });
	await fireEvent.click(shared);
	await fireEvent.click(screen.getByRole('button', { name: 'Save GPU configuration' }));
	await screen.findByText(
		'Confirm the previous LLM request has ended and follow the Dedicated OCR recovery guide.'
	);
	expect(shared).toHaveProperty('checked', true);
	expect(screen.getByRole('switch', { name: 'Dedicated OCR' }).getAttribute('aria-checked')).toBe(
		'true'
	);
	expect(fetchStatus).toHaveBeenCalledTimes(3);
});

it.each([
	[
		'model_handoff_foreign_residency',
		'Unload all models at the LLM endpoint before switching to a shared GPU.'
	],
	[
		'model_handoff_residency_unknown',
		'Check that the LLM endpoint can report which models are loaded, then retry.'
	],
	['model_handoff_busy', 'Wait for the current GPU handoff to finish, then retry.'],
	[
		'model_release_unconfirmed',
		'Check that the LLM endpoint has unloaded its models and follow the recovery guide before retrying.'
	]
])('explains why a return to a shared GPU was blocked by %s', async (reason, message) => {
	const fetchStatus = vi
		.fn()
		.mockResolvedValueOnce(status('active', null, 200, true, false))
		.mockResolvedValueOnce(Response.json({ detail: reason }, { status: 409 }))
		.mockResolvedValueOnce(status('active', null, 200, true, false));
	vi.stubGlobal('fetch', fetchStatus);

	render(Settings);
	const shared = await screen.findByRole('checkbox', { name: 'LLM and OCR share a GPU' });
	await fireEvent.click(shared);
	await fireEvent.click(await handoffConfirmation());
	await fireEvent.click(screen.getByRole('button', { name: 'Save GPU configuration' }));

	expect((await screen.findByRole('alert')).textContent).toBe(message);
	expect(shared).toHaveProperty('checked', false);
	expect(fetchStatus).toHaveBeenCalledTimes(3);
	expect(fetchStatus.mock.calls[2][1].method).toBeUndefined();
});

it('reports local OCR memory exhaustion during setup with the German GPU choice', async () => {
	setLocale('de');
	vi.stubGlobal(
		'fetch',
		vi.fn().mockResolvedValue(status('failed', 'ocr_gpu_out_of_memory', 200, false, false))
	);

	try {
		render(Settings);
		await screen.findByText('Die lokale OCR-GPU hat nicht genug Grafikspeicher.');
		expect(
			screen.getByRole('checkbox', { name: 'LLM und OCR teilen sich eine GPU' })
		).toHaveProperty('checked', false);
	} finally {
		setLocale('en');
	}
});

it('does not load Dedicated OCR setup for a non-admin who visits Settings', async () => {
	session.set({ ...admin, is_admin: false, can_see_all: false });
	const fetchStatus = vi.fn();
	vi.stubGlobal('fetch', fetchStatus);

	render(Settings);
	await screen.findByRole('heading', { name: 'Settings' });

	expect(screen.queryByRole('region', { name: 'Dedicated OCR' })).toBeNull();
	expect(fetchStatus).not.toHaveBeenCalled();
});

it('shows that a GPU profile is required and offers no activation', async () => {
	const fetchStatus = vi.fn().mockResolvedValue(status('unavailable', 'gpu_profile_required'));
	vi.stubGlobal('fetch', fetchStatus);

	render(Settings);
	await screen.findByText('Select a CUDA or ROCm profile to use Dedicated OCR.');

	expect(screen.getByRole('switch', { name: 'Dedicated OCR' })).toHaveProperty('disabled', true);
	expect(fetchStatus).toHaveBeenCalledOnce();
	const [url, options] = fetchStatus.mock.calls[0];
	expect(url).toBe('/api/v1/system/dedicated-ocr');
	expect(options.method).toBeUndefined();
	expect(options.headers.get('X-Vision-Bill-Expected-User')).toBe('admin');
});

it('submits one explicit activation and polls until the formatter shows OCR limits', async () => {
	let finishRequest!: (response: Response) => void;
	const post = new Promise<Response>((resolve) => (finishRequest = resolve));
	const fetchStatus = vi.fn(async (_url: string, options: RequestInit) =>
		options.method === 'POST' ? post : status('disabled')
	);
	vi.stubGlobal('fetch', fetchStatus);

	render(Settings);
	const enable = await screen.findByRole('switch', { name: 'Dedicated OCR' });
	await waitFor(() => expect(screen.queryByText(/Reported OCR output limit/)).toBeNull());
	expect((enable as HTMLButtonElement).disabled).toBe(true);
	await fireEvent.click(enable);
	expect(fetchStatus).toHaveBeenCalledOnce();
	await fireEvent.click(await handoffConfirmation());

	await Promise.all([fireEvent.click(enable), fireEvent.click(enable)]);
	const posts = fetchStatus.mock.calls.filter(([, options]) => options.method === 'POST');
	expect(posts).toHaveLength(1);
	expect(posts[0][0]).toBe('/api/v1/system/dedicated-ocr');
	expect(new Headers(posts[0][1].headers).get('X-Vision-Bill-Expected-User')).toBe('admin');
	expect(posts[0][1].body).toBe('{"enabled":true,"confirm_handoff":true}');
	expect(enable.getAttribute('aria-checked')).toBe('false');
	expect(enable.getAttribute('aria-busy')).toBe('true');
	expect((enable as HTMLButtonElement).disabled).toBe(true);

	finishRequest(status('requested', null, 202));
	await screen.findByText('Dedicated OCR setup is queued.');
	vi.mocked(api.getUiConfig).mockResolvedValue({ ...directVision, ocr_enabled: true });
	fetchStatus.mockResolvedValueOnce(status('installing')).mockResolvedValueOnce(status('active'));
	await screen.findByText(
		'Downloading and installing Dedicated OCR models…',
		{},
		{ timeout: 3_000 }
	);
	expect(screen.queryByText(/Reported OCR output limit/)).toBeNull();
	await screen.findByText(
		'Dedicated OCR is ready to process receipts. The first request loads the model on the GPU.',
		{},
		{ timeout: 3_000 }
	);
	await screen.findByText(/Reported OCR output limit/);
	const requestCount = fetchStatus.mock.calls.length;
	await new Promise((resolve) => setTimeout(resolve, 2_100));
	expect(fetchStatus).toHaveBeenCalledTimes(requestCount);
}, 10_000);

it('offers an explicit retry after a failed download', async () => {
	const fetchStatus = vi
		.fn()
		.mockResolvedValueOnce(status('failed', 'ocr_download_failed'))
		.mockResolvedValue(status('requested', null, 202));
	vi.stubGlobal('fetch', fetchStatus);

	render(Settings);
	await screen.findByText('Dedicated OCR model download failed.');
	const retry = await screen.findByRole('switch', { name: 'Dedicated OCR' });
	expect((retry as HTMLButtonElement).disabled).toBe(true);
	await fireEvent.click(retry);
	expect(fetchStatus).toHaveBeenCalledOnce();
	await fireEvent.click(await handoffConfirmation());
	await fireEvent.click(retry);
	await screen.findByText('Dedicated OCR setup is queued.');
	expect(fetchStatus.mock.calls[1][1].method).toBe('POST');
});

it('refreshes status when activation conflicts with an operator change', async () => {
	const fetchStatus = vi
		.fn()
		.mockResolvedValueOnce(status('disabled'))
		.mockResolvedValueOnce(Response.json({ detail: 'llm_lifecycle_required' }, { status: 409 }))
		.mockResolvedValueOnce(status('action_required', 'llm_lifecycle_required'))
		.mockResolvedValueOnce(status('requested', null, 202));
	vi.stubGlobal('fetch', fetchStatus);

	render(Settings);
	await fireEvent.click(await handoffConfirmation());
	await fireEvent.click(await screen.findByRole('switch', { name: 'Dedicated OCR' }));
	await screen.findByText('Configure a supported LLM lifecycle before setting up Dedicated OCR.');
	const retry = await screen.findByRole('switch', { name: 'Dedicated OCR' });
	await fireEvent.click(retry);
	expect(fetchStatus).toHaveBeenCalledTimes(3);
	await fireEvent.click(await handoffConfirmation());
	await fireEvent.click(retry);
	await screen.findByText('Dedicated OCR setup is queued.');
	expect(fetchStatus.mock.calls[3][1].method).toBe('POST');
});

it('checks an unavailable installed runtime without offering another setup', async () => {
	const fetchStatus = vi
		.fn()
		.mockResolvedValueOnce(status('action_required', 'ocr_runtime_unavailable'))
		.mockResolvedValueOnce(status('active'));
	vi.stubGlobal('fetch', fetchStatus);

	render(Settings);
	await screen.findByText(
		'Dedicated OCR is installed. Restore its service before processing receipts.'
	);
	expect(screen.queryByRole('button', { name: 'Retry setup' })).toBeNull();
	await fireEvent.click(screen.getByRole('button', { name: 'Check status again' }));
	await screen.findByText(
		'Dedicated OCR is ready to process receipts. The first request loads the model on the GPU.'
	);
	expect(fetchStatus).toHaveBeenCalledTimes(2);
	expect(fetchStatus.mock.calls[1][1].method).toBeUndefined();
});

it('turns installed OCR off and on without downloading or reconfirming handoff', async () => {
	vi.mocked(api.getUiConfig).mockResolvedValue({ ...directVision, ocr_enabled: true });
	const fetchStatus = vi
		.fn()
		.mockResolvedValueOnce(status('active', null, 200, true))
		.mockResolvedValueOnce(status('disabled', null, 200, true))
		.mockResolvedValueOnce(status('active', null, 200, true));
	vi.stubGlobal('fetch', fetchStatus);
	render(Settings);
	const toggle = await screen.findByRole('switch', { name: 'Dedicated OCR' });
	await waitFor(() => expect(toggle.getAttribute('aria-checked')).toBe('true'));
	await screen.findByText(/Reported OCR output limit/);
	vi.mocked(api.getUiConfig).mockResolvedValue(directVision);
	await fireEvent.click(toggle);
	await screen.findByText(
		'Dedicated OCR is off. Installed models are retained; turn it on to resume OCR.'
	);
	expect(fetchStatus.mock.calls[1][1].body).toBe('{"enabled":false,"confirm_handoff":false}');
	expect(screen.queryByRole('checkbox', { name: /I confirm the LLM endpoint/ })).toBeNull();
	await waitFor(() => expect(screen.queryByText(/Reported OCR output limit/)).toBeNull());
	vi.mocked(api.getUiConfig).mockResolvedValue({ ...directVision, ocr_enabled: true });
	await fireEvent.click(toggle);
	await screen.findByText(
		'Dedicated OCR is ready to process receipts. The first request loads the model on the GPU.'
	);
	expect(fetchStatus.mock.calls[2][1].body).toBe('{"enabled":true,"confirm_handoff":false}');
	await screen.findByText(/Reported OCR output limit/);
});

it('keeps the confirmed switch on when pausing fails and permits retry', async () => {
	const fetchStatus = vi
		.fn()
		.mockResolvedValueOnce(status('active', null, 200, true))
		.mockRejectedValueOnce(new Error('offline'))
		.mockResolvedValueOnce(status('active', null, 200, true))
		.mockResolvedValueOnce(status('disabled', null, 200, true));
	vi.stubGlobal('fetch', fetchStatus);
	render(Settings);
	const toggle = await screen.findByRole('switch', { name: 'Dedicated OCR' });
	await screen.findByText(
		'Dedicated OCR is ready to process receipts. The first request loads the model on the GPU.'
	);
	await fireEvent.click(toggle);
	await screen.findByText('Could not confirm the Dedicated OCR change.');
	expect(toggle.getAttribute('aria-checked')).toBe('true');
	await fireEvent.click(toggle);
	await screen.findByText(
		'Dedicated OCR is off. Installed models are retained; turn it on to resume OCR.'
	);
});

it('lets an admin turn off selected OCR when runtime health requires action', async () => {
	const fetchStatus = vi
		.fn()
		.mockResolvedValueOnce(
			Response.json({
				state: 'action_required',
				reason: 'ocr_runtime_unavailable',
				installed: true,
				selected: true
			})
		)
		.mockResolvedValueOnce(Response.json({ state: 'disabled', installed: true, selected: false }));
	vi.stubGlobal('fetch', fetchStatus);
	render(Settings);
	const toggle = await screen.findByRole('switch', { name: 'Dedicated OCR' });
	await screen.findByText(
		'Dedicated OCR is installed. Restore its service before processing receipts.'
	);
	expect(toggle.getAttribute('aria-checked')).toBe('true');
	expect(toggle).toHaveProperty('disabled', false);
	await fireEvent.click(toggle);
	await screen.findByText(
		'Dedicated OCR is off. Installed models are retained; turn it on to resume OCR.'
	);
	expect(fetchStatus.mock.calls[1][1].body).toBe('{"enabled":false,"confirm_handoff":false}');
});

it.each([
	[
		'llm_request_recovery_required',
		'Confirm the previous LLM request has ended and follow the Dedicated OCR recovery guide.'
	],
	['llm_quiescence_required', 'Wait for the LLM endpoint to become idle before retrying setup.']
])(
	'explains a blocked OFF request from %s while refreshed status stays selected',
	async (reason, message) => {
		const active = () => Response.json({ state: 'active', installed: true, selected: true });
		const fetchStatus = vi
			.fn()
			.mockResolvedValueOnce(active())
			.mockResolvedValueOnce(Response.json({ detail: reason }, { status: 409 }))
			.mockResolvedValueOnce(active());
		vi.stubGlobal('fetch', fetchStatus);
		render(Settings);
		const toggle = await screen.findByRole('switch', { name: 'Dedicated OCR' });
		await screen.findByText(
			'Dedicated OCR is ready to process receipts. The first request loads the model on the GPU.'
		);
		await fireEvent.click(toggle);
		const alert = await screen.findByRole('alert');
		expect(alert.textContent).toBe(message);
		expect(toggle.getAttribute('aria-checked')).toBe('true');
		expect(screen.queryByText('Could not confirm the Dedicated OCR change.')).toBeNull();
	}
);

it('can return to direct vision after a clean initial failure without starting another installation', async () => {
	const fetchStatus = vi
		.fn()
		.mockResolvedValueOnce(status('failed', 'native_qualification_failed'))
		.mockResolvedValue(status('disabled', null, 202));
	vi.stubGlobal('fetch', fetchStatus);
	render(Settings);
	await fireEvent.click(await screen.findByRole('button', { name: 'Resume direct vision' }));
	await waitFor(() => expect(fetchStatus).toHaveBeenCalledTimes(2));
	expect(JSON.parse(fetchStatus.mock.calls[1][1].body)).toEqual({
		confirm_handoff: false,
		enabled: false
	});
});

it('shows the source-build failure and allows an explicit retry', async () => {
	const fetchStatus = vi
		.fn()
		.mockResolvedValueOnce(status('failed', 'native_image_build_failed', 200, false, false))
		.mockResolvedValueOnce(status('requested', null, 202, false, false));
	vi.stubGlobal('fetch', fetchStatus);
	render(Settings);
	await screen.findByText(/GPU image build failed/);
	await fireEvent.click(screen.getByRole('switch', { name: 'Dedicated OCR' }));
	expect(JSON.parse(fetchStatus.mock.calls[1][1].body)).toEqual({
		enabled: true,
		confirm_handoff: false
	});
});
