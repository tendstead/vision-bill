import { session } from '$lib/session-lifetime';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { api, ApiError } from '$lib/api/client';
import { setLocale } from '$lib/i18n';
import { queryClient } from '$lib/query/client';
import { queryKeys } from '$lib/query/keys';
import SubmissionStatus from './SubmissionStatus.svelte';

vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
const status = (overrides = {}) => ({
	id: 'job',
	status: 'failed',
	terminal: false,
	generation: 1,
	attempt_count: 1,
	max_attempts: 3,
	retry_at: '2030-01-02T03:04:05Z',
	wait_reason: 'retry_backoff',
	result_code: 'provider_error',
	receipt_id: null,
	draft_available: false,
	source_views: [],
	...overrides
});
beforeEach(() =>
	session.set({ id: 'viewer', username: 'alice', is_admin: false, can_see_all: false })
);

afterEach(() => {
	cleanup();
	queryClient.clear();
	vi.restoreAllMocks();
	vi.useRealTimers();
});

it.each(['pending', 'processing'])(
	'explicitly adopts current processing policy on a held %s job and retains retry identity',
	async (state) => {
		setLocale('en');
		vi.spyOn(api, 'getImage').mockResolvedValue(
			status({ status: state, wait_reason: 'execution_policy_changed', retry_at: null }) as never
		);
		const retry = vi
			.spyOn(api, 'retrySubmission')
			.mockRejectedValue(new ApiError(0, 'connection lost'));
		render(SubmissionStatus, { id: 'job' });
		const button = await screen.findByRole('button', {
			name: 'Retry using current server settings'
		});
		expect(screen.queryByRole('button', { name: 'Retry analysis' })).toBeNull();
		await fireEvent.click(button);
		await screen.findByRole('alert');
		await fireEvent.click(button);
		await waitFor(() => expect(retry).toHaveBeenCalledTimes(2));
		expect(retry.mock.calls[0]).toEqual(retry.mock.calls[1]);
		expect(retry.mock.calls[0]).toEqual([
			'job',
			{ expected_generation: 1, request_key: expect.any(String), policy: 'current' }
		]);
	}
);

describe('SubmissionStatus', () => {
	it.each([
		['en', 'The local OCR GPU ran out of video memory.'],
		['de', 'Die lokale OCR-GPU hat nicht genug Grafikspeicher.']
	] as const)('reports local OCR GPU memory exhaustion in %s', async (language, message) => {
		setLocale(language);
		try {
			vi.spyOn(api, 'getImage').mockResolvedValue(
				status({
					terminal: true,
					retry_at: null,
					wait_reason: null,
					error: 'ocr_gpu_out_of_memory',
					result_code: 'ocr_gpu_out_of_memory'
				}) as never
			);

			render(SubmissionStatus, { id: 'job' });
			await screen.findByText(new RegExp(`${message} \\(ocr_gpu_out_of_memory\\)`));
		} finally {
			setLocale('en');
		}
	});

	it.each(['en', 'de'] as const)(
		'explains model unreadability despite a valid working image in %s',
		async (language) => {
			// A decoded working image does not imply the model can read the receipt.
			// This is the terminal API shape observed in the unreadable-model browser flow.
			const persisted = status({
				status: 'unreadable',
				terminal: true,
				retry_at: null,
				wait_reason: null,
				result_code: 'unreadable',
				error: 'unreadable',
				source_views: [
					{
						view_id: 'working',
						photo_index: 0,
						coordinate_space: 'working',
						width: 320,
						height: 480,
						sha256: 'df40cf0ccfd9dc1dca83eed23414fe64f6969d3e2a8fd02ee2b8c9be4cd0b43b',
						url: '/api/v1/images/job/views/working',
						transforms: [{ operation: 'test_fixture_unwarp' }],
						inverse_mapping: null
					}
				]
			});
			const before = structuredClone(persisted);
			vi.spyOn(api, 'getImage').mockResolvedValue(persisted as never);
			setLocale(language);
			try {
				render(SubmissionStatus, { id: 'job' });
				await screen.findByText(language === 'en' ? 'Attempt 1 of 3' : 'Versuch 1 von 3');
				expect(
					screen.getByText(
						language === 'en'
							? 'Analysis result: The receipt could not be read reliably. Check the file or try clearer photos. (unreadable)'
							: 'Analyseergebnis: Der Beleg konnte nicht zuverlässig gelesen werden. Prüfe die Datei oder versuche es mit deutlicheren Fotos. (unreadable)'
					)
				).not.toBeNull();
				const previewLabel = language === 'en' ? 'Photo 1 (Working image)' : 'Foto 1 (Arbeitsbild)';
				expect(screen.getByRole('link', { name: previewLabel }).getAttribute('href')).toBe(
					'/api/v1/images/job/views/working?vb_expected_user=viewer'
				);
				expect(screen.queryByText('Photo is too blurry', { exact: true })).toBeNull();
				expect(queryClient.getQueryData(queryKeys.image('job'))).toEqual(before);
				expect(persisted).toEqual(before);
			} finally {
				setLocale('en');
			}
		}
	);
	it.each([
		['failed', 'retry_backoff'],
		['pending', 'ocr_unavailable']
	])('polls nonterminal %s / %s and stops only when terminal', async (state, reason) => {
		vi.useFakeTimers();
		vi.spyOn(api, 'getImage')
			.mockResolvedValueOnce(status({ status: state, wait_reason: reason }) as never)
			.mockResolvedValue(
				status({ terminal: true, status: 'analyzed', receipt_id: 'receipt' }) as never
			);
		render(SubmissionStatus, { id: 'job' });
		await vi.advanceTimersByTimeAsync(50);
		expect(queryClient.getQueryData(queryKeys.image('job'))).toMatchObject({ terminal: false });
		expect(screen.getByText('Attempt 1 of 3')).toBeTruthy();
		expect(screen.getByText(/2030/)).toBeTruthy();
		await vi.advanceTimersByTimeAsync(2200);
		expect(api.getImage).toHaveBeenCalledTimes(2);
		expect(screen.getByRole('link', { name: /Open and review/ }).getAttribute('href')).toBe(
			'/receipts/receipt'
		);
		await vi.advanceTimersByTimeAsync(6000);
		expect(api.getImage).toHaveBeenCalledTimes(2);
	});
	it('offers draft access and authenticated source links for needs_review', async () => {
		vi.spyOn(api, 'getImage').mockResolvedValue(
			status({
				terminal: true,
				status: 'needs_review',
				draft_available: true,
				source_views: [
					{
						view_id: 'view',
						photo_index: 0,
						coordinate_space: 'original',
						url: '/api/v1/images/job/views/view'
					}
				]
			}) as never
		);
		render(SubmissionStatus, { id: 'job' });
		expect((await screen.findByRole('link', { name: /draft/i })).getAttribute('href')).toBe(
			'/upload?job=job&draft=1'
		);
		expect(screen.getByRole('link', { name: /Photo 1/ }).getAttribute('href')).toBe(
			'/api/v1/images/job/views/view?vb_expected_user=viewer'
		);
	});
	it('keeps status and retry visible without duplicating the open draft or its photos', async () => {
		vi.spyOn(api, 'getImage').mockResolvedValue(
			status({
				terminal: true,
				status: 'needs_review',
				result_code: 'needs_review',
				error: 'needs_review',
				draft_available: true,
				source_views: [
					{
						view_id: 'view',
						photo_index: 0,
						coordinate_space: 'working',
						url: '/api/v1/images/job/views/view'
					}
				]
			}) as never
		);
		render(SubmissionStatus, { id: 'job', editingDraft: true });
		expect(await screen.findByRole('button', { name: 'Retry analysis' })).toBeTruthy();
		expect(screen.queryByRole('link', { name: /open.*draft/i })).toBeNull();
		expect(screen.queryByRole('link', { name: /Photo 1/ })).toBeNull();
		expect(screen.getByRole('status').textContent).toContain('needs review');
	});
	it('suppresses duplicate retry clicks and preserves stale-generation detail', async () => {
		vi.spyOn(api, 'getImage').mockResolvedValue(
			status({ terminal: true, retry_at: null, wait_reason: null }) as never
		);
		let reject!: (error: Error) => void;
		const retry = vi.spyOn(api, 'retrySubmission').mockImplementation(
			() =>
				new Promise((_, no) => {
					reject = no;
				})
		);
		render(SubmissionStatus, { id: 'job' });
		const button = await screen.findByRole('button', { name: /Retry analysis/ });
		await fireEvent.click(button);
		await fireEvent.click(button);
		expect(retry).toHaveBeenCalledTimes(1);
		expect(retry.mock.calls[0]).toEqual([
			'job',
			{ expected_generation: 1, request_key: expect.any(String) }
		]);
		reject(new ApiError(409, 'Retry conflict: stale generation'));
		await waitFor(() =>
			expect(screen.getByRole('alert').textContent).toContain('Retry conflict: stale generation')
		);
	});
});

it('reuses retry request identity after an ambiguous retry and adopts the next generation', async () => {
	const getImage = vi
		.spyOn(api, 'getImage')
		.mockResolvedValue(status({ terminal: true, retry_at: null, wait_reason: null }) as never);
	const retry = vi
		.spyOn(api, 'retrySubmission')
		.mockRejectedValueOnce(new ApiError(0, 'Network error'))
		.mockResolvedValue(
			status({
				status: 'pending',
				generation: 2,
				wait_reason: 'ocr_unavailable',
				retry_at: null
			}) as never
		);
	render(SubmissionStatus, { id: 'job' });
	await fireEvent.click(await screen.findByRole('button', { name: 'Retry analysis' }));
	await screen.findByRole('alert');
	getImage.mockResolvedValue(
		status({
			status: 'pending',
			generation: 2,
			wait_reason: 'ocr_unavailable',
			retry_at: null
		}) as never
	);
	await fireEvent.click(screen.getByRole('button', { name: 'Retry analysis' }));
	await waitFor(() => expect(retry).toHaveBeenCalledTimes(2));
	expect(retry.mock.calls[0]).toEqual(retry.mock.calls[1]);
	await waitFor(() =>
		expect(queryClient.getQueryData(queryKeys.image('job'))).toMatchObject({ generation: 2 })
	);
});

function deferred<T>() {
	let resolve!: (value: T) => void;
	let reject!: (error: Error) => void;
	const promise = new Promise<T>((yes, no) => {
		resolve = yes;
		reject = no;
	});
	return { promise, resolve, reject };
}

it('keeps B and its pending GET untouched when A retry completes after navigation', async () => {
	const retryA = deferred<ReturnType<typeof status>>();
	const getB = deferred<ReturnType<typeof status>>();
	vi.spyOn(api, 'getImage').mockImplementation((id) =>
		id === 'a'
			? Promise.resolve(status({ id: 'a', terminal: true }) as never)
			: (getB.promise as never)
	);
	vi.spyOn(api, 'retrySubmission').mockReturnValue(retryA.promise as never);
	const cancel = vi.spyOn(queryClient, 'cancelQueries');
	const { rerender } = render(SubmissionStatus, { id: 'a' });
	await fireEvent.click(await screen.findByRole('button', { name: 'Retry analysis' }));
	await rerender({ id: 'b' });
	await waitFor(() => expect(api.getImage).toHaveBeenCalledWith('b'));
	retryA.resolve(
		status({ id: 'a', generation: 2, status: 'analyzed', terminal: true, receipt_id: 'a-receipt' })
	);
	await retryA.promise;
	await new Promise((resolve) => setTimeout(resolve, 0));
	expect(queryClient.getQueryData(queryKeys.image('b'))).toBeUndefined();
	expect(
		cancel.mock.calls.some(
			([filters]) => JSON.stringify(filters?.queryKey) === JSON.stringify(queryKeys.image('b'))
		)
	).toBe(false);
	getB.resolve(status({ id: 'b', terminal: true, status: 'analyzed', receipt_id: 'b-receipt' }));
	expect(
		(await screen.findByRole('link', { name: 'Open and review receipt' })).getAttribute('href')
	).toBe('/receipts/b-receipt');
});

it.each([0, 409])('ignores A late %s error and leaves B retry in progress', async (code) => {
	const retryA = deferred<ReturnType<typeof status>>();
	const retryB = deferred<ReturnType<typeof status>>();
	vi.spyOn(api, 'getImage').mockImplementation((id) =>
		Promise.resolve(status({ id, terminal: true }) as never)
	);
	const retry = vi
		.spyOn(api, 'retrySubmission')
		.mockImplementation((id) => (id === 'a' ? retryA.promise : retryB.promise) as never);
	const { rerender } = render(SubmissionStatus, { id: 'a' });
	await fireEvent.click(await screen.findByRole('button', { name: 'Retry analysis' }));
	await rerender({ id: 'b' });
	const buttonB = await screen.findByRole('button', { name: 'Retry analysis' });
	await fireEvent.click(buttonB);
	const readsBefore = vi.mocked(api.getImage).mock.calls.length;
	retryA.reject(new ApiError(code, 'A late error'));
	await retryA.promise.catch(() => {});
	await new Promise((resolve) => setTimeout(resolve, 0));
	expect(screen.queryByRole('alert')).toBeNull();
	expect((buttonB as HTMLButtonElement).disabled).toBe(true);
	expect(retry).toHaveBeenCalledTimes(2);
	expect(api.getImage).toHaveBeenCalledTimes(readsBefore);
	retryB.reject(new ApiError(0, 'B current error'));
	expect((await screen.findByRole('alert')).textContent).toContain('B current error');
});

it.each([false, true])('ignores retry completion after unmount (conflict=%s)', async (conflict) => {
	const pending = deferred<ReturnType<typeof status>>();
	const initial = status({ terminal: true });
	vi.spyOn(api, 'getImage').mockResolvedValue(initial as never);
	vi.spyOn(api, 'retrySubmission').mockReturnValue(pending.promise as never);
	const { unmount } = render(SubmissionStatus, { id: 'job' });
	await fireEvent.click(await screen.findByRole('button', { name: 'Retry analysis' }));
	unmount();
	const cancel = vi.spyOn(queryClient, 'cancelQueries');
	const readsBefore = vi.mocked(api.getImage).mock.calls.length;
	if (conflict) pending.reject(new ApiError(409, 'Unmounted conflict'));
	else pending.resolve(status({ generation: 2 }));
	await pending.promise.catch(() => {});
	await new Promise((resolve) => setTimeout(resolve, 0));
	expect(cancel).not.toHaveBeenCalled();
	expect(api.getImage).toHaveBeenCalledTimes(readsBefore);
	expect(queryClient.getQueryData(queryKeys.image('job'))).toEqual(initial);
});

it.each([
	{ generation: 3, status: 'pending', terminal: false },
	{ generation: 2, status: 'analyzed', terminal: true, receipt_id: 'current-receipt' }
])('does not regress a newer cached generation/state: %j', async (newer) => {
	const pending = deferred<ReturnType<typeof status>>();
	const getImage = vi.spyOn(api, 'getImage').mockResolvedValue(status({ terminal: true }) as never);
	vi.spyOn(api, 'retrySubmission').mockReturnValue(pending.promise as never);
	render(SubmissionStatus, { id: 'job' });
	await fireEvent.click(await screen.findByRole('button', { name: 'Retry analysis' }));
	const latest = status(newer);
	getImage.mockResolvedValue(latest as never);
	queryClient.setQueryData(queryKeys.image('job'), latest);
	const observed: unknown[] = [];
	const unsubscribe = queryClient.getQueryCache().subscribe((event) => {
		if (event.query.queryKey[1] === 'job') observed.push(event.query.state.data);
	});
	pending.resolve(status({ generation: 2, status: 'pending' }));
	await pending.promise;
	await new Promise((resolve) => setTimeout(resolve, 0));
	unsubscribe();
	expect(observed.every((value) => JSON.stringify(value) === JSON.stringify(latest))).toBe(true);
	expect(queryClient.getQueryData(queryKeys.image('job'))).toEqual(latest);
});

it('rechecks request ownership after awaiting query cancellation', async () => {
	const pending = deferred<ReturnType<typeof status>>();
	const cancellation = deferred<void>();
	vi.spyOn(api, 'getImage').mockImplementation((id) =>
		Promise.resolve(status({ id, terminal: true }) as never)
	);
	vi.spyOn(api, 'retrySubmission').mockReturnValue(pending.promise as never);
	const cancel = vi.spyOn(queryClient, 'cancelQueries').mockReturnValue(cancellation.promise);
	const { rerender } = render(SubmissionStatus, { id: 'a' });
	await fireEvent.click(await screen.findByRole('button', { name: 'Retry analysis' }));
	pending.resolve(status({ id: 'a', generation: 2 }));
	await waitFor(() => expect(cancel).toHaveBeenCalledOnce());
	await rerender({ id: 'b' });
	await waitFor(() =>
		expect(queryClient.getQueryData(queryKeys.image('b'))).toMatchObject({ id: 'b' })
	);
	cancellation.resolve();
	await cancellation.promise;
	await new Promise((resolve) => setTimeout(resolve, 0));
	expect(queryClient.getQueryData(queryKeys.image('b'))).toMatchObject({ id: 'b', generation: 1 });
	expect(queryClient.getQueryData(queryKeys.image('a'))).toMatchObject({ id: 'a', generation: 1 });
	expect(
		(screen.getByRole('button', { name: 'Retry analysis' }) as HTMLButtonElement).disabled
	).toBe(false);
});

it('ignores the first A retry after returning to A and starting another request', async () => {
	const first = deferred<ReturnType<typeof status>>();
	const second = deferred<ReturnType<typeof status>>();
	vi.spyOn(api, 'getImage').mockImplementation((id) =>
		Promise.resolve(status({ id, terminal: true }) as never)
	);
	const retry = vi
		.spyOn(api, 'retrySubmission')
		.mockReturnValueOnce(first.promise as never)
		.mockReturnValueOnce(second.promise as never);
	const { rerender } = render(SubmissionStatus, { id: 'a' });
	await fireEvent.click(await screen.findByRole('button', { name: 'Retry analysis' }));
	await rerender({ id: 'b' });
	await waitFor(() => expect(api.getImage).toHaveBeenCalledWith('b'));
	await rerender({ id: 'a' });
	const current = await screen.findByRole('button', { name: 'Retry analysis' });
	await fireEvent.click(current);
	expect(retry).toHaveBeenCalledTimes(2);
	first.reject(new ApiError(409, 'Earlier A conflict'));
	await first.promise.catch(() => {});
	await new Promise((resolve) => setTimeout(resolve, 0));
	expect(screen.queryByRole('alert')).toBeNull();
	expect((current as HTMLButtonElement).disabled).toBe(true);
	second.reject(new ApiError(0, 'Current A error'));
	expect((await screen.findByRole('alert')).textContent).toContain('Current A error');
});

it.each(['en', 'de'] as const)(
	'explains readiness waits without spending attempts in %s',
	async (language) => {
		const { setLocale } = await import('$lib/i18n');
		setLocale(language);
		vi.spyOn(api, 'getImage').mockResolvedValue(
			status({
				status: 'pending',
				attempt_count: 0,
				wait_reason: 'formatter_model_unavailable',
				result_code: 'ocr_unavailable'
			}) as never
		);
		render(SubmissionStatus, { id: 'job' });
		await screen.findByText(
			language === 'en'
				? /selected text formatter model is unavailable/
				: /gewählte Textformatierungsmodell ist nicht verfügbar/
		);
		expect(
			screen.getByText(
				language === 'en'
					? /Readiness waits do not use an attempt/
					: /Warten auf Bereitschaft verbraucht keinen Versuch/
			)
		).not.toBeNull();
		expect(
			screen.getByText(
				language === 'en'
					? /Waiting for the OCR service to become ready/
					: /Es wird gewartet, bis der OCR-Dienst bereit ist/
			)
		).not.toBeNull();
		setLocale('en');
	}
);

it('shows saved observations awaiting recovery at the inference limit', async () => {
	setLocale('en');
	vi.spyOn(api, 'getImage').mockResolvedValue(
		status({
			status: 'pending',
			attempt_count: 3,
			observations_available: true,
			draft_available: true,
			wait_reason: 'observation_recovery_required',
			retry_at: null
		}) as never
	);
	render(SubmissionStatus, { id: 'job' });
	await screen.findByText('Attempt 3 of 3');
	expect(screen.getByRole('status').textContent).toContain('Observations saved');
	await screen.findByText(/finish after model recovery/i);
	expect(screen.queryByRole('link', { name: 'Open receipt draft' })).toBeNull();
	expect(screen.queryByRole('link', { name: 'Open and review receipt' })).toBeNull();
});
