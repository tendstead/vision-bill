import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/svelte';
import { afterEach, expect, it, vi } from 'vitest';
import { session } from '$lib/auth';
import { api, ApiError } from '$lib/api/client';
import { queryClient } from '$lib/query/client';
import Page from './+page.svelte';

vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));

afterEach(() => {
	cleanup();
	session.set(null);
	queryClient.clear();
	vi.restoreAllMocks();
});

it('offers a receipt review action and describes the displayed queue scope', async () => {
	vi.spyOn(api, 'listImages').mockResolvedValue([
		{
			id: 'job',
			original_filename: 'receipt.png',
			status: 'needs_review',
			receipt_id: 'receipt',
			attempt_count: 1,
			max_attempts: 3
		}
	] as never);
	render(Page);
	const action = await screen.findByRole('link', { name: 'Open and review receipt' });
	expect(action.getAttribute('href')).toBe('/receipts/receipt');
	expect(screen.getByText('1 jobs shown (up to 100)')).toBeTruthy();
});

it('identifies the previous result while another attempt is processing', async () => {
	vi.spyOn(api, 'listImages').mockResolvedValue([
		{
			id: 'job',
			status: 'processing',
			result_code: 'analysis_incomplete',
			attempt_count: 2,
			max_attempts: 3
		}
	] as never);
	render(Page);
	await screen.findByText(/Previous attempt: The extraction did not produce a complete receipt/);
});

it('explains how to recover from a confirmed formatter output limit', async () => {
	vi.spyOn(api, 'listImages').mockResolvedValue([
		{
			id: 'job',
			status: 'failed',
			result_code: 'formatter_output_limit',
			attempt_count: 1,
			max_attempts: 3
		}
	] as never);
	render(Page);
	await screen.findByText(
		/increase the output token allowance, then retry using the current server settings/
	);
	expect(screen.getByRole('link', { name: 'Open job' }).getAttribute('href')).toBe(
		'/upload?job=job'
	);
});

it.each([
	['vision_model_unavailable', false],
	['A distinct provider detail', true]
])('suppresses repeated result codes but retains %s', async (error, visible) => {
	vi.spyOn(api, 'listImages').mockResolvedValue([
		{
			id: 'job',
			status: 'failed',
			result_code: 'vision_model_unavailable',
			error,
			attempt_count: 1,
			max_attempts: 3
		}
	] as never);
	render(Page);
	await screen.findByRole('link', { name: 'Open job' });
	expect(screen.queryByText(error) !== null).toBe(visible);
	expect(
		screen.getByText('No suitable selected vision model is available to read the photos.')
	).toBeTruthy();
});

it('opens an available draft directly and disables removal for linked receipts', async () => {
	vi.spyOn(api, 'listImages').mockResolvedValue([
		{ id: 'draft', status: 'needs_review', draft_available: true, receipt_id: null },
		{ id: 'linked', status: 'needs_review', draft_available: true, receipt_id: 'receipt' }
	] as never);
	render(Page);
	expect(
		(await screen.findByRole('link', { name: 'Open receipt draft' })).getAttribute('href')
	).toBe('/upload?job=draft&draft=1');
	const linked = screen.getByRole('link', { name: 'Open and review receipt' }).closest('li')!;
	expect(within(linked).getByRole('button', { name: 'Remove' }).hasAttribute('disabled')).toBe(
		true
	);
});

it('closes confirmation and shows an actionable conflict while refreshing the queue', async () => {
	session.set({ id: 'alice', username: 'alice', is_admin: false, can_see_all: false });
	const list = vi
		.spyOn(api, 'listImages')
		.mockResolvedValue([
			{ id: 'draft', status: 'needs_review', draft_available: true, receipt_id: null }
		] as never);
	vi.spyOn(api, 'deleteImage').mockRejectedValue(new ApiError(409, 'claim raced'));
	render(Page);
	await screen.findByRole('link', { name: 'Open receipt draft' });
	await fireEvent.click(screen.getByRole('button', { name: 'Remove' }));
	await fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Delete' }));
	await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
	expect(screen.getByRole('alert').textContent).toContain('queue has been refreshed');
	await waitFor(() => expect(list.mock.calls.length).toBeGreaterThan(1));
});

it('blocks processing removal and repeated actions during deletion', async () => {
	session.set({ id: 'alice', username: 'alice', is_admin: false, can_see_all: false });
	vi.spyOn(api, 'listImages').mockResolvedValue([
		{ id: 'processing', original_filename: 'Busy file', status: 'processing', receipt_id: null },
		{
			id: 'draft',
			original_filename: 'Draft file',
			status: 'needs_review',
			draft_available: true,
			receipt_id: null
		}
	] as never);
	let finish!: () => void;
	const remove = vi.spyOn(api, 'deleteImage').mockReturnValue(
		new Promise((resolve) => {
			finish = () => resolve({ deleted: 'draft' } as never);
		})
	);
	render(Page);
	const processing = (await screen.findByRole('link', { name: 'Busy file' })).closest('li')!;
	expect(within(processing).getByRole('button', { name: 'Remove' }).hasAttribute('disabled')).toBe(
		true
	);
	const draft = screen.getByRole('link', { name: 'Draft file' }).closest('li')!;
	await fireEvent.click(within(draft).getByRole('button', { name: 'Remove' }));
	const confirm = within(screen.getByRole('dialog')).getByRole('button', { name: 'Delete' });
	await fireEvent.click(confirm);
	expect(confirm.hasAttribute('disabled')).toBe(true);
	expect(within(draft).getByRole('button', { name: 'Remove' }).hasAttribute('disabled')).toBe(true);
	expect(remove).toHaveBeenCalledTimes(1);
	finish();
	await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
});
