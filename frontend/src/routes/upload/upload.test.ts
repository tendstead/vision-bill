import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, expect, it, vi } from 'vitest';
import { api } from '$lib/api/client';
import { queryClient } from '$lib/query/client';
import { session } from '$lib/auth';
import Upload from './+page.svelte';
const { goto } = vi.hoisted(() => ({ goto: vi.fn() }));
vi.mock('$app/navigation', () => ({ goto }));
vi.mock('$app/paths', () => ({ resolve: (path: string) => path }));
vi.mock('$app/state', () => ({ page: { url: new URL('http://localhost/upload') } }));
afterEach(() => {
	cleanup();
	queryClient.clear();
	localStorage.clear();
	vi.restoreAllMocks();
	goto.mockReset();
});
it.each([
	[1, 'pending'],
	[2, 'pending'],
	[1, 'analyzed'],
	[2, 'analyzed']
])('navigates every accepted %s-photo %s upload to its durable job', async (count, state) => {
	session.set({ id: 'owner', username: 'test', is_admin: false, can_see_all: false });
	let preview = 0;
	vi.spyOn(URL, 'createObjectURL').mockImplementation(() => `blob:preview-${preview++}`);
	vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
	vi.spyOn(api, 'getUiConfig').mockResolvedValue({
		bypass_review_default: true,
		registration_open: true
	});
	vi.spyOn(api, 'uploadImages').mockResolvedValue({
		id: 'durable-job',
		terminal: state === 'analyzed',
		status: state,
		receipt_id: state === 'analyzed' ? 'receipt' : null
	} as never);
	const { container } = render(Upload);
	const input = container.querySelector('input[type=file]')!;
	await fireEvent.change(input, {
		target: {
			files: Array.from({ length: Number(count) }, (_, i) => new File([String(i)], `${i}.png`))
		}
	});
	await fireEvent.click(screen.getByRole('button', { name: 'Upload' }));
	await waitFor(() => expect(goto).toHaveBeenCalledWith('/upload?job=durable-job'));
	expect(api.uploadImages).toHaveBeenCalledWith(
		expect.any(Array),
		{},
		expect.any(String),
		expect.objectContaining({ userId: 'owner' })
	);
});

it('reload recovery never posts automatically or adopts changed defaults as explicit input', async () => {
	const { prepareUpload, readPendingUpload } = await import('$lib/api/upload');
	session.set({ id: 'owner', username: 'test', is_admin: false, can_see_all: false });
	const file = new File(['same bytes'], 'photo.png');
	const pending = await prepareUpload('owner', [file], { bypass_review: false });
	vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:recover');
	vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
	vi.spyOn(api, 'getUiConfig').mockResolvedValue({
		bypass_review_default: true,
		registration_open: true
	});
	const upload = vi.spyOn(api, 'uploadImages').mockRejectedValue(new Error('lost response'));
	const { container, unmount } = render(Upload);
	await screen.findByText(/unknown outcome/);
	expect(upload).not.toHaveBeenCalled();
	// Until the photo bytes are identified, this form shows the current default.
	await waitFor(() =>
		expect((screen.getByRole('checkbox') as HTMLInputElement).checked).toBe(true)
	);
	await fireEvent.change(container.querySelector('input[type=file]')!, {
		target: { files: [file] }
	});
	await waitFor(() =>
		expect((screen.getByRole('checkbox') as HTMLInputElement).checked).toBe(false)
	);
	await fireEvent.click(screen.getByRole('button', { name: 'Upload' }));
	await waitFor(() =>
		expect(upload).toHaveBeenCalledWith(
			[file],
			pending.options,
			pending.key,
			expect.objectContaining({ userId: 'owner' })
		)
	);
	await screen.findByRole('alert');
	expect(readPendingUpload('owner')).toEqual(pending);
	unmount();
	render(Upload);
	await screen.findByText(/unknown outcome/);
	expect(upload).toHaveBeenCalledTimes(1);
});

it('does not submit under a different authenticated owner if the session changes while hashing', async () => {
	session.set({ id: 'owner', username: 'test', is_admin: false, can_see_all: false });
	let finish!: (bytes: ArrayBuffer) => void;
	const file = new File(['photo'], 'photo.png');
	Object.defineProperty(file, 'arrayBuffer', {
		value: () =>
			new Promise<ArrayBuffer>((resolve) => {
				finish = resolve;
			})
	});
	vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:owner');
	vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
	vi.spyOn(api, 'getUiConfig').mockResolvedValue({
		bypass_review_default: false,
		registration_open: true
	});
	const upload = vi.spyOn(api, 'uploadImages').mockResolvedValue({ id: 'job' } as never);
	const { container } = render(Upload);
	await fireEvent.change(container.querySelector('input[type=file]')!, {
		target: { files: [file] }
	});
	await fireEvent.click(screen.getByRole('button', { name: 'Upload' }));
	session.set({ id: 'other', username: 'other', is_admin: false, can_see_all: false });
	finish(new Uint8Array([1, 2]).buffer);
	await new Promise((resolve) => setTimeout(resolve, 0));
	expect(upload).not.toHaveBeenCalled();
	expect(localStorage.getItem('vb-upload:owner')).toBeNull();
});

it('preserves hidden legacy options when the user edits the matched review choice', async () => {
	const { prepareUpload } = await import('$lib/api/upload');
	session.set({ id: 'owner', username: 'test', is_admin: false, can_see_all: false });
	const file = new File(['same bytes'], 'photo.png');
	await prepareUpload('owner', [file], {
		model_id: 'saved-model',
		bypass_review: false,
		allow_ocr_fallback: true
	});
	vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:fallback');
	vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
	vi.spyOn(api, 'getUiConfig').mockResolvedValue({
		bypass_review_default: true,
		registration_open: true,
		ocr_enabled: true,
		ocr_required: false,
		ocr_allow_vision_fallback: true
	});
	const upload = vi.spyOn(api, 'uploadImages').mockRejectedValue(new Error('lost response'));
	const { container } = render(Upload);
	await screen.findByText(/unknown outcome/);
	const checkbox = screen.getByRole<HTMLInputElement>('checkbox');
	await waitFor(() => expect(checkbox.checked).toBe(true));
	await fireEvent.change(container.querySelector('input[type=file]')!, {
		target: { files: [file] }
	});
	await waitFor(() => expect(checkbox.checked).toBe(false));
	await fireEvent.click(checkbox);
	await fireEvent.click(screen.getByRole('button', { name: 'Upload' }));
	await waitFor(() =>
		expect(upload).toHaveBeenCalledWith(
			[file],
			{ model_id: 'saved-model', bypass_review: true, allow_ocr_fallback: true },
			expect.any(String),
			expect.objectContaining({ userId: 'owner' })
		)
	);
	expect(screen.queryByRole('checkbox', { name: 'Allow vision analysis if OCR fails' })).toBeNull();
});

it('does not adopt a later pending preference that was never shown for this selection', async () => {
	const { prepareUpload } = await import('$lib/api/upload');
	session.set({ id: 'owner', username: 'test', is_admin: false, can_see_all: false });
	const file = new File(['current bytes'], 'photo.png');
	await prepareUpload('owner', [new File(['different bytes'], 'photo.png')], {
		bypass_review: true
	});
	vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:current');
	vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
	vi.spyOn(api, 'getUiConfig').mockResolvedValue({
		bypass_review_default: false,
		registration_open: true
	});
	const upload = vi.spyOn(api, 'uploadImages').mockRejectedValue(new Error('lost response'));
	const { container } = render(Upload);
	await screen.findByText(/unknown outcome/);
	await fireEvent.change(container.querySelector('input[type=file]')!, {
		target: { files: [file] }
	});
	const button = screen.getByRole<HTMLButtonElement>('button', { name: 'Upload' });
	await waitFor(() => expect(button.disabled).toBe(false));
	expect(screen.getByRole<HTMLInputElement>('checkbox').checked).toBe(false);

	// A second tab records a pending upload of these bytes after this form is ready.
	const later = await prepareUpload('owner', [file], { bypass_review: true });
	await fireEvent.click(button);
	await waitFor(() => expect(upload).toHaveBeenCalledTimes(1));
	expect(upload.mock.calls[0][1]).toEqual({});
	expect(upload.mock.calls[0][2]).not.toBe(later.key);
});

it('resets the recovered review choice when the selected photos are reordered', async () => {
	const { prepareUpload } = await import('$lib/api/upload');
	session.set({ id: 'owner', username: 'test', is_admin: false, can_see_all: false });
	const top = new File(['top bytes'], 'top.png');
	const bottom = new File(['bottom bytes'], 'bottom.png');
	const pending = await prepareUpload('owner', [top, bottom], { bypass_review: true });
	let preview = 0;
	vi.spyOn(URL, 'createObjectURL').mockImplementation(() => `blob:reorder-${preview++}`);
	vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
	vi.spyOn(api, 'getUiConfig').mockResolvedValue({
		bypass_review_default: false,
		registration_open: true
	});
	const upload = vi.spyOn(api, 'uploadImages').mockRejectedValue(new Error('lost response'));
	const { container } = render(Upload);
	await fireEvent.change(container.querySelector('input[type=file]')!, {
		target: { files: [top, bottom] }
	});
	const checkbox = screen.getByRole<HTMLInputElement>('checkbox');
	await waitFor(() => expect(checkbox.checked).toBe(true));
	const moveDown = screen.getAllByRole<HTMLButtonElement>('button', { name: 'Move photo down' })[0];
	await fireEvent.click(moveDown);
	const button = screen.getByRole<HTMLButtonElement>('button', { name: 'Upload' });
	await waitFor(() => expect(button.disabled).toBe(false));
	expect(checkbox.checked).toBe(false);
	await fireEvent.click(button);
	await waitFor(() => expect(upload).toHaveBeenCalledTimes(1));
	expect(upload.mock.calls[0].slice(0, 2)).toEqual([[bottom, top], {}]);
	expect(upload.mock.calls[0][2]).not.toBe(pending.key);
});

it('blocks upload after an unreadable selection until recovery checks the bytes again', async () => {
	const { prepareUpload, readPendingUpload } = await import('$lib/api/upload');
	session.set({ id: 'owner', username: 'test', is_admin: false, can_see_all: false });
	const file = new File(['same bytes'], 'photo.png');
	const pending = await prepareUpload('owner', [file], { bypass_review: true });
	const bytes = new TextEncoder().encode('same bytes').buffer;
	Object.defineProperty(file, 'arrayBuffer', {
		value: vi
			.fn()
			.mockRejectedValueOnce(new DOMException('Temporarily unreadable', 'NotReadableError'))
			.mockResolvedValue(bytes)
	});
	vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:retry-read');
	vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
	vi.spyOn(api, 'getUiConfig').mockResolvedValue({
		bypass_review_default: false,
		registration_open: true
	});
	const upload = vi.spyOn(api, 'uploadImages').mockResolvedValue({ id: 'job' } as never);
	const { container } = render(Upload);
	await fireEvent.change(container.querySelector('input[type=file]')!, {
		target: { files: [file] }
	});
	expect((await screen.findByRole('alert')).textContent).toContain(
		'Could not check the selected photos'
	);
	const button = screen.getByRole<HTMLButtonElement>('button', { name: 'Upload' });
	expect(button.disabled).toBe(true);
	await fireEvent.click(button);
	expect(upload).not.toHaveBeenCalled();
	expect(readPendingUpload('owner')).toEqual(pending);

	await fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
	await waitFor(() => {
		expect(screen.getByRole<HTMLInputElement>('checkbox').checked).toBe(true);
		expect(button.disabled).toBe(false);
	});
	await fireEvent.click(button);
	await waitFor(() =>
		expect(upload).toHaveBeenCalledExactlyOnceWith(
			[file],
			pending.options,
			pending.key,
			expect.objectContaining({ userId: 'owner' })
		)
	);
});
