import { session } from '$lib/session-lifetime';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { api } from './client';
import { prepareUpload, readPendingUpload, clearPendingUpload } from './upload';
beforeEach(() =>
	session.set({ id: 'owner', username: 'alice', is_admin: false, can_see_all: false })
);

afterEach(() => {
	localStorage.clear();
	vi.unstubAllGlobals();
});
const files = () => [new File(['private one'], 'same.png'), new File(['private two'], 'same.png')];
it('reuses key and first explicit options after network and truncated-success ambiguity, including reload', async () => {
	const input = files();
	const first = await prepareUpload('owner', input, { bypass_review: false });
	vi.stubGlobal(
		'fetch',
		vi
			.fn()
			.mockRejectedValueOnce(new Error('offline'))
			.mockResolvedValueOnce({
				ok: true,
				status: 202,
				json: async () => {
					throw new SyntaxError('truncated');
				}
			})
	);
	await expect(api.uploadImages(input, first.options, first.key)).rejects.toBeTruthy();
	const retry = await prepareUpload('owner', input);
	expect(retry).toEqual(first);
	await expect(api.uploadImages(input, retry.options, retry.key)).rejects.toMatchObject({
		status: 202
	});
	expect(readPendingUpload('owner')).toEqual(first);
	expect((await prepareUpload('owner', files())).key).toBe(first.key);
	expect(fetch).toHaveBeenCalledTimes(2);
	expect(Object.values(localStorage).join()).not.toMatch(/private|same.png|blob:/);
});
it('changes key for changed content, order, explicit options/presence and owner', async () => {
	const input = files();
	const first = await prepareUpload('a', input, {});
	const reordered = await prepareUpload('a', [...input].reverse(), {});
	expect(reordered.key).not.toBe(first.key);
	const explicit = await prepareUpload('a', [...input].reverse(), { bypass_review: false });
	expect(explicit.key).not.toBe(reordered.key);
	expect((await prepareUpload('b', [...input].reverse(), { bypass_review: false })).key).not.toBe(
		explicit.key
	);
	const changed = await prepareUpload('a', [new File(['different'], 'same.png'), input[0]], {
		bypass_review: false
	});
	expect(changed.key).not.toBe(explicit.key);
	clearPendingUpload('a', changed.key);
	expect(readPendingUpload('a')).toEqual(explicit);
});
it('hashes actual bytes and generates keys on local HTTP without subtle or randomUUID', async () => {
	vi.stubGlobal('crypto', {
		getRandomValues: (bytes: Uint8Array) => {
			for (let i = 0; i < bytes.length; i++) bytes[i] = i;
			return bytes;
		}
	});
	const one = await prepareUpload('a', files(), {});
	expect(one.key.length).toBeGreaterThanOrEqual(32);
	expect(one.identity).toMatch(/^[0-9a-f]{64}$/);
	expect((await prepareUpload('a', files(), {})).key).toBe(one.key);
});
it('binds fallback consent presence and keeps every untouched recovery option', async () => {
	const input = files();
	const absent = await prepareUpload('owner', input, {});
	const denied = await prepareUpload('owner', input, { allow_ocr_fallback: false });
	expect(denied.key).not.toBe(absent.key);
	const allowed = await prepareUpload('owner', input, {
		model_id: 'chosen',
		allow_ocr_fallback: true
	});
	expect(allowed.key).not.toBe(denied.key);
	expect(readPendingUpload('owner')).toEqual(allowed);
	expect(await prepareUpload('owner', input)).toEqual(allowed);
	const edited = await prepareUpload('owner', input, { ...allowed.options, bypass_review: true });
	expect(edited.options).toEqual({
		model_id: 'chosen',
		allow_ocr_fallback: true,
		bypass_review: true
	});
	vi.stubGlobal(
		'fetch',
		vi.fn().mockResolvedValue({ ok: true, json: async () => ({ id: 'job' }) })
	);
	await api.uploadImages(input, edited.options, edited.key);
	expect(String(vi.mocked(fetch).mock.calls[0][0])).toContain('allow_ocr_fallback=true');
});

it('isolates different photo submissions in the same account without losing either replay key', async () => {
	const filesA = [new File(['first receipt'], 'receipt.png')];
	const filesB = [new File(['different receipt'], 'receipt.png')];
	const b = await prepareUpload('owner', filesB, { bypass_review: true });
	const a = await prepareUpload('owner', filesA);
	expect(a.options).toEqual({});
	expect(a.key).not.toBe(b.key);
	expect(await prepareUpload('owner', filesB)).toEqual(b);
	expect(await prepareUpload('owner', filesA)).toEqual(a);
	clearPendingUpload('owner', a.key);
	expect(await prepareUpload('owner', filesB)).toEqual(b);
});

it('recovers an old single-slot upload only when its ordered photos match', async () => {
	const original = files();
	const legacy = await prepareUpload('owner', original, { bypass_review: true });
	clearPendingUpload('owner', legacy.key);
	localStorage.setItem('vb-upload:owner', JSON.stringify(legacy));
	const newUpload = await prepareUpload('owner', [...original].reverse());
	expect(newUpload.options).toEqual({});
	expect(await prepareUpload('owner', original)).toEqual(legacy);
	expect(await prepareUpload('owner', [...original].reverse())).toEqual(newUpload);
});

it('does not store a pending upload if the actor changes after hashing', async () => {
	let current = true;
	const checkCurrent = () => {
		if (!current) throw new DOMException('Session changed', 'AbortError');
		queueMicrotask(() => (current = false));
	};
	const original = [new File(['same bytes'], 'photo.png')];
	await expect(prepareUpload('owner', original, {}, checkCurrent)).rejects.toMatchObject({
		name: 'AbortError'
	});
	expect(readPendingUpload('owner')).toBeNull();
});
