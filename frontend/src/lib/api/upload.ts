import { sha256 } from '@noble/hashes/sha256';
import { bytesToHex, utf8ToBytes } from '@noble/hashes/utils';
import type { UploadOptions } from '$lib/types';

/** Only opaque request identity and explicit options enter browser storage. */
export interface PendingUpload {
	key: string;
	identity: string;
	options: UploadOptions;
}
const storageKey = (owner: string) => `vb-upload:${owner}`;

interface PendingRecord {
	storage: string;
	upload: PendingUpload;
	photos?: string;
	savedAt: number;
}

export function requestKey(): string {
	// getRandomValues is available on local HTTP; subtle/randomUUID require HTTPS.
	return bytesToHex(crypto.getRandomValues(new Uint8Array(24)));
}

function readRecord(storage: string): PendingRecord | null {
	const raw = localStorage.getItem(storage);
	try {
		const value = JSON.parse(raw ?? 'null');
		if (!value || typeof value.key !== 'string' || !/^[a-f0-9]{64}$/.test(value.identity))
			return null;
		const options = value.options;
		if (!options || typeof options !== 'object' || Array.isArray(options)) return null;
		if (
			Object.keys(options).some(
				(key) => key !== 'bypass_review' && key !== 'model_id' && key !== 'allow_ocr_fallback'
			)
		)
			return null;
		if ('bypass_review' in options && typeof options.bypass_review !== 'boolean') return null;
		if ('allow_ocr_fallback' in options && typeof options.allow_ocr_fallback !== 'boolean')
			return null;
		if ('model_id' in options && typeof options.model_id !== 'string') return null;
		return {
			storage,
			upload: { key: value.key, identity: value.identity, options },
			photos: typeof value.photos === 'string' ? value.photos : undefined,
			savedAt: Number.isSafeInteger(value.savedAt) ? value.savedAt : 0
		};
	} catch {
		return null;
	}
}

function pendingRecords(owner: string): PendingRecord[] {
	const prefix = storageKey(owner);
	const records: PendingRecord[] = [];
	for (let index = 0; index < localStorage.length; index++) {
		const storage = localStorage.key(index);
		if (storage !== prefix && !storage?.startsWith(prefix + ':')) continue;
		const record = readRecord(storage);
		if (record) records.push(record);
	}
	return records.sort((a, b) => b.savedAt - a.savedAt);
}

export function readPendingUpload(owner: string): PendingUpload | null {
	try {
		return pendingRecords(owner)[0]?.upload ?? null;
	} catch {
		return null;
	}
}

function fileBytes(file: File): Promise<ArrayBuffer> {
	if (file.arrayBuffer) return file.arrayBuffer();
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve(reader.result as ArrayBuffer);
		reader.onerror = () => reject(new Error('Could not read selected photo'));
		reader.readAsArrayBuffer(file);
	});
}

async function hashFiles(files: File[], checkCurrent?: () => void): Promise<string[]> {
	const hashes: string[] = [];
	for (const file of files) {
		const bytes = await fileBytes(file);
		checkCurrent?.();
		hashes.push(bytesToHex(sha256(new Uint8Array(bytes))));
	}
	return hashes;
}

function uploadIdentity(owner: string, hashes: string[], options: UploadOptions): string {
	return bytesToHex(sha256(utf8ToBytes(JSON.stringify({ owner, hashes, options }))));
}

/** Find a replay for the selected photos without creating a request or changing storage. */
export async function matchPendingUpload(
	owner: string,
	files: File[],
	checkCurrent?: () => void
): Promise<PendingUpload | null> {
	const records = pendingRecords(owner);
	if (records.length === 0) return null;
	const hashes = await hashFiles(files, checkCurrent);
	// The full identity also works for legacy single-slot records without a photo digest.
	return (
		records.find(({ upload }) => uploadIdentity(owner, hashes, upload.options) === upload.identity)
			?.upload ?? null
	);
}

/** Call only after an explicit upload click, never on recovery/mount. */
export async function prepareUpload(
	owner: string,
	files: File[],
	explicitOptions?: UploadOptions,
	checkCurrent?: () => void
): Promise<PendingUpload> {
	const hashes = await hashFiles(files, checkCurrent);
	checkCurrent?.();
	const photos = bytesToHex(sha256(utf8ToBytes(JSON.stringify(hashes))));
	const identityOf = (options: UploadOptions) => uploadIdentity(owner, hashes, options);
	const records = pendingRecords(owner);

	// A legacy single-slot record lacks a separate photo hash. Its opaque identity
	// still proves the exact ordered bytes when reproduced with its saved options.
	if (explicitOptions === undefined) {
		const match = records.find(({ upload, photos: saved }) =>
			saved === undefined ? identityOf(upload.options) === upload.identity : saved === photos
		);
		if (match) return match.upload;
	}
	const options: UploadOptions = {};
	if (explicitOptions?.model_id !== undefined) options.model_id = explicitOptions.model_id;
	if (explicitOptions?.bypass_review !== undefined)
		options.bypass_review = explicitOptions.bypass_review;
	if (explicitOptions?.allow_ocr_fallback !== undefined)
		options.allow_ocr_fallback = explicitOptions.allow_ocr_fallback;
	const identity = identityOf(options);
	const matching = records.find(({ upload }) => upload.identity === identity);
	if (matching) return matching.upload;

	const next = { key: requestKey(), identity, options };
	const savedAt = Math.max(Date.now(), (records[0]?.savedAt ?? 0) + 1);
	// Independent records prevent another tab from overwriting a pending request.
	// Persist before POST so an ambiguous response can be retried after reload.
	localStorage.setItem(
		`${storageKey(owner)}:${next.key}`,
		JSON.stringify({ ...next, photos, savedAt })
	);
	return next;
}

export function clearPendingUpload(owner: string, key: string): void {
	const matched = pendingRecords(owner).find(({ upload }) => upload.key === key);
	if (matched) localStorage.removeItem(matched.storage);
}
