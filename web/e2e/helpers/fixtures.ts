/**
 * Committed ground-truth fixtures for the e2e LLM stub.
 *
 * Image bytes AND expected receipt JSON are fetched from the stub itself
 * (GET /__fixtures/<name>[.json]) so tests can never drift from what the stub
 * actually returns. No Node APIs: the frontend workspace has no @types/node.
 */

const STUB_BASE = 'http://localhost:9123';

export type FixtureName = 'a' | 'b' | 'c';

export interface ExpectedLine {
	description: string;
	quantity: number;
	unit_price: string;
	total_price: string;
	tags?: string[];
}

export interface ExpectedTax {
	name: string;
	rate: number | null;
	amount: string;
}

export interface ExpectedReceipt {
	confidence: number;
	merchant_name: string;
	merchant_address?: string;
	receipt_number?: string;
	date: string;
	time?: string;
	currency: string;
	category: string;
	line_items: ExpectedLine[];
	taxes: ExpectedTax[];
	subtotal: string;
	tax_total: string;
	tip?: string;
	total: string;
	payment_method: string;
}

export interface Fixture {
	name: FixtureName;
	mimeType: string;
	/** Upload-ready bytes: a Uint8Array whose toString() returns base64 (Playwright requirement). */
	bytes: Uint8Array;
	expected: ExpectedReceipt;
}

function toBase64(bytes: Uint8Array): string {
	let bin = '';
	for (let i = 0; i < bytes.length; i += 1) {
		bin += String.fromCharCode(bytes[i]);
	}
	return btoa(bin);
}

export async function loadFixture(name: FixtureName): Promise<Fixture> {
	const image = await fetch(`${STUB_BASE}/__fixtures/${name}`);
	if (!image.ok) throw new Error(`stub fixture ${name} unavailable: ${image.status}`);
	const raw = new Uint8Array(await image.arrayBuffer());
	const b64 = toBase64(raw);
	const data = await fetch(`${STUB_BASE}/__fixtures/${name}.json`);
	if (!data.ok) throw new Error(`stub fixture ${name}.json unavailable: ${data.status}`);
	const expected = (await data.json()) as ExpectedReceipt;
	// Playwright serializes `buffer` via toString('base64'); a plain
	// Uint8Array ignores the argument — pin the pre-computed base64 (same
	// pattern as src/lib/e2e/uploadPng.ts).
	const bytes = Object.assign(raw, { toString: () => b64 });
	return { name, mimeType: 'image/png', bytes, expected };
}
