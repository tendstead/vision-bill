// A tiny valid 1x1 PNG, built in-memory for the e2e upload flows.
//
// The image content is irrelevant to the LLM stub (it maps sha256(image) to a
// canned Receipt); an image not in the fixture map yields the stub's generic
// receipt, which is exactly what these UI flows need. Building it from base64
// avoids any Node filesystem/path access (the frontend has no @types/node and
// the test files are ESM).
const PNG_B64 =
	'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgAAIAAAUAAXpeqz8AAAAASUVORK5CYII=';

/** A `page.setInputFiles` file descriptor: a valid 1x1 PNG. */
export function uploadPng(): { name: string; mimeType: string; buffer: Uint8Array } {
	const bin = atob(PNG_B64);
	const bytes = new Uint8Array(bin.length);
	for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
	// Playwright serializes the buffer via `buffer.toString("base64")`. A plain
	// Uint8Array ignores the encoding argument (it comma-joins the bytes), so
	// the browser-side atob fails with invalid base64. This repo has no
	// @types/node (no Node Buffer), so give the array a toString that returns
	// the pre-computed base64 instead.
	const buffer = Object.assign(bytes, { toString: () => PNG_B64 });
	return { name: 'e2e-upload.png', mimeType: 'image/png', buffer };
}
