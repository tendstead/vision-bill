import { afterEach, beforeEach, expect, it, vi } from 'vitest';
const { readFileSync } = await vi.importActual<{
	readFileSync(path: string, encoding: 'utf8'): string;
}>('node:fs');
const html = readFileSync('../image-sticher/image-stitcher.html', 'utf8');
const readers: FakeReader[] = [],
	images: FakeImage[] = [];
class FakeReader {
	onload?: (event: { target: { result: string } }) => void;
	onerror?: () => void;
	onabort?: () => void;
	constructor() {
		readers.push(this);
	}
	readAsDataURL() {}
}
class FakeImage {
	onload?: () => void;
	onerror?: () => void;
	width = 10;
	height = 20;
	src = '';
	constructor() {
		images.push(this);
	}
}
let tool: {
	handleFileSelect(event: { target: { files: File[] } }): void;
	stitchImages(): void;
	downloadStitchedImage(): void;
};
let draw: ReturnType<typeof vi.fn>, exportCanvas: ReturnType<typeof vi.spyOn>;
const choose = (...names: string[]) =>
	tool.handleFileSelect({ target: { files: names.map((name) => new File(['synthetic'], name)) } });
const read = (index: number) => readers[index].onload?.({ target: { result: `image-${index}` } });
const decode = (index: number) => images[index].onload?.();
const stitch = () => document.getElementById('stitchButton') as HTMLButtonElement;
const download = () => document.getElementById('downloadButton') as HTMLButtonElement;
beforeEach(() => {
	readers.length = images.length = 0;
	document.body.innerHTML = html.replace(/<script>[\s\S]*?<\/script>/, '');
	draw = vi.fn();
	vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
		drawImage: draw,
		clearRect: vi.fn()
	} as never);
	exportCanvas = vi
		.spyOn(HTMLCanvasElement.prototype, 'toDataURL')
		.mockReturnValue('data:image/png;base64,');
	vi.stubGlobal('FileReader', FakeReader);
	vi.stubGlobal('Image', FakeImage);
	tool = new Function(
		html.match(/<script>([\s\S]*?)<\/script>/)![1] +
			';return {handleFileSelect, stitchImages, downloadStitchedImage};'
	)();
});
afterEach(() => {
	document.body.innerHTML = '';
	vi.restoreAllMocks();
	vi.unstubAllGlobals();
});
it('owns both reader and decoder callbacks by selection and preserves indexed order', () => {
	choose('A-slow-read', 'A-slow-decode');
	read(1);
	choose('B-first', 'B-second');
	read(3);
	read(2);
	decode(2);
	decode(1);
	read(0);
	decode(0); // abandoned callbacks must not alter B or start a new decoder
	expect(images).toHaveLength(3);
	expect([...document.querySelectorAll('.imageItem')].map((node) => node.textContent)).toEqual([
		'B-first',
		'B-second'
	]);
	expect(stitch().disabled).toBe(false);
	tool.stitchImages();
	expect(draw.mock.calls.map((call) => call[0].src)).toEqual(['image-2', 'image-3']);
});
it('blocks stitch/download after reselection until every image is ready', () => {
	choose('first');
	read(0);
	decode(0);
	tool.stitchImages();
	expect(download().style.display).toBe('inline-block');
	draw.mockClear();
	choose('next-1', 'next-2');
	expect(stitch().disabled).toBe(true);
	expect(download().style.display).toBe('none');
	read(1);
	decode(1);
	tool.stitchImages();
	tool.downloadStitchedImage();
	expect(draw).not.toHaveBeenCalled();
	expect(exportCanvas).not.toHaveBeenCalled();
	read(2);
	decode(2);
	expect(stitch().disabled).toBe(false);
});
it.each(['read', 'decode'])(
	'reports actual %s failure and never exports a partial batch',
	(stage) => {
		choose('good', 'bad');
		read(0);
		decode(0);
		if (stage === 'read') readers[1].onerror?.();
		else {
			read(1);
			images[1].onerror?.();
		}
		expect(document.getElementById('status')?.textContent).toMatch(/Could not (read|decode).*bad/);
		expect(stitch().disabled).toBe(true);
		tool.stitchImages();
		expect(draw).not.toHaveBeenCalled();
	}
);
