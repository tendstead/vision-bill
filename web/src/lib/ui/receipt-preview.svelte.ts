// One shared hover preview (a small receipt thumbnail) for every receipt id on
// the benchmark page: tables, chart labels and chart markers.
export const preview = $state<{ imageId: string | null; left: number; top: number }>({
	imageId: null,
	left: 0,
	top: 0
});

const WIDTH = 168;
const HEIGHT = 210;

export function showPreview(imageId: string | null, anchor: Element) {
	if (!imageId) return;
	const r = anchor.getBoundingClientRect();
	let left = r.right + 10;
	if (left + WIDTH > window.innerWidth - 8) left = Math.max(8, r.left - WIDTH - 10);
	const top = Math.min(Math.max(8, r.top - 12), Math.max(8, window.innerHeight - HEIGHT - 8));
	preview.imageId = imageId;
	preview.left = left;
	preview.top = top;
}

export function hidePreview() {
	preview.imageId = null;
}

/** Event handlers that show the preview while the element is hovered or focused. */
export function previewProps(imageId: string | null | undefined) {
	return {
		onpointerenter: (e: Event) => showPreview(imageId ?? null, e.currentTarget as Element),
		onpointerleave: hidePreview,
		onfocus: (e: Event) => showPreview(imageId ?? null, e.currentTarget as Element),
		onblur: hidePreview
	};
}
