import { session } from '$lib/session-lifetime';
import { cleanup, render, screen, fireEvent } from '@testing-library/svelte';
import { afterEach, beforeEach, expect, it } from 'vitest';
import ExtractionSourceViewer from './ExtractionSourceViewer.svelte';
const view = {
	view_id: 'unwarped',
	photo_index: 0,
	coordinate_space: 'working' as const,
	width: 1200,
	height: 1800,
	sha256: 'hash',
	url: '/exact-working.png',
	transforms: [{ unwarp: true }],
	inverse_mapping: null
};
const source = {
	part_id: 'part',
	photo_index: 0,
	source_sha256: 'original',
	view_id: 'unwarped',
	coordinate_space: 'working' as const,
	bbox: [
		[10, 20],
		[40, 20],
		[40, 60]
	],
	block_id: 'block',
	text_span: null
};
beforeEach(() =>
	session.set({ id: 'viewer', username: 'alice', is_admin: false, can_see_all: false })
);

afterEach(cleanup);
it('uses the exact working view and its own dimensions', () => {
	render(ExtractionSourceViewer, { views: [view], sources: [source] });
	const img = screen.getByRole('img');
	expect(img.getAttribute('src')).toBe('/exact-working.png?vb_expected_user=viewer');
	expect(img.getAttribute('width')).toBe('1200');
	expect(img.getAttribute('height')).toBe('1800');
	expect(document.querySelector('svg')?.getAttribute('viewBox')).toBe('0 0 1200 1800');
	expect(document.querySelector('polygon')?.getAttribute('points')).toBe('10,20 40,20 40,60');
});
it('never draws a box for unknown geometry', () => {
	render(ExtractionSourceViewer, {
		views: [view],
		sources: [{ ...source, coordinate_space: 'unknown', view_id: null }]
	});
	expect(document.querySelector('polygon')).toBeNull();
	expect(screen.getByText('Source mapping unknown')).toBeTruthy();
});
it('reports display failure without claiming the original is missing', async () => {
	render(ExtractionSourceViewer, { views: [view], sources: [source] });
	await fireEvent.error(screen.getByRole('img'));
	expect(screen.getByRole('alert').textContent).toContain('Source image unavailable');
	expect(screen.queryByText('Original unavailable')).toBeNull();
});
it('allows selecting an original without drawing working coordinates on it', async () => {
	render(ExtractionSourceViewer, {
		views: [
			view,
			{
				...view,
				view_id: 'original',
				coordinate_space: 'original',
				width: 600,
				height: 900,
				url: '/original.png'
			}
		],
		sources: [source]
	});
	await fireEvent.change(screen.getByRole('combobox'), { target: { value: 'original' } });
	expect(screen.getByRole('img').getAttribute('src')).toBe('/original.png?vb_expected_user=viewer');
	expect(document.querySelector('polygon')).toBeNull();
});

it('offers bounded zoom without replacing the selected image', async () => {
	render(ExtractionSourceViewer, { views: [view], sources: [source] });
	const image = screen.getByRole('img');
	await fireEvent.click(screen.getByRole('button', { name: 'Zoom in' }));
	expect(screen.getByRole('img')).toBe(image);
	expect(screen.getByText('125%')).toBeTruthy();
});

it('reselects evidence on a repeated source request after manual page navigation', async () => {
	const views = [view, { ...view, view_id: 'other', photo_index: 1, url: '/other.png' }];
	const rendered = render(ExtractionSourceViewer, { views, sources: [source] });
	await fireEvent.change(screen.getByRole('combobox'), { target: { value: 'other' } });
	expect(screen.getByRole('img').getAttribute('src')).toContain('/other.png');
	await rendered.rerender({ views, sources: [source] });
	expect(screen.getByRole('img').getAttribute('src')).toContain('/exact-working.png');
});
