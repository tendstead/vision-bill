import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { session } from '$lib/session-lifetime';
import { api } from '$lib/api/client';
import { queryClient } from '$lib/query/client';
import type { SourceRef, SourceView } from '$lib/types';
import ReceiptPhotos from './ReceiptPhotos.svelte';
beforeEach(() => {
	session.set({ id: 'viewer', username: 'Alice', is_admin: false, can_see_all: false });
	vi.spyOn(api, 'getImage').mockResolvedValue({ additional_images: ['second'] } as never);
	queryClient.clear();
});
afterEach(() => {
	cleanup();
	queryClient.clear();
	vi.restoreAllMocks();
});
it('opens the correct original photo for evidence without a view ID', async () => {
	render(ReceiptPhotos, {
		imageId: 'job',
		alt: 'Receipt',
		sources: [{ photo_index: 1, view_id: null, coordinate_space: 'original' } as SourceRef]
	});
	await waitFor(() => expect(screen.getByRole('img').getAttribute('src')).toContain('part=1'));
	expect(screen.getByRole('button', { name: 'Photo 2' }).getAttribute('aria-pressed')).toBe('true');
});
it('does not display an unrelated working photo for unmatched evidence', () => {
	render(ReceiptPhotos, {
		imageId: 'job',
		alt: 'Receipt',
		views: [
			{
				view_id: 'other',
				photo_index: 0,
				coordinate_space: 'working',
				width: 100,
				height: 100,
				url: '/other.png'
			} as SourceView
		],
		sources: [{ view_id: 'missing', photo_index: 1, coordinate_space: 'working' } as SourceRef]
	});
	expect(screen.queryByRole('img', { name: 'Source view' })).toBeNull();
	expect(screen.getByText('Source mapping unknown')).toBeTruthy();
});

it('keeps original source views out of the working image selector', () => {
	const working = {
		view_id: 'first',
		photo_index: 0,
		coordinate_space: 'working',
		width: 100,
		height: 100,
		url: '/first.png'
	} as SourceView;
	render(ReceiptPhotos, {
		imageId: 'job',
		alt: 'Receipt',
		views: [
			working,
			{ ...working, view_id: 'second', photo_index: 1 },
			{ ...working, view_id: 'original', coordinate_space: 'original' }
		],
		sources: [{ view_id: 'first', photo_index: 0, coordinate_space: 'working' } as SourceRef]
	});
	expect(screen.queryByRole('option', { name: /original/ })).toBeNull();
	expect((screen.getByRole('combobox') as HTMLSelectElement).options.length).toBe(2);
});

it.each([1, 2])(
	'opens a working image after original evidence with %s working pages',
	async (pages) => {
		const working = {
			view_id: 'first',
			photo_index: 0,
			coordinate_space: 'working',
			width: 100,
			height: 100,
			url: '/first.png'
		} as SourceView;
		render(ReceiptPhotos, {
			imageId: 'job',
			alt: 'Receipt',
			views:
				pages === 1
					? [working]
					: [working, { ...working, view_id: 'second', photo_index: 1, url: '/second.png' }],
			sources: [{ photo_index: 1, view_id: null, coordinate_space: 'original' } as SourceRef]
		});
		await waitFor(() =>
			expect(screen.getByRole('button', { name: 'Photo 2' }).getAttribute('aria-pressed')).toBe(
				'true'
			)
		);
		await fireEvent.click(screen.getByRole('button', { name: 'Working image' }));
		expect(screen.getByRole('img', { name: 'Source view' }).getAttribute('src')).toContain(
			pages === 1 ? '/first.png' : '/second.png'
		);
		expect(document.querySelector('polygon')).toBeNull();
	}
);

it('resumes strict matching when new evidence arrives after manual browsing', async () => {
	const views = [
		{
			view_id: 'first',
			photo_index: 0,
			coordinate_space: 'working',
			width: 100,
			height: 100,
			url: '/first.png'
		} as SourceView
	];
	const sources = [
		{ view_id: 'missing', photo_index: 1, coordinate_space: 'working' } as SourceRef
	];
	const rendered = render(ReceiptPhotos, { imageId: 'job', alt: 'Receipt', views, sources });
	await screen.findByRole('button', { name: 'Photo 2' });
	await fireEvent.click(screen.getByRole('button', { name: 'Working image' }));
	expect(screen.getByRole('img', { name: 'Source view' }).getAttribute('src')).toContain(
		'/first.png'
	);
	await rendered.rerender({ imageId: 'job', alt: 'Receipt', views, sources: [...sources] });
	expect(screen.queryByRole('img', { name: 'Source view' })).toBeNull();
});

it('distinguishes an explicit unknown-source request from no evidence request', async () => {
	const views = [
		{
			view_id: 'first',
			photo_index: 0,
			coordinate_space: 'working',
			width: 100,
			height: 100,
			url: '/first.png'
		} as SourceView
	];
	const rendered = render(ReceiptPhotos, { imageId: 'job', alt: 'Receipt', views });
	await fireEvent.click(screen.getByRole('button', { name: 'Original photos' }));
	await rendered.rerender({ imageId: 'job', alt: 'Receipt', views, sources: [] });
	expect(screen.getByRole('button', { name: 'Working image' }).getAttribute('aria-pressed')).toBe(
		'true'
	);
	expect(screen.queryByRole('img', { name: 'Source view' })).toBeNull();
	expect(document.querySelector('polygon')).toBeNull();
	await fireEvent.click(screen.getByRole('button', { name: 'Working image' }));
	expect(screen.getByRole('img', { name: 'Source view' }).getAttribute('src')).toContain(
		'/first.png'
	);
});
