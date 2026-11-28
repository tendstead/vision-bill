import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it } from 'vitest';
import ReceiptLink from './ReceiptLink.svelte';
import ReceiptPreview from './ReceiptPreview.svelte';
import { hidePreview, preview } from './receipt-preview.svelte';

afterEach(() => {
	cleanup();
	hidePreview();
});

const ID = '33333333-3333-3333-3333-333333333333';

describe('receipt link and hover preview', () => {
	it('links to the receipt and shows its thumbnail only while hovered or focused', async () => {
		render(ReceiptLink, { receiptId: ID, imageId: 'img-1' });
		render(ReceiptPreview);
		const link = screen.getByRole('link', { name: '33333333' });
		expect(link.getAttribute('href')).toContain(`/receipts/${ID}`);
		expect(screen.queryByTestId('receipt-preview')).toBeNull();

		await fireEvent.pointerEnter(link);
		const img = screen.getByTestId('receipt-preview') as HTMLImageElement;
		expect(img.getAttribute('src')).toBe('/api/v1/images/img-1/thumb');

		// No thumbnail stored: fall back to the photo, then give up.
		await fireEvent.error(img);
		expect(img.getAttribute('src')).toBe('/api/v1/images/img-1/file');
		await fireEvent.error(img);
		expect(screen.queryByTestId('receipt-preview')).toBeNull();

		await fireEvent.pointerEnter(link);
		expect(screen.getByTestId('receipt-preview')).toBeTruthy();
		await fireEvent.pointerLeave(link);
		expect(screen.queryByTestId('receipt-preview')).toBeNull();
		await fireEvent.focus(link);
		expect(preview.imageId).toBe('img-1');
		await fireEvent.blur(link);
		expect(preview.imageId).toBeNull();
	});

	it('does nothing without an image and does not link deleted receipts', async () => {
		render(ReceiptLink, { receiptId: ID, imageId: null });
		await fireEvent.pointerEnter(screen.getByRole('link'));
		expect(preview.imageId).toBeNull();
		cleanup();
		render(ReceiptLink, { receiptId: ID, imageId: 'img-1', exists: false });
		expect(screen.queryByRole('link')).toBeNull();
		expect(screen.getByTitle('This receipt was deleted').textContent).toBe('33333333');
	});
});
