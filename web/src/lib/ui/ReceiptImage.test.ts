import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import ReceiptImage from './ReceiptImage.svelte';

const id = '00000000-0000-0000-0000-000000000001';

beforeEach(() => {
	// Mock matchMedia so the mobile-view thumbnail check works in jsdom
	Object.defineProperty(window, 'matchMedia', {
		writable: true,
		value: (query: string) => ({
			matches: query.includes('max-width: 768px'),
			media: query,
			onchange: null,
			addListener: () => {},
			removeListener: () => {},
			addEventListener: () => {},
			removeEventListener: () => {},
			dispatchEvent: () => false
		})
	});
});

afterEach(cleanup);

describe('ReceiptImage', () => {
	it('renders the thumbnail URL when a thumbnail path is present', () => {
		const { container } = render(ReceiptImage, {
			imageId: id,
			thumbnailPath: '/srv/receipt.thumb.webp',
			alt: 'ACME',
			class: 'size-12'
		});
		const img = container.querySelector('img')!;
		expect(img.getAttribute('src')).toBe(`/api/v1/images/${id}/thumb`);
	});

	it('renders the full image (policy) when there is no thumbnail path', () => {
		const { container } = render(ReceiptImage, {
			imageId: id,
			thumbnailPath: null,
			alt: 'ACME',
			class: 'size-12'
		});
		const img = container.querySelector('img')!;
		expect(img.getAttribute('src')).toBe(`/api/v1/images/${id}/file`);
	});

	it('renders the icon when there is no image at all', () => {
		const { container } = render(ReceiptImage, {
			imageId: null,
			thumbnailPath: null,
			alt: 'ACME',
			class: 'size-12'
		});
		expect(container.querySelector('img')).toBeNull();
		expect(container.querySelector('svg')).not.toBeNull();
	});

	it('clicking opens the lightbox with the full image, Esc closes it', async () => {
		const { container } = render(ReceiptImage, {
			imageId: id,
			thumbnailPath: '/srv/receipt.thumb.webp',
			alt: 'ACME',
			class: 'size-12'
		});
		await fireEvent.click(container.querySelector('img')!);

		const dialog = screen.getByRole('dialog');
		const full = dialog.querySelector('img')!;
		expect(full.getAttribute('src')).toBe(`/api/v1/images/${id}/file`);

		await fireEvent.keyDown(window, { key: 'Escape' });
		expect(screen.queryByRole('dialog')).toBeNull();
	});
});
