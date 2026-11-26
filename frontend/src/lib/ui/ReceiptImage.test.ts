import { session } from '$lib/session-lifetime';
import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import ReceiptImage from './ReceiptImage.svelte';

const id = '00000000-0000-0000-0000-000000000001';

beforeEach(() => {
	session.set({ id: 'viewer', username: 'alice', is_admin: false, can_see_all: false });
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
		expect(img.getAttribute('src')).toBe(`/api/v1/images/${id}/thumb?vb_expected_user=viewer`);
	});

	it('renders the full image (policy) when there is no thumbnail path', () => {
		const { container } = render(ReceiptImage, {
			imageId: id,
			thumbnailPath: null,
			alt: 'ACME',
			class: 'size-12'
		});
		const img = container.querySelector('img')!;
		expect(img.getAttribute('src')).toBe(`/api/v1/images/${id}/file?vb_expected_user=viewer`);
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

	it('gives an unlabelled thumbnail an accessible preview action', () => {
		render(ReceiptImage, { imageId: id, thumbnailPath: null, alt: '' });
		expect(screen.getByRole('button', { name: 'View full size' })).toBeTruthy();
	});

	it('uses an image placeholder if private media has no active session', () => {
		session.set(null);
		const { container } = render(ReceiptImage, { imageId: id, thumbnailPath: null, alt: '' });
		expect(container.querySelector('img')).toBeNull();
		expect(container.querySelector('svg')).not.toBeNull();
	});

	it('uses the next receipt thumbnail even if the previous thumbnail failed', async () => {
		const { container, rerender } = render(ReceiptImage, {
			imageId: id,
			thumbnailPath: '/srv/receipt.thumb.webp',
			alt: 'ACME'
		});
		await fireEvent.error(container.querySelector('img')!);
		expect(container.querySelector('img')!.getAttribute('src')).toContain('/file?');
		const nextId = '00000000-0000-0000-0000-000000000002';
		await rerender({ imageId: nextId, thumbnailPath: '/srv/next.thumb.webp', alt: 'Next' });
		expect(container.querySelector('img')!.getAttribute('src')).toContain(`/${nextId}/thumb?`);
	});

	it('clicking opens the lightbox with the full image, Esc closes it', async () => {
		render(ReceiptImage, {
			imageId: id,
			thumbnailPath: '/srv/receipt.thumb.webp',
			alt: 'ACME',
			class: 'size-12'
		});
		const opener = screen.getByRole('button', { name: 'ACME' });
		opener.focus();
		await fireEvent.click(opener);

		const dialog = screen.getByRole('dialog');
		const full = dialog.querySelector('img')!;
		expect(full.getAttribute('src')).toBe(`/api/v1/images/${id}/file?vb_expected_user=viewer`);
		const close = screen.getByRole('button', { name: 'Close' });
		expect(document.activeElement).toBe(close);

		await fireEvent.keyDown(close, { key: 'Escape' });
		expect(screen.queryByRole('dialog')).toBeNull();
		expect(document.activeElement).toBe(opener);
	});
});
