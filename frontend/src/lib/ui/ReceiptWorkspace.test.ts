import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, expect, it, vi } from 'vitest';
import Fixture from './fixtures/ReceiptWorkspaceFixture.svelte';
afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});
it('retains edited data, photo position and field focus between mobile views', async () => {
	render(Fixture);
	const input = screen.getByLabelText('Merchant') as HTMLInputElement;
	const photo = screen.getByTestId('photo-position');
	input.focus();
	await fireEvent.input(input, { target: { value: 'Corrected merchant' } });
	photo.scrollTop = 48;
	await fireEvent.click(screen.getByRole('button', { name: 'Receipt' }));
	await fireEvent.click(screen.getByRole('button', { name: 'Data' }));
	expect(screen.getByLabelText('Merchant')).toBe(input);
	expect(input.value).toBe('Corrected merchant');
	expect(screen.getByTestId('photo-position')).toBe(photo);
	expect(photo.scrollTop).toBe(48);
	expect(document.activeElement).toBe(input);
});

it('shows only the data panel for image-less receipts', () => {
	render(Fixture, { hasPhoto: false });
	expect(screen.queryByRole('button', { name: 'Receipt' })).toBeNull();
	expect(screen.queryByTestId('photo-position')).toBeNull();
	expect(document.querySelector('.workspace')?.classList.contains('without-photo')).toBe(true);
});
it('reopens the receipt for a repeated source request after returning to Data', async () => {
	render(Fixture);
	const source = screen.getByRole('button', { name: 'Show evidence' });
	await fireEvent.click(source);
	expect(screen.getByRole('button', { name: 'Receipt' }).getAttribute('aria-pressed')).toBe('true');
	await fireEvent.click(screen.getByRole('button', { name: 'Data' }));
	await fireEvent.click(source);
	expect(screen.getByRole('button', { name: 'Receipt' }).getAttribute('aria-pressed')).toBe('true');
});

it('hands focus to the visible receipt selector for mobile source requests', async () => {
	vi.stubGlobal('matchMedia', () => ({ matches: true }));
	render(Fixture);
	const field = screen.getByRole('button', { name: 'Show evidence' });
	field.focus();
	await fireEvent.click(screen.getByRole('button', { name: 'Show evidence' }));
	expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Receipt' }));
	await fireEvent.click(screen.getByRole('button', { name: 'Data' }));
	expect(document.activeElement).toBe(field);
});
it('keeps focus in the data panel for desktop source requests', async () => {
	vi.stubGlobal('matchMedia', () => ({ matches: false }));
	render(Fixture);
	const field = screen.getByRole('button', { name: 'Show evidence' });
	field.focus();
	await fireEvent.click(screen.getByRole('button', { name: 'Show evidence' }));
	expect(document.activeElement).toBe(field);
});
