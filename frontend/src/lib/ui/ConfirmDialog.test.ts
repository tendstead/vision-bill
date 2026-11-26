import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, expect, it, vi } from 'vitest';
import ConfirmDialog from './ConfirmDialog.svelte';
afterEach(cleanup);
it('focuses inside, wraps Tab, closes on Escape, and restores the trigger', async () => {
	const trigger = document.createElement('button');
	document.body.append(trigger);
	trigger.focus();
	const cancel = vi.fn();
	const view = render(ConfirmDialog, {
		open: true,
		title: 'Delete collection',
		body: 'Really?',
		onConfirm: vi.fn(),
		onCancel: cancel
	});
	const dialog = screen.getByRole('dialog');
	expect(dialog.contains(document.activeElement)).toBe(true);
	const buttons = dialog.querySelectorAll('button');
	buttons[buttons.length - 1].focus();
	await fireEvent.keyDown(dialog, { key: 'Tab' });
	expect(document.activeElement).toBe(buttons[0]);
	await fireEvent.keyDown(dialog, { key: 'Escape' });
	expect(cancel).toHaveBeenCalledOnce();
	await view.rerender({ open: false });
	expect(document.activeElement).toBe(trigger);
	trigger.remove();
});
it('keeps a busy operation open on Escape', async () => {
	const cancel = vi.fn();
	render(ConfirmDialog, {
		open: true,
		busy: true,
		title: 'Wait',
		body: 'Saving',
		onConfirm: vi.fn(),
		onCancel: cancel
	});
	await fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' });
	expect(cancel).not.toHaveBeenCalled();
});

it('continues to dismiss on a backdrop click only while idle', async () => {
	const cancel = vi.fn();
	const dialog = render(ConfirmDialog, {
		open: true,
		busy: true,
		title: 'Wait',
		body: 'Saving',
		onConfirm: vi.fn(),
		onCancel: cancel
	});
	await fireEvent.click(screen.getByRole('dialog'), { clientX: 10, clientY: 10 });
	expect(cancel).not.toHaveBeenCalled();
	await dialog.rerender({ busy: false });
	await fireEvent.click(screen.getByRole('dialog'), { clientX: 10, clientY: 10 });
	expect(cancel).toHaveBeenCalledOnce();
});
