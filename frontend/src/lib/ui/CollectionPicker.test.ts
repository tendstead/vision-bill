import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import CollectionPicker from './CollectionPicker.svelte';
import type { CollectionSummary } from '$lib/types';

const c = (id: string, name: string, color: string | null): CollectionSummary => ({
	id,
	name,
	color,
	start_date: null,
	end_date: null,
	active: false,
	created_at: null,
	receipt_count: 0,
	totals: []
});

const options = [c('1', 'Berlin', '#3B82F6'), c('2', 'Christmas', null)];

afterEach(cleanup);

describe('CollectionPicker', () => {
	it('single mode selects an option', async () => {
		const onchange = vi.fn();
		render(CollectionPicker, { options, mode: 'single', value: '', onchange });

		await fireEvent.click(screen.getByRole('combobox'));
		await fireEvent.click(screen.getByText('Berlin'));

		expect(onchange).toHaveBeenCalledWith('1');
	});

	it('multi mode toggles a selection on and off', async () => {
		const onchange = vi.fn();
		render(CollectionPicker, { options, mode: 'multi', value: [], onchange });

		await fireEvent.click(screen.getByRole('combobox'));
		await fireEvent.click(screen.getByText('Christmas'));
		expect(onchange).toHaveBeenLastCalledWith(['2']);
	});

	it('single mode: Enter on the default (All) row selects empty id', async () => {
		const onchange = vi.fn();
		render(CollectionPicker, { options, mode: 'single', value: '2', onchange });
		const input = screen.getByRole('combobox');

		await fireEvent.click(input);
		await fireEvent.keyDown(input, { key: 'Enter' });

		expect(onchange).toHaveBeenCalledWith('');
	});

	it('single mode: ArrowDown then Enter selects the first option', async () => {
		const onchange = vi.fn();
		render(CollectionPicker, { options, mode: 'single', value: '', onchange });
		const input = screen.getByRole('combobox');

		await fireEvent.click(input);
		await fireEvent.keyDown(input, { key: 'ArrowDown' });
		await fireEvent.keyDown(input, { key: 'Enter' });

		expect(onchange).toHaveBeenCalledWith('1');
	});

	it('multi mode: allowCreate creates from a non-matching query', async () => {
		const onchange = vi.fn();
		const oncreate = vi.fn();
		render(CollectionPicker, {
			options,
			mode: 'multi',
			value: [],
			allowCreate: true,
			onchange,
			oncreate
		});
		const input = screen.getByRole('combobox');

		await fireEvent.click(input);
		await fireEvent.input(input, { target: { value: 'New Year' } });
		await fireEvent.keyDown(input, { key: 'Enter' });

		expect(oncreate).toHaveBeenCalledWith('New Year');
	});
});
it('blocks already-open custom membership and create options while disabled', async () => {
	const onchange = vi.fn(),
		oncreate = vi.fn();
	const props = {
		options,
		mode: 'multi' as const,
		value: [] as string[],
		allowCreate: true,
		onchange,
		oncreate
	};
	const view = render(CollectionPicker, props);
	const input = screen.getByRole('combobox');
	await fireEvent.click(input);
	await view.rerender({ ...props, disabled: true });
	await fireEvent.click(screen.getByRole('option', { name: 'Berlin' }));
	expect(onchange).not.toHaveBeenCalled();
	await fireEvent.input(input, { target: { value: 'Late' } });
	await fireEvent.click(screen.getByRole('option', { name: '+ Create “Late”' }));
	expect(oncreate).not.toHaveBeenCalled();
});

it('supports native text editing, filtering and inline creation', async () => {
	const create = vi.fn();
	render(CollectionPicker, {
		options,
		mode: 'multi',
		value: [],
		allowCreate: true,
		oncreate: create,
		onchange: vi.fn()
	});
	const input = screen.getByRole('combobox') as HTMLInputElement;
	await fireEvent.click(input);
	// Native text-selection editing is unavailable on checkboxes. Unlike
	// target.value injection, this executes the editable input contract.
	input.setSelectionRange(0, 0);
	input.setRangeText('Berl', 0, 0, 'end');
	await fireEvent.input(input);
	expect(screen.getByRole('option', { name: 'Berlin' })).not.toBeNull();
	expect(screen.queryByRole('option', { name: 'Christmas' })).toBeNull();
	input.setRangeText('Summer Trip', 0, input.value.length, 'end');
	await fireEvent.input(input);
	await fireEvent.keyDown(input, { key: 'Enter' });
	expect(create).toHaveBeenCalledWith('Summer Trip');
});
