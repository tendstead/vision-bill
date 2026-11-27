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
