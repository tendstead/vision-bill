import { cleanup, fireEvent, render, screen } from '@testing-library/svelte';
import { afterEach, expect, it, vi } from 'vitest';
import ReviewFindings from './ReviewFindings.svelte';
import type { ReviewFinding } from '$lib/types';
const findings: ReviewFinding[] = [
	{
		id: 'i',
		code: 'original_unavailable',
		severity: 'integrity' as const,
		field_path: 'originals',
		source_refs: [],
		comparison: {}
	},
	{
		id: 'r',
		code: 'total_mismatch',
		severity: 'review' as const,
		field_path: 'total',
		source_refs: [],
		comparison: { expected: '1.00000001' }
	}
];
afterEach(cleanup);
it('only review findings accept comments and reset on a new review version', async () => {
	const onChange = vi.fn();
	const view = render(ReviewFindings, { findings, revision: 1, onChange });
	expect(screen.getAllByRole('textbox')).toHaveLength(1);
	await fireEvent.input(screen.getByRole('textbox'), {
		target: { value: 'Printed total checked' }
	});
	expect(onChange).toHaveBeenLastCalledWith([
		{ finding_id: 'r', comment: 'Printed total checked' }
	]);
	await view.rerender({ findings, revision: 2, onChange });
	expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('');
	expect(onChange).toHaveBeenLastCalledWith([]);
});
it('clears prior comments as soon as material editing starts', async () => {
	const onChange = vi.fn();
	const view = render(ReviewFindings, { findings, revision: 1, onChange });
	await fireEvent.input(screen.getByRole('textbox'), { target: { value: 'Old acknowledgement' } });
	await view.rerender({ findings, revision: 1, onChange, editable: false });
	await view.rerender({ findings, revision: 1, onChange, editable: true });
	expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('');
	expect(onChange).toHaveBeenLastCalledWith([]);
});

it.each(['en', 'de'] as const)(
	'explains checks2 and exact comparison strings in %s',
	async (language) => {
		const { setLocale } = await import('$lib/i18n');
		setLocale(language);
		const checks = [
			'uncertain_net_subtotal_basis',
			'unknown_discount_treatment',
			'missing_tax_information',
			'amount_discrepancy',
			'unknown_source_mapping'
		];
		render(ReviewFindings, {
			revision: 2,
			findings: checks.map((code, i): ReviewFinding => {
				const comparison: Record<string, string | null> =
					i === 3
						? { observed: '-0.00000000100', expected: '1.234567890123', tolerance: '0.05' }
						: {};
				return {
					id: String(i),
					code,
					severity: 'review',
					field_path: 'subtotal',
					source_refs: [],
					comparison
				};
			})
		});
		expect(
			screen.getByText(
				language === 'en'
					? /A net subtotal cannot be compared reliably/
					: /Eine Nettozwischensumme lässt sich nicht zuverlässig/,
				{ selector: 'p' }
			)
		).not.toBeNull();
		expect(
			screen.getByText(
				language === 'en' ? /Specify whether the discount/ : /Gib an, ob der Rabatt/,
				{ selector: 'p' }
			)
		).not.toBeNull();
		expect(
			screen.getByText(language === 'en' ? /Tax information is missing/ : /Steuerangaben fehlen/, {
				selector: 'p'
			})
		).not.toBeNull();
		expect(
			screen.getByText(
				language === 'en' ? /Observed: -0.00000000100/ : /Beobachtet: -0.00000000100/
			)
		).not.toBeNull();
		expect(
			screen.getByText(language === 'en' ? /Expected: 1.234567890123/ : /Erwartet: 1.234567890123/)
		).not.toBeNull();
		expect(
			screen.getByText(language === 'en' ? /Tolerance: 0.05/ : /Toleranz: 0.05/)
		).not.toBeNull();
		setLocale('en');
	}
);

it.each(['en', 'de'] as const)(
	'explains the retained OCR fallback finding on saved review in %s',
	async (language) => {
		const { setLocale } = await import('$lib/i18n');
		// This serialized shape is checked against the actual pipeline and saved-review
		// evaluator by test_fallback_preserves_ocr_failure_as_stable_finding.
		const { default: retained } = await import('$lib/test-fixtures/retained-ocr-finding.json');
		const saved: ReviewFinding = { ...retained, severity: 'info', id: 'saved-material-finding-id' };
		const onChange = vi.fn();
		setLocale(language);
		try {
			render(ReviewFindings, { findings: [saved, findings[1]], revision: 9, onChange });
			expect(
				screen.getByText(
					language === 'en'
						? 'Information: OCR could not read every photo in this receipt.'
						: 'Information: OCR konnte nicht jedes Foto dieses Belegs lesen.'
				)
			).not.toBeNull();
			expect(
				screen.queryByText(
					/Additional diagnostic: ocr page failed|Zusätzliche Diagnose: ocr page failed/
				)
			).toBeNull();
			expect(screen.getAllByRole('textbox')).toHaveLength(1);
			await fireEvent.input(screen.getByRole('textbox'), { target: { value: 'Checked original' } });
			expect(onChange).toHaveBeenLastCalledWith([{ finding_id: 'r', comment: 'Checked original' }]);
			expect(saved).toEqual({ ...retained, id: 'saved-material-finding-id' });
			expect(
				screen.getByText(language === 'en' ? 'Expected: 1.00000001' : 'Erwartet: 1.00000001')
			).not.toBeNull();
		} finally {
			setLocale('en');
		}
	}
);
