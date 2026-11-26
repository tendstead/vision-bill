import { describe, expect, it } from 'vitest';
import { defaultExperiment, experimentRequest } from './benchmark-experiments';

describe('benchmark experiment options', () => {
	it('uses the provider default when temperature and thinking are unspecified', () => {
		const request = experimentRequest(defaultExperiment());
		expect(request.temperature).toBeNull();
		expect(request.think).toBeNull();
		expect(request.allow_ocr_fallback).toBe(false);
	});
	it('preserves an explicit zero temperature and disabled thinking', () => {
		const request = experimentRequest({ ...defaultExperiment(), temperature: 0, thinking: 'off' });
		expect(request.temperature).toBe(0);
		expect(request.think).toBe(false);
	});
	it('keeps classification selectable without granting fallback from legacy state', () => {
		const legacySettings = {
			...defaultExperiment(),
			pipeline: 'ocr' as const,
			includeClassification: false,
			allowFallback: true
		};
		const request = experimentRequest(legacySettings);
		expect(request.pipeline).toBe('ocr');
		expect(request.include_classification).toBe(false);
		expect(request.allow_ocr_fallback).toBe(false);
	});
});
