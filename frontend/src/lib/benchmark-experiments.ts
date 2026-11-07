export type ExperimentSettings = {
	pipeline: 'upload' | 'direct_vision' | 'ocr';
	temperature: number | undefined;
	thinking: 'default' | 'on' | 'off';
	includeClassification: boolean;
	allowFallback: boolean;
};

export const defaultExperiment = (): ExperimentSettings => ({
	pipeline: 'upload',
	temperature: undefined,
	thinking: 'default',
	includeClassification: true,
	allowFallback: false
});

export function experimentRequest(settings: ExperimentSettings) {
	return {
		pipeline: settings.pipeline,
		temperature: settings.temperature ?? null,
		think: settings.thinking === 'default' ? null : settings.thinking === 'on',
		include_classification: settings.includeClassification,
		allow_ocr_fallback: settings.allowFallback
	};
}

export type IndividualResult = {
	model_id: string;
	receipt_id: string;
	success: boolean;
	error?: string;
	scores?: Record<string, number | null>;
	rows?: { missing: number; extra: number; matched: number };
	differences?: { path: string; expected: unknown; actual: unknown; equivalent: boolean }[];
	metadata?: Record<string, unknown>;
	expected?: unknown;
	actual?: unknown;
};
