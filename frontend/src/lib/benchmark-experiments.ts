import type { ClassificationEnrichment } from '$lib/types';

export type ExperimentSettings = {
	pipeline: 'upload' | 'direct_vision' | 'ocr';
	temperature: number | undefined;
	thinking: 'default' | 'on' | 'off';
	includeClassification: boolean;
};

export const defaultExperiment = (): ExperimentSettings => ({
	pipeline: 'upload',
	temperature: undefined,
	thinking: 'default',
	includeClassification: true
});

export function experimentRequest(settings: ExperimentSettings) {
	return {
		pipeline: settings.pipeline,
		temperature: settings.temperature ?? null,
		think: settings.thinking === 'default' ? null : settings.thinking === 'on',
		include_classification: settings.includeClassification,
		allow_ocr_fallback: false
	};
}

type IndividualResultBase = {
	model_id: string;
	receipt_id: string;
	success: boolean;
	error?: string;
	scores?: Record<string, number | null>;
	rows?: { missing: number; extra: number; matched: number };
	differences?: { path: string; expected: unknown; actual: unknown; equivalent: boolean }[];
	metadata?: Record<string, unknown> & {
		classification_enrichment?: ClassificationEnrichment;
	};
	expected?: unknown;
	actual?: unknown;
};

export type IndividualResult = IndividualResultBase &
	(
		| { version?: '3' }
		| {
				version: '4';
				legacy_detail_version: '3';
				scored_fields: string[];
				extraction_failed: boolean;
				full_receipt_correct: boolean;
				item_tp: number;
				item_fp: number;
				item_fn: number;
				item_matches: [number, number][];
				precision: number;
				recall: number;
				exact_item_count: boolean;
				exact_item_order: boolean;
				exact_amounts: boolean;
		  }
	);
