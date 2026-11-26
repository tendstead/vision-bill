import type { ClassificationEnrichment } from '$lib/types';

export type ExperimentSettings = {
	pipeline: 'direct_vision' | 'ocr' | 'compare';
	temperature: number | undefined;
	thinking: 'default' | 'on' | 'off';
	includeClassification: boolean;
};

export const defaultExperiment = (): ExperimentSettings => ({
	pipeline: 'direct_vision',
	temperature: undefined,
	thinking: 'default',
	includeClassification: true
});

export function experimentRequest(settings: ExperimentSettings) {
	return {
		pipeline: settings.pipeline === 'compare' ? ('direct_vision' as const) : settings.pipeline,
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
		telemetry?: { stage_ms?: Record<string, number> };
	};
	expected?: unknown;
	actual?: unknown;
};

export type IndividualResult = IndividualResultBase &
	(
		| { version?: '3' }
		| {
				version: '4' | '5';
				legacy_detail_version?: '3';
				scored_fields: string[];
				extraction_failed: boolean;
				full_receipt_correct: boolean;
				item_tp: number | null;
				item_fp: number | null;
				item_fn: number | null;
				item_matches: [number, number][];
				precision: number | null;
				recall: number | null;
				exact_item_count: boolean | null;
				exact_item_order: boolean | null;
				exact_amounts: boolean | null;
		  }
	);

export function selectedReceiptIds(value: string) {
	return [
		...new Set(
			value
				.split(',')
				.map((id) => id.trim().toLowerCase())
				.filter(Boolean)
		)
	];
}
