export type ImageStatus =
	'pending' | 'processing' | 'analyzed' | 'failed' | 'timed_out' | 'unreadable' | 'needs_review';

export type ReceiptStatus = 'unverified' | 'verified';

export type Category =
	| 'grocery'
	| 'electronics'
	| 'clothing'
	| 'restaurant'
	| 'fuel'
	| 'pharmacy'
	| 'entertainment'
	| 'other';

export type PaymentMethod =
	'cash' | 'credit_card' | 'debit_card' | 'mobile_payment' | 'check' | 'other' | 'unknown';

export type EnrichmentReason =
	| 'invalid_response'
	| 'request_rejected'
	| 'not_dispatched'
	| 'classification_budget_exhausted'
	| 'classification_disabled'
	| 'incomplete_observations'
	| 'recovered_without_classification';

export interface ClassificationEnrichment {
	state: 'not_recorded' | 'pending' | 'completed' | 'unavailable' | 'skipped';
	reason_code: EnrichmentReason | null;
	missing_fields: string[];
}

export interface SourceView {
	view_id: string;
	photo_index: number;
	coordinate_space: 'original' | 'working';
	width: number;
	height: number;
	sha256: string;
	url: string;
	transforms: Record<string, unknown>[];
	inverse_mapping: null;
}

export interface SourceRef {
	part_id: string;
	photo_index: number;
	source_sha256: string;
	view_id?: string | null;
	coordinate_space: 'original' | 'working' | 'unknown';
	block_id?: string | null;
	bbox?: number[][] | null;
	text_span?: [number, number] | null;
}
export interface AmountPolicy {
	tax_basis: 'gross' | 'net' | 'unknown';
	discount_treatment: 'included_in_lines' | 'subtract' | 'unknown';
	tip_treatment: 'included' | 'add' | 'outside_receipt' | 'none' | 'unknown';
	subtotal_origin: 'printed' | 'derived_net' | 'unknown';
	tax_information: 'present' | 'explicit_zero' | 'missing';
}
export interface ReviewFinding {
	id: string;
	code: string;
	severity: 'integrity' | 'review' | 'info';
	field_path: string;
	source_refs: SourceRef[];
	comparison: Record<string, string | null>;
}
export interface DraftLineItem {
	description?: string | null;
	spending_category?: string | null;
	quantity?: string | number | null;
	unit_price?: string | null;
	total_price?: string | null;
	tags?: string[] | null;
}
export interface DraftTaxLine {
	name?: string | null;
	rate?: string | number | null;
	amount?: string | null;
}
export interface ExtractionDraft {
	confidence?: number | null;
	merchant_name?: string | null;
	merchant_address?: string | null;
	receipt_number?: string | null;
	date?: string | null;
	time?: string | null;
	currency?: string | null;
	category?: Category | null;
	payment_method?: PaymentMethod | null;
	subtotal?: string | null;
	discount_total?: string | null;
	tax_total?: string | null;
	tip?: string | null;
	total?: string | null;
	line_items?: DraftLineItem[] | null;
	taxes?: DraftTaxLine[] | null;
}
export interface DraftView {
	image_id: string;
	revision: number;
	document: ExtractionDraft;
	amount_policy: AmountPolicy;
	findings: ReviewFinding[];
	source_views: SourceView[];
	finalized_receipt_id: string | null;
	field_sources: Record<string, SourceRef[]>;
	overlap_decisions: Record<string, unknown>[];
}
export interface DraftPatch {
	expected_revision: number;
	changes: Partial<ExtractionDraft> & {
		amount_policy?: AmountPolicy;
		field_sources?: Record<string, SourceRef[]>;
		overlap_decisions?: Record<string, unknown>[];
	};
}
export interface DraftFinalize {
	expected_revision: number;
	request_key: string;
}

export interface SubmissionStatus {
	generation: number;
	terminal: boolean;
	attempt_count: number;
	max_attempts: number;
	retry_at: string | null;
	result_code: string | null;
	wait_reason: string | null;
	draft_available: boolean;
	observations_available: boolean;
	latest_extraction_id: string | null;
	source_views: SourceView[];
	bypass_review: boolean;
	model_id: string | null;
	additional_images?: { original_filename: string | null }[];
	id: string;
	original_filename: string | null;
	media_type: string | null;
	size_bytes: number | null;
	image_path: string | null;
	thumbnail_path: string | null;
	status: ImageStatus;
	error: string | null;
	receipt_id: string | null;
	user_id: string | null;
	created_at: string | null;
	analyzed_at: string | null;
	processing_at: string | null;
}

/** List rows omit artifact detail; terminal state is determined from status. */
export type ImageRow = Omit<SubmissionStatus, 'terminal' | 'source_views' | 'latest_extraction_id'>;

export interface UploadOptions {
	allow_ocr_fallback?: boolean;
	model_id?: string;
	bypass_review?: boolean;
}
export interface RetryRequest {
	expected_generation: number;
	request_key: string;
	policy?: 'preserve' | 'current';
}
export interface ReviewRequest {
	expected_operation_revision?: number;
	expected_review_revision?: number;
	acknowledgements: { finding_id: string; comment: string }[];
}
export interface ReviewPreview {
	operation_revision: number;
	review_revision: number;
	findings: ReviewFinding[];
}
export interface ModelInfo {
	id: string;
	parameter_size?: string | null;
	context_window_tokens?: number | null;
}

export interface ReceiptRow {
	operation_revision: number;
	review_revision: number;
	amount_policy: AmountPolicy;
	findings: ReviewFinding[];
	field_sources: Record<string, SourceRef[]>;
	overlap_decisions: Record<string, unknown>[];
	acknowledgements: { finding_id: string; comment: string }[];
	verification_source: string | null;
	checks_version: string | null;
	id: string;
	confidence: number;
	merchant_name: string;
	merchant_address: string | null;
	receipt_number: string | null;
	/** ISO date, YYYY-MM-DD */
	date: string;
	/** Local HH:MM[:SS[.ffffff]], preserving sub-minute precision */
	time: string | null;
	currency: string;
	category: Category;
	subtotal: string;
	discount_total: string | null;
	tax_total: string;
	tip: string | null;
	total: string;
	payment_method: PaymentMethod;
	/** ISO date, YYYY-MM-DD */
	created_at: string | null;
	status: ReceiptStatus;
	image_id: string | null;
	thumbnail_path: string | null;
	verified: boolean;
	user_id: string | null;
}

export interface LineItemRow {
	position?: number;
	id: string;
	receipt_id: string;
	description: string;
	original_description: string | null;
	quantity: string | number | null;
	unit_price: string | null;
	total_price: string;
	spending_category: string;
	category_source: 'llm' | 'history' | 'user' | 'unknown';
	tags: string[];
}

export interface SpendingCategory {
	code: string;
}

export interface CategorySuggestion {
	category: string | null;
	source: 'history' | null;
}

export interface ItemSpendingStatistics {
	months: number;
	date_from: string;
	date_to: string;
	categories: { category: string; currency: string; total: string; share: string | null }[];
	monthly: {
		month: string;
		category: string;
		currency: string;
		total: string;
		share: string | null;
	}[];
	reconciliation: {
		currency: string;
		receipt_total: string;
		item_total: string;
		deposit_total: string;
		unallocated_total: string;
	}[];
}

export interface TaxLineRow {
	id: string;
	receipt_id: string;
	name: string;
	rate: string | number | null;
	amount: string;
}

export interface ReceiptWithDetails {
	receipt: ReceiptRow;
	can_rereview: boolean;
	line_items: LineItemRow[];
	taxes: TaxLineRow[];
	image_path: string | null;
	/** Ids of the collections this receipt belongs to (detail view only). */
	collection_ids: string[];
	classification_enrichment: ClassificationEnrichment;
}

/** Authenticated principal (mirrors the backend `User` model). */
export interface User {
	id: string;
	username: string;
	is_admin: boolean;
	can_see_all: boolean;
}

/** Safe server-side defaults that affect browser workflow. */
export interface UiConfig {
	bypass_review_default: boolean;
	registration_open: boolean;
	ocr_enabled?: boolean;
	ocr_required?: boolean;
	ocr_allow_vision_fallback?: boolean;
	formatter_max_output_tokens?: number;
}

export interface CurrencyStatistics {
	currency: string;
	receipt_count: number;
	total: string;
	average: string;
	median: string;
	minimum: string;
	maximum: string;
	subtotal: string;
	discounts: string;
	taxes: string;
	tips: string;
}

export interface NamedStatistics {
	name: string;
	currency: string;
	receipt_count: number;
	total: string;
	average: string;
}

export interface WeekdayStatistics {
	weekday: number;
	currency: string;
	receipt_count: number;
	total: string;
	average: string;
}

export interface WeeklyStatistics {
	week_start: string;
	currency: string;
	receipt_count: number;
	total: string;
	average: string;
}

export interface ReceiptStatistics {
	verified_receipt_count: number;
	currencies: CurrencyStatistics[];
	merchants: NamedStatistics[];
	categories: NamedStatistics[];
	payment_methods: NamedStatistics[];
	weekdays: WeekdayStatistics[];
	weekly_spending: WeeklyStatistics[];
}

export type SettingSource = 'environment' | 'config' | 'default';

export interface EditableLlmSettings {
	provider: 'ollama' | 'anthropic' | 'openai';
	host: string;
	model_name: string;
	temperature: number;
}

export type LlmAuthenticationUpdate = { mode: 'none' } | { mode: 'bearer'; api_key: string };

export interface LlmAuthenticationView {
	mode: 'none' | 'bearer';
	source: SettingSource | 'settings';
	status: 'none' | 'configured' | 'origin_mismatch';
}

export interface LlmConnectionResult {
	status:
		| 'connected'
		| 'authentication_failed'
		| 'unreachable'
		| 'invalid_response'
		| 'origin_mismatch'
		| 'unsupported_provider';
}

export interface SettingsView {
	llm: EditableLlmSettings;
	allow_registration: boolean;
	sources: Record<string, SettingSource>;
	restart_required: boolean;
	llm_authentication?: LlmAuthenticationView;
}

export interface SettingsUpdate {
	changed_paths?: string[];
	llm: EditableLlmSettings;
	allow_registration: boolean;
	llm_authentication?: LlmAuthenticationUpdate;
}

export interface DedicatedOcrStatus {
	selected?: boolean;
	installed?: boolean;
	shared_gpu?: boolean;
	state:
		| 'unavailable'
		| 'disabled'
		| 'requested'
		| 'installing'
		| 'active'
		| 'failed'
		| 'action_required';
	reason?: string | null;
}

export interface LineItemWrite {
	id?: string;
	description: string;
	quantity: string | number | null;
	unit_price: string | null;
	total_price: string;
	spending_category: string;
	remember_category?: boolean;
	tags?: string[];
}

export interface TaxLineWrite {
	name: string;
	rate?: string | number | null;
	amount: string;
}

/** Body for PUT /receipts/{id} (mirrors the backend `Receipt` write model). */
export interface ReceiptWrite {
	expected_operation_revision: number;
	amount_policy?: AmountPolicy;
	field_sources?: Record<string, SourceRef[]>;
	overlap_decisions?: Record<string, unknown>[];
	confidence: number;
	merchant_name: string;
	merchant_address?: string | null;
	receipt_number?: string | null;
	/** ISO date, YYYY-MM-DD */
	date: string;
	time?: string | null;
	currency?: string;
	category?: Category;
	line_items: LineItemWrite[];
	taxes?: TaxLineWrite[];
	subtotal: string;
	discount_total?: string | null;
	tax_total?: string;
	tip?: string | null;
	total: string;
	payment_method?: PaymentMethod;
	/**
	 * Collection membership on PUT. Omit to leave membership unchanged; send an
	 * explicit empty list to clear all memberships.
	 */
	collection_ids?: string[];
}

/** One result of POST /images/analyze (backend PendingImageResult). */
export interface PendingImageResult {
	image_id: string;
	status: 'analyzed' | 'failed';
	receipt_id?: string | null;
	error?: string | null;
}

export interface AnalyzeResponse {
	results: PendingImageResult[];
}

export interface ReceiptListFilters {
	limit?: number;
	offset?: number;
	status?: ReceiptStatus[];
	/** ISO date, YYYY-MM-DD (inclusive) */
	date_from?: string;
	/** ISO date, YYYY-MM-DD (inclusive) */
	date_to?: string;
	search?: string;
	/** Only receipts in this collection. */
	collection_id?: string;
}

export interface ImageListFilters {
	status?: ImageStatus[];
	limit?: number;
	offset?: number;
}

/** One verified line-item purchase matching a product search. */
export interface ProductPurchase {
	receipt_id: string;
	description: string;
	merchant_name: string;
	/** ISO date, YYYY-MM-DD */
	date: string;
	/** Local HH:MM[:SS[.ffffff]], preserving sub-minute precision */
	time: string | null;
	quantity: string | number | null;
	unit_price: string | null;
	currency: string;
}

/** Response of GET /search: purchase history + unit-price summary. */
export interface ProductSearchResponse {
	query: string;
	purchases: ProductPurchase[];
	latest_price: string | null;
	cheapest_price: string | null;
	average_price: string | null;
	currency: string | null;
}

/** A per-currency total (mirrors backend `CurrencyTotal`). */
export interface CurrencyTotal {
	currency: string;
	total: string;
}

/** A named, per-user group that whole receipts can belong to. */
export interface Collection {
	id: string;
	name: string;
	color: string | null;
	start_date: string | null;
	end_date: string | null;
	active: boolean;
	created_at: string | null;
}

/** A collection with its receipt count and per-currency totals (list view). */
export interface CollectionSummary extends Collection {
	receipt_count: number;
	totals: CurrencyTotal[];
}

/** A collection with its totals and full receipt rows (detail view). */
export interface CollectionDetail extends Collection {
	totals: CurrencyTotal[];
	receipts: ReceiptRow[];
	receipt_count?: number;
}

export interface CollectionCreate {
	name: string;
	color?: string | null;
	start_date?: string | null;
	end_date?: string | null;
	active?: boolean;
}

export interface CollectionUpdate {
	name?: string;
	color?: string | null;
	start_date?: string | null;
	end_date?: string | null;
}

export interface BenchmarkReferenceConfirmation {
	expected_review_revision: number;
	expected_document_sha256: string;
}

export interface BenchmarkReferencePreview {
	receipt_identity: string;
	review_revision: number;
	document_sha256: string;
	expected: Record<string, unknown>;
	scored_fields: string[];
	sources: unknown[];
	findings: unknown[];
}

export interface BenchmarkCapabilities {
	ocr_available: boolean;
	ocr_enabled: boolean;
	thinking_control: boolean;
}

export type BenchmarkCouncilPolicy = 'all' | 'material' | 'custom';

export interface BenchmarkCreate {
	model_ids: string[];
	experiment: {
		pipeline: 'upload' | 'direct_vision' | 'ocr';
		temperature: number | null;
		think: boolean | null;
		include_classification: boolean;
		allow_ocr_fallback: boolean;
	};
	request_timeout_seconds: number;
	council_policy: BenchmarkCouncilPolicy;
	apply_council_flags: boolean;
	receipt_ids?: string[];
	category?: string;
	max_source_confidence?: number;
	limit?: number;
	council_absolute_threshold?: number;
	council_relative_threshold?: number;
}

export interface BenchmarkRun {
	id: string;
	status: string;
	model_ids: string[];
	receipt_ids: string[];
	scoring_version: string;
	experiment?: Record<string, unknown> | null;
}

export interface BenchmarkSummary {
	model_id: string;
	model_digest: string | null;
	completed: number;
	succeeded: number;
	failed: number;
	average_score: number | null;
	all_case_average_score: number | null;
	full_receipt_correct_rate: number | null;
	average_confidence: number | null;
	average_attempts: number | null;
	average_latency_ms: number | null;
	council_candidates: number;
	council_findings: number;
}

export interface BenchmarkStatus {
	run: BenchmarkRun;
	queued: number;
	running: number;
	retrying: number;
	waiting_for_model: number;
	terminal: number;
	summaries: BenchmarkSummary[];
}
