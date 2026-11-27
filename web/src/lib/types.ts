export type ImageStatus =
	'pending' | 'processing' | 'analyzed' | 'failed' | 'timed_out' | 'unreadable';

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

export interface ImageRow {
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

export interface ReceiptRow {
	id: string;
	confidence: number;
	merchant_name: string;
	merchant_address: string | null;
	receipt_number: string | null;
	/** ISO date, YYYY-MM-DD */
	date: string;
	/** HH:MM 24h */
	time: string | null;
	currency: string;
	category: Category;
	subtotal: string;
	discount_total: string;
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
	id: string;
	receipt_id: string;
	description: string;
	original_description: string | null;
	quantity: number;
	unit_price: string;
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
	rate: number | null;
	amount: string;
}

export interface ReceiptWithDetails {
	receipt: ReceiptRow;
	line_items: LineItemRow[];
	taxes: TaxLineRow[];
	image_path: string | null;
	/** Ids of the collections this receipt belongs to (detail view only). */
	collection_ids: string[];
}

/** Body for POST /images (file uploaded via FormData, model_id as query param). */
export interface ImageCreated {
	image_id: string;
	status: ImageStatus;
	receipt_id?: string | null;
	warning?: string;
	original_filename?: string;
	media_type?: string;
	size_bytes?: number;
	image_path?: string;
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

export interface SettingsView {
	llm: EditableLlmSettings;
	allow_registration: boolean;
	sources: Record<string, SettingSource>;
	restart_required: boolean;
}

export interface SettingsUpdate {
	llm: EditableLlmSettings;
	allow_registration: boolean;
}

export interface LineItemWrite {
	id?: string;
	description: string;
	quantity: number;
	unit_price: string;
	total_price: string;
	spending_category: string;
	remember_category?: boolean;
	tags?: string[];
}

export interface TaxLineWrite {
	name: string;
	rate?: number | null;
	amount: string;
}

/** Body for PUT /receipts/{id} (mirrors the backend `Receipt` write model). */
export interface ReceiptWrite {
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
	discount_total?: string;
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
	/** HH:MM 24h */
	time: string | null;
	quantity: number;
	unit_price: string;
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
