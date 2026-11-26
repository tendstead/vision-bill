import {
	captureSessionActor,
	currentSessionGeneration,
	isCurrentSessionActor,
	requireCurrentSessionActor,
	requireSessionGeneration,
	type SessionActor
} from '$lib/session-lifetime';
import type { IndividualResult } from '$lib/benchmark-experiments';
import type {
	AnalyzeResponse,
	BenchmarkCapabilities,
	BenchmarkCreate,
	BenchmarkReferenceConfirmation,
	BenchmarkReferencePreview,
	BenchmarkRun,
	BenchmarkStatus,
	DraftView,
	DraftPatch,
	DraftFinalize,
	DedicatedOcrStatus,
	CategorySuggestion,
	Collection,
	CollectionCreate,
	CollectionDetail,
	CollectionSummary,
	CollectionUpdate,
	SubmissionStatus,
	UploadOptions,
	RetryRequest,
	ReviewPreview,
	ReviewRequest,
	ModelInfo,
	LlmConnectionResult,
	ImageListFilters,
	ImageRow,
	ItemSpendingStatistics,
	ProductSearchResponse,
	ReceiptListFilters,
	ReceiptRow,
	ReceiptStatistics,
	ReceiptWithDetails,
	ReceiptWrite,
	SettingsUpdate,
	SettingsView,
	SpendingCategory,
	User,
	UiConfig
} from '$lib/types';

export const API_BASE = (import.meta.env.VITE_API_BASE ?? '/api/v1').replace(/\/$/, '');

/**
 * Handler invoked on any 401 from a protected (non-`/auth/`) endpoint. Wired up
 * by `$lib/auth` to drop the session and redirect to the login page. Kept as a
 * swappable callback so this module never imports the session store (no cycle).
 */
let unauthorizedHandler: (() => void) | null = null;
let sessionActorChangedHandler: (() => void) | null = null;

export function onUnauthorized(handler: () => void): void {
	unauthorizedHandler = handler;
}

export function onSessionActorChanged(handler: () => void): void {
	sessionActorChangedHandler = handler;
}

export class ApiError extends Error {
	status: number;

	detail: string;

	constructor(
		status: number,
		detail: string,
		public imageId?: string,
		public problem?: { code: string; fields: string[] }
	) {
		super(detail);
		this.name = 'ApiError';
		this.status = status;
		this.detail = detail;
	}
}

export async function request<T>(
	path: string,
	init?: RequestInit,
	actor = captureSessionActor()
): Promise<T> {
	const isProtected = !path.startsWith('/auth/') && path !== '/system/ui-config';
	const generation = actor?.generation ?? currentSessionGeneration();
	function requireCurrent() {
		requireSessionGeneration(generation);
		if (isProtected) requireCurrentSessionActor(actor);
	}
	requireCurrent();

	const options: RequestInit = { credentials: 'same-origin', ...init };
	if (isProtected && actor) {
		const headers = new Headers(options.headers);
		headers.set('X-Vision-Bill-Expected-User', actor.userId);
		options.headers = headers;
	}
	if (isProtected || path === '/auth/me') options.cache = 'no-store';

	let response: Response;
	try {
		response = await fetch(`${API_BASE}${path}`, options);
	} catch {
		requireCurrent();
		throw new ApiError(0, 'Network error');
	}

	requireCurrent();
	if (!response.ok) {
		// A 401 on a protected endpoint means the session expired: bounce to
		// login. The /auth/ endpoints 401 for bad credentials, not an expired
		// session, so they are excluded.
		if (response.status === 401 && isProtected) {
			unauthorizedHandler?.();
		}
		let detail = `${response.status} ${response.statusText}`;
		let imageId: string | undefined;
		let problem: ApiError['problem'];
		try {
			const body: unknown = await response.json();
			if (
				body &&
				typeof body === 'object' &&
				'image_id' in body &&
				typeof body.image_id === 'string'
			) {
				imageId = body.image_id;
			}
			if (body && typeof body === 'object' && 'detail' in body) {
				const raw = (body as { detail: unknown }).detail;
				detail = typeof raw === 'string' ? raw : JSON.stringify(raw);
				if (raw && typeof raw === 'object' && 'code' in raw && typeof raw.code === 'string') {
					problem = {
						code: raw.code,
						fields:
							'fields' in raw && Array.isArray(raw.fields)
								? raw.fields.filter((path): path is string => typeof path === 'string')
								: []
					};
				}
			}
		} catch {
			// Non-JSON error body: keep the status text.
		}
		requireCurrent();
		if (isProtected && response.status === 409 && detail === 'session_actor_changed') {
			sessionActorChangedHandler?.();
			requireCurrent();
		}
		throw new ApiError(response.status, detail, imageId, problem);
	}

	if (response.status === 204) {
		return undefined as T;
	}

	try {
		const body = (await response.json()) as T;
		requireCurrent();
		return body;
	} catch {
		requireCurrent();
		// A successful status with an unreadable body is still an ambiguous mutation.
		throw new ApiError(response.status, 'Response body could not be read');
	}
}

function toQueryString(
	params: Record<string, string | number | undefined>,
	preserveEmpty = false
): string {
	const parts: string[] = [];
	for (const [key, value] of Object.entries(params)) {
		if (value === undefined || (value === '' && !preserveEmpty)) {
			continue;
		}
		parts.push(`${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`);
	}
	return parts.length > 0 ? `?${parts.join('&')}` : '';
}

function privateMediaUrl(path: string, actor: SessionActor | null): string | undefined {
	if (!isCurrentSessionActor(actor)) return undefined;
	return `${path}${path.includes('?') ? '&' : '?'}vb_expected_user=${encodeURIComponent(actor.userId)}`;
}

export const api = {
	/** Files are appended in capture order; absent options stay absent. */
	async uploadImages(
		files: File[],
		options: UploadOptions,
		idempotencyKey: string,
		actor = captureSessionActor()
	): Promise<SubmissionStatus> {
		const form = new FormData();
		for (const image of files) form.append('receipt', image);
		return request<SubmissionStatus>(
			`/images${toQueryString(
				{
					model_id: options.model_id,
					allow_ocr_fallback:
						options.allow_ocr_fallback === undefined
							? undefined
							: String(options.allow_ocr_fallback),
					bypass_review:
						options.bypass_review === undefined ? undefined : String(options.bypass_review)
				},
				true
			)}`,
			{ method: 'POST', headers: { 'Idempotency-Key': idempotencyKey }, body: form },
			actor
		);
	},

	retrySubmission(id: string, body: RetryRequest): Promise<SubmissionStatus> {
		return request<SubmissionStatus>(`/images/${id}/retry`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body)
		});
	},

	listModels(
		capability: 'vision' | 'text' = 'vision',
		actor = captureSessionActor()
	): Promise<ModelInfo[]> {
		return request<ModelInfo[]>(
			`/llm/models${capability === 'text' ? '?capability=text' : ''}`,
			undefined,
			actor
		);
	},

	getBenchmarkCapabilities(actor: SessionActor): Promise<BenchmarkCapabilities> {
		return request<BenchmarkCapabilities>('/benchmarks/capabilities', undefined, actor);
	},

	previewBenchmarkReference(id: string, actor: SessionActor): Promise<BenchmarkReferencePreview> {
		return request<BenchmarkReferencePreview>(
			`/benchmarks/references/${encodeURIComponent(id)}`,
			undefined,
			actor
		);
	},

	qualifyBenchmarkReference(
		id: string,
		body: BenchmarkReferenceConfirmation,
		actor: SessionActor
	): Promise<unknown> {
		return request(
			`/benchmarks/references/${encodeURIComponent(id)}`,
			{
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(body)
			},
			actor
		);
	},

	createBenchmark(body: BenchmarkCreate, actor: SessionActor): Promise<BenchmarkRun> {
		return request<BenchmarkRun>(
			'/benchmarks',
			{
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(body)
			},
			actor
		);
	},

	listBenchmarks(
		actor: SessionActor,
		filters?: { comparison_id?: string; offset?: number }
	): Promise<BenchmarkRun[]> {
		return request<BenchmarkRun[]>('/benchmarks' + toQueryString(filters ?? {}), undefined, actor);
	},

	getBenchmarkStatus(id: string, actor: SessionActor): Promise<BenchmarkStatus> {
		return request<BenchmarkStatus>(`/benchmarks/${encodeURIComponent(id)}`, undefined, actor);
	},

	getBenchmarkResults(
		id: string,
		offset: number,
		actor: SessionActor,
		selection?: { model: string; cases: string[] }
	): Promise<IndividualResult[]> {
		const query = new URLSearchParams({ limit: '50', offset: String(offset) });
		if (selection) {
			query.set('model_id', selection.model);
			for (const id of selection.cases) query.append('receipt_id', id);
		}
		return request<IndividualResult[]>(
			`/benchmarks/${encodeURIComponent(id)}/results?${query}`,
			undefined,
			actor
		);
	},

	getUiConfig(): Promise<UiConfig> {
		return request<UiConfig>('/system/ui-config');
	},

	getStatistics(weeks = 12, collectionId?: string): Promise<ReceiptStatistics> {
		return request<ReceiptStatistics>(
			`/statistics${toQueryString({ weeks, collection_id: collectionId })}`
		);
	},

	getItemStatistics(months = 12, collectionId?: string): Promise<ItemSpendingStatistics> {
		return request<ItemSpendingStatistics>(
			`/statistics/items${toQueryString({ months, collection_id: collectionId })}`
		);
	},

	searchProducts(query: string): Promise<ProductSearchResponse> {
		return request<ProductSearchResponse>(`/search${toQueryString({ query })}`);
	},

	getSettings(): Promise<SettingsView> {
		return request<SettingsView>('/system/settings');
	},

	getDedicatedOcr(actor: SessionActor | null): Promise<DedicatedOcrStatus> {
		return request<DedicatedOcrStatus>('/system/dedicated-ocr', undefined, actor);
	},

	setDedicatedOcrGpuTopology(
		actor: SessionActor | null,
		sharedGpu: boolean,
		confirmHandoff: boolean
	): Promise<DedicatedOcrStatus> {
		return request<DedicatedOcrStatus>(
			'/system/dedicated-ocr/gpu-topology',
			{
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ shared_gpu: sharedGpu, confirm_handoff: confirmHandoff })
			},
			actor
		);
	},

	requestDedicatedOcr(
		actor: SessionActor | null,
		confirmHandoff: boolean,
		enabled = true
	): Promise<DedicatedOcrStatus> {
		return request<DedicatedOcrStatus>(
			'/system/dedicated-ocr',
			{
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ enabled, confirm_handoff: confirmHandoff })
			},
			actor
		);
	},

	updateSettings(body: SettingsUpdate): Promise<SettingsView> {
		return request<SettingsView>('/system/settings', {
			method: 'PUT',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				llm: body.llm,
				allow_registration: body.allow_registration,
				changed_paths: body.changed_paths,
				llm_authentication: body.llm_authentication
			})
		});
	},

	testLlmConnection(
		host: string,
		provider: SettingsUpdate['llm']['provider']
	): Promise<LlmConnectionResult> {
		return request<LlmConnectionResult>('/system/settings/llm/connection-test', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ host, provider })
		});
	},

	/** GET /auth/me returns the current user (401 when the session is gone). */
	me(): Promise<User> {
		return request<User>('/auth/me');
	},

	/** POST /auth/login authenticates and sets the session cookie. */
	login(username: string, password: string): Promise<User> {
		return request<User>('/auth/login', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ username, password })
		});
	},

	/** POST /auth/register creates an account (403 if registration is disabled). */
	register(username: string, password: string): Promise<User> {
		return request<User>('/auth/register', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ username, password })
		});
	},

	/** POST /auth/logout clears the session cookie. */
	logout(expectedUserId: string): Promise<void> {
		return request<void>('/auth/logout', {
			method: 'POST',
			headers: { 'X-Vision-Bill-Expected-User': expectedUserId }
		});
	},

	listImages(filters: ImageListFilters = {}): Promise<ImageRow[]> {
		return request<ImageRow[]>(
			`/images${toQueryString({
				status: filters.status?.join(','),
				limit: filters.limit,
				offset: filters.offset
			})}`
		);
	},

	getImage(id: string): Promise<SubmissionStatus> {
		return request<SubmissionStatus>(`/images/${id}`);
	},

	deleteImage(id: string): Promise<void> {
		return request<void>(`/images/${id}`, { method: 'DELETE' });
	},

	analyzePending(): Promise<AnalyzeResponse> {
		return request<AnalyzeResponse>('/images/analyze', { method: 'POST' });
	},

	/** Private URLs for image tags and direct navigation. */
	imageFileUrl(id: string, part = 0, actor = captureSessionActor()): string | undefined {
		return privateMediaUrl(`${API_BASE}/images/${id}/file${part ? `?part=${part}` : ''}`, actor);
	},

	imageThumbUrl(id: string, actor = captureSessionActor()): string | undefined {
		return privateMediaUrl(`${API_BASE}/images/${id}/thumb`, actor);
	},

	sourceViewUrl(url: string, actor = captureSessionActor()): string | undefined {
		return privateMediaUrl(url, actor);
	},

	listReceipts(
		filters: ReceiptListFilters = {},
		actor = captureSessionActor()
	): Promise<ReceiptRow[]> {
		return request<ReceiptRow[]>(
			`/receipts${toQueryString({
				limit: filters.limit,
				offset: filters.offset,
				status: filters.status?.join(','),
				date_from: filters.date_from,
				date_to: filters.date_to,
				search: filters.search,
				collection_id: filters.collection_id
			})}`,
			undefined,
			actor
		);
	},

	getReceipt(id: string): Promise<ReceiptWithDetails> {
		return request<ReceiptWithDetails>(`/receipts/${id}`);
	},

	/** PUT /receipts/{id} returns the updated receipt row (without line items). */
	getDraft(id: string): Promise<DraftView> {
		return request<DraftView>(`/images/${id}/draft`);
	},
	patchDraft(id: string, body: DraftPatch): Promise<DraftView> {
		return request<DraftView>(`/images/${id}/draft`, {
			method: 'PATCH',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body)
		});
	},
	finalizeDraft(id: string, body: DraftFinalize): Promise<ReceiptRow> {
		return request<ReceiptRow>(`/images/${id}/draft/finalize`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body)
		});
	},

	updateReceipt(id: string, body: ReceiptWrite): Promise<ReceiptRow> {
		return request<ReceiptRow>(`/receipts/${id}`, {
			method: 'PUT',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body)
		});
	},

	previewHumanReview(id: string): Promise<ReviewPreview> {
		return request<ReviewPreview>(`/receipts/${id}/verify`);
	},

	/** Verify the receipt or record a human review of an automatically or historically verified receipt. */
	verifyReceipt(id: string, body?: ReviewRequest): Promise<ReceiptRow> {
		return request<ReceiptRow>(`/receipts/${id}/verify`, {
			method: 'POST',
			...(body
				? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
				: {})
		});
	},

	/** DELETE /receipts/{id} removes the receipt (and its image). 409 if a benchmark references it. */
	deleteReceipt(id: string): Promise<void> {
		return request<void>(`/receipts/${id}`, { method: 'DELETE' });
	},

	listMerchants(
		params: {
			merchant_name?: string;
			merchant_address?: string;
			q?: string;
			receipt_id?: string;
			image_id?: string;
		},
		actor = captureSessionActor()
	) {
		return request<{ suggestion: string | null; merchants: string[] }>(
			'/merchants/suggestions',
			{
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(params)
			},
			actor
		);
	},

	/** GET /tags returns the line-item tag vocabulary (the select source). */
	listTags(): Promise<string[]> {
		return request<string[]>('/tags');
	},

	listSpendingCategories(): Promise<SpendingCategory[]> {
		return request<SpendingCategory[]>('/tags/categories');
	},

	getCategorySuggestion(params: {
		merchant_name: string;
		merchant_address?: string;
		merchant_display_name?: string;
		receipt_id: string;
		description: string;
	}): Promise<CategorySuggestion> {
		return request<CategorySuggestion>('/tags/category-suggestion', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(params)
		});
	},

	/**
	 * POST /tags creates (or confirms) a tag. Idempotent: an existing tag is
	 * returned as-is. Returns the normalized name actually stored.
	 */
	createTag(name: string): Promise<{ name: string; created: boolean }> {
		return request<{ name: string; created: boolean }>('/tags', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ name })
		});
	},

	listCollections(): Promise<CollectionSummary[]> {
		return request<CollectionSummary[]>('/collections');
	},
	getCollection(id: string): Promise<CollectionDetail> {
		return request<CollectionDetail>(`/collections/${id}`);
	},
	/** The caller's active (auto-capturing) collection, or null when none (204). */
	async getActiveCollection(): Promise<Collection | null> {
		return (await request<Collection>('/collections/active')) ?? null;
	},
	createCollection(body: CollectionCreate): Promise<Collection> {
		return request<Collection>('/collections', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body)
		});
	},
	updateCollection(id: string, body: CollectionUpdate): Promise<Collection> {
		return request<Collection>(`/collections/${id}`, {
			method: 'PATCH',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(body)
		});
	},
	deleteCollection(id: string): Promise<void> {
		return request<void>(`/collections/${id}`, { method: 'DELETE' });
	},
	assignReceiptToCollection(
		collectionId: string,
		receiptId: string
	): Promise<{ assigned: boolean }> {
		return request<{ assigned: boolean }>(`/collections/${collectionId}/receipts/${receiptId}`, {
			method: 'POST'
		});
	},
	unassignReceiptFromCollection(collectionId: string, receiptId: string): Promise<void> {
		return request<void>(`/collections/${collectionId}/receipts/${receiptId}`, {
			method: 'DELETE'
		});
	},
	activateCollection(id: string): Promise<Collection> {
		return request<Collection>(`/collections/${id}/activate`, { method: 'POST' });
	},
	deactivateCollection(id: string): Promise<Collection> {
		return request<Collection>(`/collections/${id}/deactivate`, { method: 'POST' });
	}
};
