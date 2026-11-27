import type { ImageListFilters, ReceiptListFilters } from '$lib/types';

export const queryKeys = {
	images: (filters: ImageListFilters = {}) => ['images', filters] as const,
	image: (id: string) => ['images', id] as const,
	receipts: (filters: ReceiptListFilters = {}) => ['receipts', filters] as const,
	receipt: (id: string) => ['receipts', id] as const,
	search: (term: string) => ['search', term] as const,
	tags: () => ['tags'] as const,
	spendingCategories: () => ['tags', 'categories'] as const,
	collections: () => ['collections'] as const,
	collection: (id: string) => ['collections', id] as const,
	collectionsActive: () => ['collections', 'active'] as const,
	uiConfig: () => ['system', 'ui-config'] as const,
	statistics: (weeks = 12, collectionId = '') => ['statistics', weeks, collectionId] as const,
	itemStatistics: (months = 12, collectionId = '') =>
		['statistics', 'items', months, collectionId] as const,
	settings: () => ['system', 'settings'] as const
};
