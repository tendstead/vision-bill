import { api, API_BASE, request } from './client';
import type { SessionActor } from '$lib/session-lifetime';
import type { ExtractionDraft } from '$lib/types';

export interface BenchmarkCaseSummary {
	id: string;
	name: string;
	revision: number;
	confirmed: boolean;
	photo_count: number;
	owner_id: string;
	created_at: string;
	updated_at: string;
}
export interface BenchmarkCase extends BenchmarkCaseSummary {
	expected: ExtractionDraft;
}
export interface CaseUpdate {
	expected_revision: number;
	name: string;
	expected: ExtractionDraft;
	confirm: boolean;
}
const collection = '/benchmarks/cases';
const path = (id: string) => collection + '/' + encodeURIComponent(id);

export const caseApi = {
	list: (actor: SessionActor) => request<BenchmarkCaseSummary[]>(collection, undefined, actor),
	get: (id: string, actor: SessionActor) => request<BenchmarkCase>(path(id), undefined, actor),
	create(name: string, photos: File[], actor: SessionActor) {
		const form = new FormData();
		form.append('name', name);
		for (const photo of photos) form.append('photos', photo);
		return request<BenchmarkCase>(collection, { method: 'POST', body: form }, actor);
	},
	save(id: string, update: CaseUpdate, actor: SessionActor) {
		return request<BenchmarkCase>(
			path(id),
			{
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(update)
			},
			actor
		);
	},
	photoUrl(id: string, index: number, actor: SessionActor) {
		const url = new URL(API_BASE + path(id) + '/photos/' + index, window.location.origin);
		if (url.origin !== window.location.origin) return undefined;
		return api.sourceViewUrl(url.pathname, actor);
	}
};
