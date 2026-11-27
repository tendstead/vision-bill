import { translate } from '$lib/i18n';

/**
 * Format an ISO timestamp as a short relative label ("just now", "5 min ago",
 * …). Returns an empty string for missing/invalid values.
 */
export function formatRelativeTime(iso: string | null | undefined): string {
	if (!iso) {
		return '';
	}
	const then = new Date(iso).getTime();
	if (Number.isNaN(then)) {
		return '';
	}
	const diffSec = Math.max(0, (Date.now() - then) / 1000);
	if (diffSec < 90) {
		return translate('time.justNow');
	}
	const minutes = Math.round(diffSec / 60);
	if (minutes < 60) {
		return translate('time.minutesAgo', { values: { count: minutes } });
	}
	const hours = Math.round(minutes / 60);
	if (hours < 24) {
		return translate('time.hoursAgo', { values: { count: hours } });
	}
	const days = Math.round(hours / 24);
	return translate('time.daysAgo', { values: { count: days } });
}

/** Format an ISO timestamp as a locale date ("Aug 30, 2026"). */
export function formatDate(iso: string | null | undefined): string {
	if (!iso) {
		return '';
	}
	const then = new Date(iso);
	if (Number.isNaN(then.getTime())) {
		return '';
	}
	try {
		return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(then);
	} catch {
		return iso;
	}
}
