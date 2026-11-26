import type { ImageRow } from '$lib/types';
import en from '$lib/i18n/en.json';
import type { translate } from '$lib/i18n';

export function diagnosticLabel(
	t: (...args: Parameters<typeof translate>) => ReturnType<typeof translate>,
	group: 'checks' | 'reasons' | 'severity' | 'comparison',
	code: string
): string {
	if (Object.hasOwn(en.diagnostics[group], code)) return t(`diagnostics.${group}.${code}`);
	// Saved reviews retain informational provider/fallback reasons alongside
	// financial checks. Both are known findings, with different dictionaries.
	if (group === 'checks' && Object.hasOwn(en.diagnostics.reasons, code))
		return t(`diagnostics.reasons.${code}`);
	return t('diagnostics.unknown', { values: { label: code.replace(/[_-]+/g, ' ') } });
}

export function hasPreviousAttemptResult(image: Pick<ImageRow, 'status' | 'attempt_count'>) {
	return (image.status === 'pending' || image.status === 'processing') && image.attempt_count > 0;
}
