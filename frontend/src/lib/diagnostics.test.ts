import { describe, expect, it } from 'vitest';
import { diagnosticLabel } from './diagnostics';
import { setLocale, translate } from './i18n';

describe('current safe processing reasons', () => {
	it.each(['en', 'de'] as const)(
		'explains readiness, failed generation and exhausted attempts in %s',
		(language) => {
			setLocale(language);
			for (const code of [
				'ocr_startup_timeout',
				'ocr_qualification_invalidated',
				'ocr_worker_unstoppable',
				'ocr_invalid_result',
				'ocr_deadline_exceeded',
				'ocr_queue_full',
				'ocr_legacy_replay_only',
				'attempt_limit',
				'execution_snapshot_invalid'
			]) {
				const message = diagnosticLabel(translate, 'reasons', code);
				expect(message).not.toContain(code.replaceAll('_', ' '));
				expect(message).not.toContain(code);
				expect(message.length).toBeGreaterThan(30);
			}
			const unknown = diagnosticLabel(translate, 'reasons', 'future_safe_reason');
			expect(unknown).toContain('future safe reason');
			setLocale('en');
		}
	);
});
