import { api } from '$lib/api/client';
import { captureSessionActor } from '$lib/session-lifetime';

/**
 * Decides what to render when a receipt image has no usable thumbnail.
 *
 * Currently: fall back to the full-size image, loaded lazily. A future user
 * setting can change this to return the literal string "icon" (render the
 * default receipt icon instead) without touching the backend.
 */
export function missingThumbSource(
	imageId: string,
	actor = captureSessionActor()
): string | 'icon' {
	return api.imageFileUrl(imageId, 0, actor) ?? 'icon';
}
