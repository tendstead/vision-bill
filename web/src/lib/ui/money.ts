/**
 * Display-only decimal helpers.
 *
 * Amounts travel as strings (backend `Decimal`), so these helpers use exact
 * integer (BigInt) math instead of floats to avoid drift when rendering
 * computed totals. They are intentionally not used for values that are sent
 * back to the API.
 */

const INT_RE = /^\d+$/;
const DEC_RE = /^\d+\.\d+$/;
const SIGNED_RE = /^-?\d+(\.\d+)?$/;

/** True when the value is a plain (possibly negative-free) decimal string. */
export function isPlainDecimal(value: string | null | undefined): boolean {
	if (value === null || value === undefined) {
		return false;
	}
	const s = value.trim();
	return INT_RE.test(s) || DEC_RE.test(s);
}

/** True when the value is a decimal string that may be negative (e.g. "-0.75"). */
export function isSignedDecimal(value: string | null | undefined): boolean {
	if (value === null || value === undefined) {
		return false;
	}
	return SIGNED_RE.test(value.trim());
}

interface Scaled {
	int: bigint;
	digits: number;
}

function toScaled(value: string): Scaled {
	const [intPart, fracPart = ''] = value.trim().split('.');
	return {
		int: BigInt(intPart || '0') * 10n ** BigInt(fracPart.length),
		digits: fracPart.length
	};
}

function formatScaled(int: bigint, digits: number): string {
	const sign = int < 0n ? '-' : '';
	const abs = int < 0n ? -int : int;
	const base = 10n ** BigInt(digits);
	const intPart = abs / base;
	const fracRaw = digits > 0 ? (abs % base).toString().padStart(digits, '0') : '';
	const frac = fracRaw.replace(/0+$/, '');
	return `${sign}${intPart.toString()}${frac.length > 0 ? `.${frac}` : ''}`;
}

/** Exact multiplication of a quantity (number) by a decimal unit price string. */
export function decMul(quantity: number, unitPrice: string): string {
	if (!Number.isFinite(quantity) || quantity < 0) {
		throw new Error(`Invalid quantity: ${quantity}`);
	}
	const pq = toScaled(String(quantity));
	const pu = toScaled(unitPrice);
	return formatScaled(pq.int * pu.int, pq.digits + pu.digits);
}

/**
 * Format an amount for display, e.g. formatMoney("12.34", "EUR") -> "12,34 €".
 * Returns an empty string for missing/invalid values.
 */
export function formatMoney(
	value: string | number | null | undefined,
	currency: string,
	locale?: string
): string {
	if (value === null || value === undefined || value === '') {
		return '';
	}
	const num = typeof value === 'number' ? value : Number(value);
	if (!Number.isFinite(num)) {
		return '';
	}
	const code = /^[A-Za-z]{3}$/.test(currency) ? currency : 'USD';
	try {
		return new Intl.NumberFormat(locale, { style: 'currency', currency: code }).format(num);
	} catch {
		return `${num.toFixed(2)} ${code}`;
	}
}
