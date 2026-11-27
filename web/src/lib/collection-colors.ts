export const PRESET_COLORS: string[] = [
	'#EF4444', // red
	'#F97316', // orange
	'#F59E0B', // amber
	'#22C55E', // green
	'#14B8A6', // teal
	'#3B82F6', // blue
	'#6366F1', // indigo
	'#A855F7', // purple
	'#EC4899', // pink
	'#64748B' // slate
];

export function colorToStyle(color: string | null): string {
	return `background: ${color ?? 'var(--color-neutral, #64748B)'};`;
}
