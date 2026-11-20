/** Playwright runs in Node; keep the browser-oriented test TS project dependency-free. */
export function testEnvironment(name: string): string {
	const runtime = globalThis as typeof globalThis & {
		process?: { env: Record<string, string | undefined> };
	};
	const value = runtime.process?.env[name];
	if (!value) throw new Error(`${name} is required for the owned E2E stack`);
	return value;
}
