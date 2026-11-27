# Roadmap plans

One plan per task, each meant to become its own branch and PR. Order and
status:

| # | Plan | Status |
| - | ---- | ------ |
| 1 | [Comparable benchmarks](01-benchmark-compare.md) | model-major ordering done (`feat/benchmark-compare`); rest planned |
| 2 | [Stale and failing queued images](02-stale-images.md) | planned |
| 3 | [Optional integrations (shopping list, roadbook)](03-integrations.md) | planned, blocked on the other services |
| 4 | [Cherry-picks from PR #16 (OCR)](04-pr16-cherry-picks.md) | to be scoped in its own session |

## Rules for every task

- **Branch and PR per task.** Do not mix tasks; do not start a task before its
  open questions are answered.
- **Additive migrations only** (new nullable/defaulted columns, new tables).
  The `alembic_version` marker stays at `0010`, so anything else breaks the
  rollback described in [UPGRADING.md](../UPGRADING.md). A non-additive
  migration needs an explicit decision and a release note.
- **Optional means off by default.** Integrations and other connections do
  nothing and make no outbound calls until configured (privacy rule in
  AGENTS.md: receipt data goes only to the configured inference endpoint).
- **Tests with the change:** Go tests with a real PostgreSQL, the API e2e suite
  when a contract changes, Playwright when the UI changes. `make lint`,
  `make test`, `make test-e2e`, `make fe-verify` must pass before the PR.
- **Contract changes** update `internal/httpapi/openapi.json` in the same PR.
- **Releases:** the project is still in the 0.x phase. Releases are minor bumps
  (`0.2.0` for the Go port); fixes may go out as patches. Pick `1.0.0` when the
  project is considered stable.
