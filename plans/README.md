# Roadmap plans

One plan per task, each meant to become its own branch and PR. Order and
status:

| # | Plan | Status |
| - | ---- | ------ |
| 1 | [Comparable benchmarks](01-benchmark-compare.md) | model-major ordering done (`feat/benchmark-compare`); rest planned |
| 2 | [Stale and failing queued images](02-stale-images.md) | planned |
| 3 | [Optional integrations (shopping list, roadbook)](03-integrations.md) | planned, blocked on the other services |
| 4 | [Cherry-picks from PR #16 (OCR and more)](04-pr16-cherry-picks.md) | inventory done, picks to be decided |

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

## Credit: work from PR #16

PR #16 (David Hupperich, 188 commits; GitHub identities `David Hupperich
<d.hupperich@web.de>` and `Davidoo`) already contains work that overlaps with
plans 1-3. These plans stay the priority, but every place where PR #16 has the
same idea is marked **PR #16 overlap** (with the files to look at) so nothing is
re-invented or forgotten.

Whenever code, schema, tests, wording of a contract or a design from PR #16 is
ported, adapted or used as the model for a change:

1. Add `Co-authored-by: David Hupperich <d.hupperich@web.de>` to the commit
   message of every commit that contains it.
2. Name him and the PR #16 files in the PR description.
3. If a whole feature is ported, mention it in the release notes.

PR #16 is based on the Python backend. `main` is the only source of truth for
the database schema: its Alembic revisions `0011`-`0025` are not adopted
(David uses a test server only); ported features get new additive Go migrations
(see plan 4).
