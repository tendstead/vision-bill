# Upgrading and rolling back

## Switching from the Python image to the Go image

The Go image is a drop-in replacement: same environment variables, port, user
and mounts. Back up the PostgreSQL database, then change the image tag. On
start the server migrates the schema itself (no separate step).

An existing database is **adopted** only when the legacy `alembic_version`
marker says exactly `0010` and all expected columns exist. Anything else
(older or newer revision, tables without a marker) is refused at start with an
explanatory error; bring the database to `0010` with the previous release first.

## How migrations are tracked

- The Go server records applied migrations (`internal/store/migrations/*.sql`)
  in the table `vision_bill_go_migrations`.
- `001_baseline.sql` is the schema at legacy revision `0010`. Fresh installs
  and adopted databases both get the legacy `alembic_version` table set to
  `0010`, so the previous release still recognises the database.
- **The `alembic_version` marker is frozen at `0010`.** Go migrations
  (`002_*.sql` and later) never touch it.

## Rolling back

Re-pin the previous image tag. This is safe as long as every Go migration since
`0010` is **additive** (new nullable or defaulted columns, new tables, new
indexes). The previous release does not know the new objects, but ignores them.
Currently that holds for:

| Migration | Change |
| --- | --- |
| `002_image_retry_backoff.sql` | `images.analysis_attempts` (default 0), `images.retry_after` |

Consequences to be aware of:

- The previous release treats `timed_out` images as final. After a rollback they
  are no longer retried automatically; upload them again.
- **A migration that is not additive (drops, renames, type changes, tightened
  constraints) breaks the rollback.** Because the marker still says `0010`, the
  previous release would start against a schema it does not understand without
  any warning. Such a migration must be called out in the release notes, and a
  database backup is then the only way back.

## Running an older Go image on a newer database

The Go migrator ignores ledger entries it does not know. Starting an *older* Go
image against a database that a newer Go image already migrated therefore
succeeds silently; it only works while the newer migrations were additive (see
above). Do not rely on it for anything but a short-lived rollback.
