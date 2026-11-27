// Package store owns PostgreSQL access: the connection pool, embedded SQL
// migrations and (in sibling files) explicit-SQL queries.
package store

import (
	"context"
	"embed"
	"errors"
	"fmt"
	"io/fs"
	"sort"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

//go:embed migrations/*.sql
var migrationFS embed.FS

//go:embed schema_columns.txt
var expectedColumns string

const (
	// LegacyHead is the only revision of the legacy migration marker
	// (alembic_version) from which an existing database is adopted.
	LegacyHead  = "0010"
	baseline    = "001_baseline"
	ledgerTable = "vision_bill_go_migrations"
	// advisoryLockKey serialises migrations across replicas.
	advisoryLockKey int64 = 0x76624d6967 // "vbMig"
)

// Store wraps the pgx pool.
type Store struct {
	Pool *pgxpool.Pool
}

// Open connects to PostgreSQL. It does not run migrations.
func Open(ctx context.Context, dsn string) (*Store, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, err
	}
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, err
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	return &Store{Pool: pool}, nil
}

// Close releases the pool.
func (s *Store) Close() {
	if s != nil && s.Pool != nil {
		s.Pool.Close()
	}
}

type migration struct{ version, sql string }

func loadMigrations() ([]migration, error) {
	entries, err := fs.ReadDir(migrationFS, "migrations")
	if err != nil {
		return nil, err
	}
	var out []migration
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".sql") {
			continue
		}
		b, err := fs.ReadFile(migrationFS, "migrations/"+e.Name())
		if err != nil {
			return nil, err
		}
		out = append(out, migration{strings.TrimSuffix(e.Name(), ".sql"), string(b)})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].version < out[j].version })
	return out, nil
}

// LatestVersion is the newest embedded migration.
func LatestVersion() string {
	ms, err := loadMigrations()
	if err != nil || len(ms) == 0 {
		return ""
	}
	return ms[len(ms)-1].version
}

// Migrate brings the database to the latest embedded migration inside one
// transaction guarded by an advisory lock. An existing database carrying the
// legacy migration marker is adopted when it is exactly at LegacyHead and its
// schema matches.
func Migrate(ctx context.Context, pool *pgxpool.Pool) error {
	ms, err := loadMigrations()
	if err != nil {
		return err
	}
	tx, err := pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if _, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock($1)", advisoryLockKey); err != nil {
		return err
	}
	var ledger *string
	if err := tx.QueryRow(ctx, "SELECT to_regclass($1)::text", ledgerTable).Scan(&ledger); err != nil {
		return err
	}
	if ledger == nil {
		if _, err := tx.Exec(ctx, "CREATE TABLE "+ledgerTable+" (version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"); err != nil {
			return fmt.Errorf("create migration ledger: %w", err)
		}
	}
	applied := map[string]bool{}
	rows, err := tx.Query(ctx, "SELECT version FROM "+ledgerTable)
	if err != nil {
		return err
	}
	for rows.Next() {
		var v string
		if err := rows.Scan(&v); err != nil {
			rows.Close()
			return err
		}
		applied[v] = true
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}
	for _, m := range ms {
		if applied[m.version] {
			continue
		}
		if m.version == baseline {
			if err := installBaseline(ctx, tx, m); err != nil {
				return err
			}
		} else if _, err := tx.Exec(ctx, m.sql); err != nil {
			return fmt.Errorf("migration %s: %w", m.version, err)
		}
		if _, err := tx.Exec(ctx, "INSERT INTO "+ledgerTable+" (version) VALUES ($1)", m.version); err != nil {
			return err
		}
	}
	return tx.Commit(ctx)
}

func installBaseline(ctx context.Context, tx pgx.Tx, m migration) error {
	var alembic *string
	if err := tx.QueryRow(ctx, "SELECT to_regclass('alembic_version')::text").Scan(&alembic); err != nil {
		return err
	}
	if alembic != nil {
		var versions []string
		rows, err := tx.Query(ctx, "SELECT version_num FROM alembic_version")
		if err != nil {
			return err
		}
		for rows.Next() {
			var v string
			if err := rows.Scan(&v); err != nil {
				rows.Close()
				return err
			}
			versions = append(versions, v)
		}
		rows.Close()
		if len(versions) != 1 || versions[0] != LegacyHead {
			return fmt.Errorf("database is at schema revision %v but only %s can be adopted: "+
				"upgrade older databases to %s first; newer revisions are not supported", versions, LegacyHead, LegacyHead)
		}
		if missing, err := missingColumns(ctx, tx); err != nil {
			return err
		} else if len(missing) > 0 {
			return fmt.Errorf("schema does not match revision %s; missing columns: %s", LegacyHead, strings.Join(missing, ", "))
		}
		return nil
	}
	var existing *string
	if err := tx.QueryRow(ctx, "SELECT to_regclass('receipts')::text").Scan(&existing); err != nil {
		return err
	}
	if existing != nil {
		return errors.New("database has application tables but no alembic_version marker; refusing to adopt it")
	}
	if _, err := tx.Exec(ctx, m.sql); err != nil {
		return fmt.Errorf("apply baseline: %w", err)
	}
	// Keep the legacy marker so existing tooling and older releases keep working on this data.
	if _, err := tx.Exec(ctx, "CREATE TABLE alembic_version (version_num varchar(32) NOT NULL CONSTRAINT alembic_version_pkc PRIMARY KEY)"); err != nil {
		return err
	}
	_, err := tx.Exec(ctx, "INSERT INTO alembic_version (version_num) VALUES ($1)", LegacyHead)
	return err
}

func missingColumns(ctx context.Context, tx pgx.Tx) ([]string, error) {
	rows, err := tx.Query(ctx, `SELECT table_name || '.' || column_name FROM information_schema.columns
		WHERE table_schema = current_schema()`)
	if err != nil {
		return nil, err
	}
	have := map[string]bool{}
	for rows.Next() {
		var c string
		if err := rows.Scan(&c); err != nil {
			rows.Close()
			return nil, err
		}
		have[c] = true
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}
	var missing []string
	for _, c := range strings.Fields(expectedColumns) {
		if !have[c] {
			missing = append(missing, c)
		}
	}
	return missing, nil
}

// SchemaReady reports whether the latest embedded migration has been recorded.
func SchemaReady(ctx context.Context, pool *pgxpool.Pool) (bool, error) {
	var ok bool
	err := pool.QueryRow(ctx, "SELECT EXISTS (SELECT 1 FROM "+ledgerTable+" WHERE version = $1)", LatestVersion()).Scan(&ok)
	if err != nil {
		return false, err
	}
	return ok, nil
}
