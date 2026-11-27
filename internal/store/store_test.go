package store

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
)

// newDB creates a throwaway database on the server in VB_TEST_DATABASE_URL
// (a URL for any database; "postgres" is typical) and returns a pool for it.
func newDB(t *testing.T) *pgxpool.Pool {
	t.Helper()
	base := os.Getenv("VB_TEST_DATABASE_URL")
	if base == "" {
		t.Skip("VB_TEST_DATABASE_URL not set")
	}
	ctx := context.Background()
	admin, err := pgxpool.New(ctx, base)
	if err != nil {
		t.Fatal(err)
	}
	name := fmt.Sprintf("vbtest_%d", rand.Int63())
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+name); err != nil {
		t.Fatal(err)
	}
	cfg := admin.Config().ConnConfig.Copy()
	cfg.Database = name
	pool, err := pgxpool.New(ctx, fmt.Sprintf("postgres://%s:%s@%s:%d/%s", cfg.User, cfg.Password, cfg.Host, cfg.Port, name))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		_, _ = admin.Exec(ctx, "DROP DATABASE "+name+" WITH (FORCE)")
		admin.Close()
	})
	return pool
}

func TestFreshInstall(t *testing.T) {
	pool := newDB(t)
	ctx := context.Background()
	if err := Migrate(ctx, pool); err != nil {
		t.Fatal(err)
	}
	if err := Migrate(ctx, pool); err != nil { // idempotent
		t.Fatal(err)
	}
	var v string
	if err := pool.QueryRow(ctx, "SELECT version_num FROM alembic_version").Scan(&v); err != nil || v != AlembicHead {
		t.Fatalf("alembic marker %q %v", v, err)
	}
	var n int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM tags").Scan(&n); err != nil || n != 16 {
		t.Fatalf("seed tags %d %v", n, err)
	}
	if ok, err := SchemaReady(ctx, pool); err != nil || !ok {
		t.Fatalf("not ready: %v %v", ok, err)
	}
}

func TestAdoptAlembicHead(t *testing.T) {
	pool := newDB(t)
	ctx := context.Background()
	ms, _ := loadMigrations()
	// Simulate an Alembic-managed database: baseline SQL + marker, no ledger.
	if _, err := pool.Exec(ctx, ms[0].sql); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, "CREATE TABLE alembic_version (version_num varchar(32) PRIMARY KEY); INSERT INTO alembic_version VALUES ('0010')"); err != nil {
		t.Fatal(err)
	}
	if _, err := pool.Exec(ctx, "INSERT INTO users (username, hashed_password) VALUES ('keep', 'h')"); err != nil {
		t.Fatal(err)
	}
	if err := Migrate(ctx, pool); err != nil {
		t.Fatal(err)
	}
	var n int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM users").Scan(&n); err != nil || n != 1 {
		t.Fatalf("data lost: %d %v", n, err)
	}
}

func TestRejectsOtherRevisionsAndDrift(t *testing.T) {
	ctx := context.Background()
	for _, rev := range []string{"0009", "0020"} {
		pool := newDB(t)
		if _, err := pool.Exec(ctx, "CREATE TABLE alembic_version (version_num varchar(32)); INSERT INTO alembic_version VALUES ('"+rev+"')"); err != nil {
			t.Fatal(err)
		}
		if err := Migrate(ctx, pool); err == nil || !strings.Contains(err.Error(), rev) {
			t.Fatalf("rev %s: %v", rev, err)
		}
	}
	pool := newDB(t)
	if _, err := pool.Exec(ctx, "CREATE TABLE alembic_version (version_num varchar(32)); INSERT INTO alembic_version VALUES ('0010')"); err != nil {
		t.Fatal(err)
	}
	if err := Migrate(ctx, pool); err == nil || !strings.Contains(err.Error(), "missing columns") {
		t.Fatalf("drift not detected: %v", err)
	}
	pool = newDB(t)
	if _, err := pool.Exec(ctx, "CREATE TABLE receipts (id int)"); err != nil {
		t.Fatal(err)
	}
	if err := Migrate(ctx, pool); err == nil {
		t.Fatal("unmarked database adopted")
	}
}
