// Package testdb creates throwaway PostgreSQL databases for integration tests.
// Tests skip when VB_TEST_DATABASE_URL (any database on the target server,
// with CREATEDB rights) is unset.
package testdb

import (
	"context"
	"fmt"
	"math/rand"
	"net/url"
	"os"
	"testing"

	"github.com/jackc/pgx/v5/pgxpool"
)

// New returns a pool connected to a fresh empty database and its DSN.
func New(t testing.TB) (*pgxpool.Pool, string) {
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
	u, err := url.Parse(base)
	if err != nil {
		t.Fatal(err)
	}
	u.Path = "/" + name
	dsn := u.String()
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		pool.Close()
		_, _ = admin.Exec(ctx, "DROP DATABASE "+name+" WITH (FORCE)")
		admin.Close()
	})
	return pool, dsn
}
