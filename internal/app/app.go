// Package app assembles configuration, the database and background workers.
package app

import (
	"context"
	"errors"
	"log/slog"
	"sync/atomic"
	"time"

	"github.com/tendstead/vision-bill/internal/config"
	"github.com/tendstead/vision-bill/internal/security"
	"github.com/tendstead/vision-bill/internal/store"
)

// DB lazily connects to PostgreSQL and exposes the live store (nil while down).
type DB struct {
	ptr atomic.Pointer[store.Store]
}

// Get returns the store or nil when the database is unavailable.
func (d *DB) Get() *store.Store { return d.ptr.Load() }

// Connect opens the pool, migrates and bootstraps the first admin. A fatal
// migration/adoption problem is returned (the caller must stop); plain
// connection failures are retried until ctx ends.
func (d *DB) Connect(ctx context.Context, cfg config.Config, log *slog.Logger) error {
	delay := time.Second
	for {
		st, err := store.Open(ctx, cfg.PG.DSN())
		if err == nil {
			if err := store.Migrate(ctx, st.Pool); err != nil {
				st.Close()
				return err
			}
			if err := BootstrapAdmin(ctx, st, cfg, log); err != nil {
				log.Error("authentication bootstrap failed", "err", err)
			}
			d.ptr.Store(st)
			return nil
		}
		log.Warn("database unavailable, retrying", "err", err, "in", delay)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
		}
		if delay < 15*time.Second {
			delay *= 2
		}
	}
}

// BootstrapAdmin creates the first admin on an empty users table and assigns
// legacy orphan rows to it. Idempotent.
func BootstrapAdmin(ctx context.Context, st *store.Store, cfg config.Config, log *slog.Logger) error {
	n, err := st.CountUsers(ctx)
	if err != nil || n > 0 {
		return err
	}
	if cfg.Auth.BootstrapUsername == nil || cfg.Auth.BootstrapPassword == nil ||
		*cfg.Auth.BootstrapUsername == "" || *cfg.Auth.BootstrapPassword == "" {
		return nil
	}
	hashed, err := security.HashPassword(cfg.Auth.EffectivePepper(), *cfg.Auth.BootstrapPassword)
	if err != nil {
		return err
	}
	admin, err := st.CreateUser(ctx, *cfg.Auth.BootstrapUsername, hashed, true)
	if errors.Is(err, store.ErrUsernameTaken) {
		log.Info("bootstrap user already exists, skipping")
		return nil
	}
	if err != nil {
		return err
	}
	if err := st.AssignOrphans(ctx, admin.ID); err != nil {
		return err
	}
	log.Info("bootstrapped first admin", "username", admin.Username, "id", admin.ID)
	return nil
}
