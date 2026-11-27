// Command server runs the Vision Bill API and serves the embedded SPA.
package main

import (
	"context"
	"embed"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"runtime/debug"
	"strings"
	"syscall"
	"time"

	"github.com/tendstead/vision-bill/internal/analysis"
	"github.com/tendstead/vision-bill/internal/app"
	"github.com/tendstead/vision-bill/internal/benchmark"
	"github.com/tendstead/vision-bill/internal/config"
	"github.com/tendstead/vision-bill/internal/files"
	"github.com/tendstead/vision-bill/internal/httpapi"
	"github.com/tendstead/vision-bill/internal/llm"
)

// defaultMemoryLimit is the soft heap target of the server process.
const defaultMemoryLimit = 192 << 20

//go:embed all:static
var staticFiles embed.FS

func main() {
	if err := run(); err != nil {
		slog.Error("fatal", "err", err)
		os.Exit(1)
	}
}

func run() error {
	if len(os.Args) > 1 && os.Args[1] == "-healthcheck" {
		return healthcheck()
	}
	// Password hashing and photo uploads allocate large transient buffers. A soft
	// memory limit makes the collector reclaim them promptly instead of letting
	// the heap grow to twice its live size (GOMEMLIMIT overrides this default).
	if os.Getenv("GOMEMLIMIT") == "" {
		debug.SetMemoryLimit(defaultMemoryLimit)
	}
	src := config.OSSources()
	cfg, err := config.Load(src)
	if err != nil {
		return err
	}
	log := newLogger(cfg.API.LogLevel)
	slog.SetDefault(log)
	if created, err := config.Initialize(src.YAMLPath, cfg); err != nil {
		log.Warn("could not create YAML config", "path", src.YAMLPath, "err", err)
	} else if created {
		log.Info("created initial YAML config", "path", src.YAMLPath)
	}
	holder := config.NewHolder(cfg)
	static, err := fs.Sub(staticFiles, "static")
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	fileSvc, err := files.New(files.Options{
		TmpDir: cfg.Images.TmpDir, SaveDir: cfg.Images.SaveDir,
		ThumbMaxEdge: cfg.Images.ThumbnailMaxEdge, ThumbQuality: cfg.Images.ThumbnailQuality, Logger: log,
	})
	if err != nil {
		return fmt.Errorf("image storage: %w", err)
	}
	provider, err := llm.New(cfg.LLM)
	if err != nil {
		return err
	}

	db := &app.DB{}
	analysisSvc := &analysis.Service{Store: db.Get, Files: fileSvc, Provider: provider, Config: holder, Log: log}
	scheduler := analysis.NewScheduler(analysisSvc)
	scheduler.Start(ctx)
	benchmarks := &benchmark.Service{Store: db.Get, Provider: provider, Analysis: analysisSvc, Log: log}
	benchmarks.Start(ctx)
	dbErr := make(chan error, 1)
	go func() { dbErr <- db.Connect(ctx, holder.Get(), log) }()

	// Like a conventional startup, wait briefly for the database before
	// accepting requests; if it stays unreachable the server starts degraded
	// and keeps retrying in the background.
	select {
	case <-db.Ready():
	case err := <-dbErr:
		// Connect only returns for fatal problems (migration/adoption) or shutdown.
		if err != nil && ctx.Err() == nil {
			return fmt.Errorf("database setup: %w", err)
		}
	case <-time.After(30 * time.Second):
		log.Warn("database not ready yet, starting in degraded mode")
	case <-ctx.Done():
	}

	srv := &http.Server{
		Addr: listenAddr(),
		Handler: httpapi.New(httpapi.Options{
			Config: holder, Store: db.Get, Static: static, Logger: log, ConfigSources: src,
			Files: fileSvc, Provider: provider, Analysis: analysisSvc, Scheduler: scheduler, Benchmarks: benchmarks,
			OnSettingsChanged: func(c config.Config) { provider.SetTemperature(c.LLM.Temperature) },
		}),
		ReadHeaderTimeout: 10 * time.Second,
		// Whole-request read limit, which bounds a (slow) upload of up to 200 MB.
		ReadTimeout: 5 * time.Minute,
	}
	serveErr := make(chan error, 1)
	go func() {
		log.Info("listening", "addr", srv.Addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serveErr <- err
		}
	}()

	var runErr error
	select {
	case <-ctx.Done():
	case runErr = <-serveErr:
	case err := <-dbErr:
		// Connect only returns for fatal problems (migration/adoption) or shutdown.
		if err != nil && ctx.Err() == nil {
			runErr = fmt.Errorf("database setup: %w", err)
		} else {
			select {
			case <-ctx.Done():
			case runErr = <-serveErr:
			}
		}
	}
	shutdown, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdown)
	scheduler.Stop()
	benchmarks.Stop()
	if st := db.Get(); st != nil {
		st.Close()
	}
	return runErr
}

// healthcheck probes the local liveness endpoint (used by the container HEALTHCHECK).
func healthcheck() error {
	url, err := healthURL(listenAddr())
	if err != nil {
		return err
	}
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("healthz returned %d", resp.StatusCode)
	}
	return nil
}

// healthURL builds the loopback liveness URL for a listen address such as
// ":8080", "0.0.0.0:8080", "[::]:8080" or "[::1]:8080". A wildcard or empty
// host is probed on 127.0.0.1; an explicit host (also IPv6) is used as is.
func healthURL(addr string) (string, error) {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return "", fmt.Errorf("invalid listen address %q: %w", addr, err)
	}
	switch host {
	case "", "0.0.0.0", "::":
		// Go's wildcard listeners are dual-stack, so IPv4 loopback always works.
		host = "127.0.0.1"
	}
	return "http://" + net.JoinHostPort(host, port) + "/healthz", nil
}

// listenAddr is where the server binds: a fixed container port (8080) that
// does not follow api.port, which only mirrors the published host port in
// Compose. Override with VB_LISTEN_ADDR.
func listenAddr() string {
	if a := os.Getenv("VB_LISTEN_ADDR"); a != "" {
		return a
	}
	return ":8080"
}

// newLogger builds the stdout logger honouring api.log_level.
func newLogger(level string) *slog.Logger {
	var l slog.Level
	switch strings.ToUpper(level) {
	case "DEBUG":
		l = slog.LevelDebug
	case "WARNING", "WARN":
		l = slog.LevelWarn
	case "ERROR", "CRITICAL":
		l = slog.LevelError
	default:
		l = slog.LevelInfo
	}
	return slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: l}))
}
