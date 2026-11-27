// Command server runs the Vision Bill API and serves the embedded SPA.
package main

import (
	"context"
	"embed"
	"errors"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/tendstead/vision-bill/internal/httpapi"
)

//go:embed all:static
var staticFiles embed.FS

func main() {
	port := os.Getenv("API__PORT")
	if port == "" {
		port = "8080"
	}
	static, err := fs.Sub(staticFiles, "static")
	if err != nil {
		slog.Error("embedded static", "err", err)
		os.Exit(1)
	}
	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           httpapi.New(static),
		ReadHeaderTimeout: 10 * time.Second,
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdown)
	}()
	slog.Info("listening", "addr", srv.Addr)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		slog.Error("server", "err", err)
		os.Exit(1)
	}
}
