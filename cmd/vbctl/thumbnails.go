package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/tendstead/vision-bill/internal/config"
	"github.com/tendstead/vision-bill/internal/files"
	"github.com/tendstead/vision-bill/internal/store"
)

// loadToolConfig resolves the application configuration. Values the tools
// do not need (LLM and auth settings) get harmless placeholders so a minimal
// environment (database + image directories) is enough.
func loadToolConfig() (*config.Config, error) {
	for k, v := range map[string]string{
		"AUTH__SECRET_KEY": "tool-not-used", "LLM__HOST": "localhost", "LLM__API_KEY": "none",
		"LLM__MODEL_NAME": "none", "LLM__TEMPERATURE": "0", "PG__USER": "postgres", "PG__PASSWORD": "postgres",
	} {
		if os.Getenv(k) == "" {
			_ = os.Setenv(k, v)
		}
	}
	return config.Load(config.OSSources())
}

// resolveStored maps a stored (possibly container-absolute) path to a local
// file, also probing the bare file name inside the configured directories.
func resolveStored(stored string, dirs []string) string {
	stored = strings.TrimSpace(stored)
	if stored == "" {
		return ""
	}
	candidates := []string{stored}
	for _, d := range dirs {
		if d != "" {
			candidates = append(candidates, filepath.Join(d, filepath.Base(stored)))
		}
	}
	for _, c := range candidates {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			return c
		}
	}
	return ""
}

func runThumbnails(args []string) error {
	fs := flag.NewFlagSet("thumbnails", flag.ContinueOnError)
	dry := fs.Bool("dry-run", false, "report, do not write")
	limit := fs.Int("limit", 0, "cap the number of rows processed")
	if err := fs.Parse(args); err != nil {
		return usageError("")
	}
	cfg, err := loadToolConfig()
	if err != nil {
		return err
	}
	ctx := context.Background()
	st, err := store.Open(ctx, cfg.PG.DSN())
	if err != nil {
		return err
	}
	defer st.Close()
	svc, err := files.New(files.Options{TmpDir: cfg.Images.TmpDir, SaveDir: cfg.Images.SaveDir,
		ThumbMaxEdge: cfg.Images.ThumbnailMaxEdge, ThumbQuality: cfg.Images.ThumbnailQuality, Logger: slog.Default()})
	if err != nil {
		return err
	}
	query := `SELECT id::text, image_path FROM images WHERE thumbnail_path IS NULL AND image_path IS NOT NULL ORDER BY created_at`
	if *limit > 0 {
		query += " LIMIT " + strconv.Itoa(*limit)
	}
	rows, err := st.Pool.Query(ctx, query)
	if err != nil {
		return err
	}
	type row struct{ id, path string }
	var todo []row
	for rows.Next() {
		var r row
		if err := rows.Scan(&r.id, &r.path); err != nil {
			rows.Close()
			return err
		}
		todo = append(todo, r)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}
	dirs := []string{cfg.Images.SaveDir, cfg.Images.TmpDir}
	processed, skipped, failed := 0, 0, 0
	for _, r := range todo {
		path := resolveStored(r.path, dirs)
		if path == "" {
			skipped++
			continue
		}
		if *dry {
			processed++
			continue
		}
		thumb := svc.GenerateThumbnail(ctx, path)
		if thumb == "" {
			failed++
			continue
		}
		if err := st.UpdateImageThumbnailPath(ctx, r.id, thumb); err != nil {
			return err
		}
		processed++
	}
	verb := "processed"
	if *dry {
		verb = "would process"
	}
	fmt.Printf("%s %d image(s); %d had no resolvable file; %d failed to decode.\n", verb, processed, skipped, failed)
	return nil
}
