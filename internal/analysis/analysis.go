// Package analysis runs model extraction for stored submissions and drains
// the durable queue of pending images in the background.
package analysis

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"sync"

	"github.com/tendstead/vision-bill/internal/config"
	"github.com/tendstead/vision-bill/internal/domain"
	"github.com/tendstead/vision-bill/internal/files"
	"github.com/tendstead/vision-bill/internal/llm"
	"github.com/tendstead/vision-bill/internal/store"
)

// DefaultInferenceSlots bounds concurrent model calls (uploads, the worker and
// benchmarks share the same slots) so a busy server cannot flood the model host.
const DefaultInferenceSlots = 2

// TimedOutMessage is stored with a submission whose analysis timed out.
const TimedOutMessage = "The LLM request timed out. The image stays queued and is retried automatically with increasing delays."

// Service wires extraction to storage.
type Service struct {
	Store    func() *store.Store
	Files    *files.Service
	Provider llm.Provider
	Config   *config.Holder
	Log      *slog.Logger

	gateOnce sync.Once
	gate     chan struct{}
}

// Inference runs fn while holding an inference slot; it returns ctx.Err()
// when no slot becomes free before ctx ends.
func (s *Service) Inference(ctx context.Context, fn func(context.Context) error) error {
	s.gateOnce.Do(func() { s.gate = make(chan struct{}, DefaultInferenceSlots) })
	select {
	case s.gate <- struct{}{}:
		defer func() { <-s.gate }()
		return fn(ctx)
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *Service) log() *slog.Logger {
	if s.Log != nil {
		return s.Log
	}
	return slog.Default()
}

// Tags returns the tag vocabulary for the prompt (empty when the DB is down).
func (s *Service) Tags(ctx context.Context) []string {
	st := s.Store()
	if st == nil {
		return nil
	}
	tags, err := st.ListTags(ctx)
	if err != nil {
		s.log().Warn("could not load tag vocabulary", "err", err)
		return nil
	}
	return tags
}

// LoadImages reads the photos from disk, converting HEIC/HEIF for the model.
func (s *Service) LoadImages(ctx context.Context, parts []domain.ImagePart) ([]llm.Image, error) {
	out := make([]llm.Image, 0, len(parts))
	for _, p := range parts {
		data, err := os.ReadFile(p.ImagePath)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return nil, fmt.Errorf("Image not found at: %s", p.ImagePath)
			}
			return nil, err
		}
		data, mt, err := s.Files.ForModel(ctx, data)
		if err != nil {
			return nil, err
		}
		out = append(out, llm.Image{Name: p.ImagePath, MediaType: mt, Data: data})
	}
	return out, nil
}

// Analyse extracts a receipt from the photos within the configured deadline.
// Deadline expiry is reported as *llm.TimeoutError.
func (s *Service) Analyse(ctx context.Context, model string, parts []domain.ImagePart) (*domain.Receipt, error) {
	// The deadline covers loading and the model calls but not the wait for an
	// inference slot: a busy server must delay an analysis, never time it out.
	var receipt *domain.Receipt
	err := s.Inference(ctx, func(ctx context.Context) error {
		ctx, cancel := context.WithTimeout(ctx, s.Provider.Timeout())
		defer cancel()
		images, err := s.LoadImages(ctx, parts)
		if err != nil {
			return err
		}
		a, err := llm.Analyse(ctx, s.Provider, model, images, s.Tags(ctx), llm.PromptOptions{}, llm.ChatOptions{})
		if err != nil {
			return err
		}
		receipt = a.Receipt
		return nil
	})
	if err != nil {
		return nil, s.mapErr(err)
	}
	return receipt, nil
}

func (s *Service) mapErr(err error) error {
	if llm.IsTimeout(err) {
		return llm.NewTimeoutError()
	}
	return err
}

// ChooseModel prefers the requested model, then the configured one, then the
// first available vision model. It returns "" when none is available.
func ChooseModel(models []llm.ModelInfo, requested *string, configured string) string {
	has := func(id string) bool {
		for _, m := range models {
			if m.ID == id {
				return true
			}
		}
		return false
	}
	switch {
	case requested != nil && *requested != "" && has(*requested):
		return *requested
	case has(configured):
		return configured
	case len(models) > 0:
		return models[0].ID
	}
	return ""
}

// MovePermanent moves all photos of a submission from temporary to permanent
// storage and records the new paths. It is safe to repeat after a partial run.
func (s *Service) MovePermanent(ctx context.Context, st *store.Store, img *domain.ImageRow, receiptID string) error {
	if img.ImagePath != nil && *img.ImagePath != "" {
		path, thumb, err := s.Files.StorePerm(*img.ImagePath, receiptID, true)
		if err != nil {
			return err
		}
		if path != "" {
			if err := st.UpdateImagePath(ctx, img.ID, path); err != nil {
				return err
			}
		}
		if thumb != "" {
			if err := st.UpdateImageThumbnailPath(ctx, img.ID, thumb); err != nil {
				return err
			}
		}
	}
	if len(img.AdditionalImages) == 0 {
		return nil
	}
	parts := append([]domain.ImagePart(nil), img.AdditionalImages...)
	for i := range parts {
		path, thumb, err := s.Files.StorePerm(parts[i].ImagePath, receiptID, false)
		if err != nil {
			return err
		}
		if path != "" {
			parts[i].ImagePath = path
		}
		if thumb != "" {
			parts[i].ThumbnailPath = &thumb
		}
		if err := st.UpdateAdditionalImages(ctx, img.ID, parts); err != nil {
			return err
		}
	}
	return nil
}

// DeleteFiles removes every photo (and thumbnail) of a submission.
func (s *Service) DeleteFiles(img *domain.ImageRow) {
	for _, p := range img.Parts() {
		if p.ImagePath != "" {
			s.Files.Delete(p.ImagePath)
		}
	}
}
