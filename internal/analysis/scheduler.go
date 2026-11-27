package analysis

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/tendstead/vision-bill/internal/domain"
	"github.com/tendstead/vision-bill/internal/llm"
)

// Result is the outcome of analysing one queued image.
type Result struct {
	ImageID   string  `json:"image_id"`
	Status    string  `json:"status"` // analyzed | failed | timed_out | unreadable
	ReceiptID *string `json:"receipt_id"`
	Error     *string `json:"error"`
}

// Scheduler drains the pending-image queue. A manual trigger and the periodic
// cycle never overlap. There is deliberately no terminal "unreachable" state:
// when the provider is down images stay pending and are retried next cycle.
type Scheduler struct {
	svc  *Service
	mu   sync.Mutex
	wake chan struct{}

	cancel context.CancelFunc
	done   chan struct{}
}

// NewScheduler creates a scheduler (not yet running).
func NewScheduler(svc *Service) *Scheduler {
	return &Scheduler{svc: svc, wake: make(chan struct{}, 1)}
}

// Start launches the background worker; it runs a first cycle immediately.
func (s *Scheduler) Start(ctx context.Context) {
	if s.cancel != nil {
		return
	}
	ctx, s.cancel = context.WithCancel(ctx)
	s.done = make(chan struct{})
	s.Trigger()
	go func() {
		defer close(s.done)
		for {
			interval := time.Duration(s.svc.Config.Get().Worker.CheckIntervalSeconds) * time.Second
			timer := time.NewTimer(interval)
			select {
			case <-ctx.Done():
				timer.Stop()
				return
			case <-s.wake:
			case <-timer.C:
			}
			timer.Stop()
			if _, err := s.ProcessPending(ctx); err != nil {
				s.svc.log().Error("analysis cycle failed", "err", err)
			}
		}
	}()
	s.svc.log().Info("analysis scheduler started")
}

// Stop ends the worker and waits for the running cycle to stop.
func (s *Scheduler) Stop() {
	if s.cancel == nil {
		return
	}
	s.cancel()
	<-s.done
	s.cancel = nil
}

// Trigger wakes the worker immediately; the work itself lives in PostgreSQL.
func (s *Scheduler) Trigger() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

// ProcessPending runs one cycle, serialised against concurrent calls.
func (s *Scheduler) ProcessPending(ctx context.Context) ([]Result, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.svc.Store() == nil {
		s.svc.log().Info("database not ready, skipping analysis cycle")
		return []Result{}, nil
	}
	return s.cycle(ctx)
}

func (s *Scheduler) cycle(ctx context.Context) ([]Result, error) {
	log := s.svc.log()
	st := s.svc.Store()
	results := []Result{}

	mctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	models, err := s.svc.Provider.Models(mctx)
	cancel()
	if err != nil {
		log.Warn("LLM provider unreachable, pending images stay queued")
		return results, nil
	}
	if len(models) == 0 {
		log.Warn("no vision models available, pending images stay queued")
		return results, nil
	}
	modelID := ChooseModel(models, nil, s.svc.Config.Get().LLM.ModelName)
	pending, err := st.ListPendingImages(ctx)
	if err != nil {
		return results, err
	}
	log.Info("analysis cycle", "queued", len(pending), "model", modelID)
	for _, img := range pending {
		if ctx.Err() != nil {
			break
		}
		claimed, err := st.ClaimImage(ctx, img.ID)
		if err != nil {
			return results, err
		}
		if claimed == nil {
			log.Info("image was claimed by another worker", "image", img.ID)
			continue
		}
		chosen := modelID
		if claimed.ModelID != nil {
			for _, m := range models {
				if m.ID == *claimed.ModelID {
					chosen = m.ID
				}
			}
		}
		results = append(results, s.analyzeOne(ctx, claimed, chosen))
	}
	return results, nil
}

func strp(s string) *string { return &s }

func (s *Scheduler) analyzeOne(ctx context.Context, img *domain.ImageRow, model string) Result {
	st := s.svc.Store()
	log := s.svc.log()
	// A detached context lets recording the outcome finish during shutdown.
	rec := context.WithoutCancel(ctx)

	if img.ReceiptID != nil {
		_ = st.MarkImageAnalyzed(rec, img.ID, *img.ReceiptID)
		return Result{ImageID: img.ID, Status: "analyzed", ReceiptID: img.ReceiptID}
	}
	if existing, err := st.ReceiptByImageID(rec, img.ID); err == nil && existing != nil {
		_ = st.MarkImageAnalyzed(rec, img.ID, existing.ID)
		return Result{ImageID: img.ID, Status: "analyzed", ReceiptID: &existing.ID}
	}
	if img.ImagePath == nil || *img.ImagePath == "" || !fileExists(*img.ImagePath) {
		path := ""
		if img.ImagePath != nil {
			path = *img.ImagePath
		}
		msg := "Image file missing: " + path
		log.Error(msg)
		_ = st.MarkImageFailed(rec, img.ID, msg)
		return Result{ImageID: img.ID, Status: "failed", Error: &msg}
	}

	receipt, err := s.svc.Analyse(ctx, model, img.Parts())
	if err != nil {
		var unreadable *llm.UnreadableError
		var timeout *llm.TimeoutError
		switch {
		case errors.As(err, &timeout):
			_ = st.MarkImageTimedOut(rec, img.ID, TimedOutMessage)
			return Result{ImageID: img.ID, Status: "timed_out", Error: strp(TimedOutMessage)}
		case errors.As(err, &unreadable):
			_ = st.MarkImageTerminal(rec, img.ID, "unreadable", err.Error())
			return Result{ImageID: img.ID, Status: "unreadable", Error: strp(err.Error())}
		}
		_ = st.MarkImageFailed(rec, img.ID, err.Error())
		return Result{ImageID: img.ID, Status: "failed", Error: strp(err.Error())}
	}

	bypass := img.BypassReview
	status := "unverified"
	if bypass {
		status = "verified"
	}
	row, err := st.PersistReceipt(rec, receipt, &img.ID, status, bypass, img.UserID)
	if err == nil {
		err = st.MarkImageAnalyzed(rec, img.ID, row.ID)
	}
	if err != nil {
		msg := fmt.Sprintf("Failed to persist receipt: %v", err)
		_ = st.MarkImageFailed(rec, img.ID, msg)
		return Result{ImageID: img.ID, Status: "failed", Error: &msg}
	}
	if bypass {
		if err := s.svc.MovePermanent(rec, st, img, row.ID); err != nil {
			log.Error("failed to move bypass-reviewed image to permanent storage", "image", img.ID, "err", err)
		}
	}
	return Result{ImageID: img.ID, Status: "analyzed", ReceiptID: &row.ID}
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}
