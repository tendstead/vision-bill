package analysis

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tendstead/vision-bill/internal/domain"
	"github.com/tendstead/vision-bill/internal/files"
	"github.com/tendstead/vision-bill/internal/llm"
	"github.com/tendstead/vision-bill/internal/store"
)

// shortProvider answers every chat with "unreadable" and has a tiny deadline.
type shortProvider struct{ llm.Provider }

func (shortProvider) Timeout() time.Duration { return 50 * time.Millisecond }
func (shortProvider) Chat(context.Context, string, []llm.Message, llm.ChatOptions) (string, error) {
	return `{"error": {"code": "unreadable", "message": "blurry"}}`, nil
}

// Waiting for a busy inference slot must not eat into the extraction deadline.
func TestAnalyseDeadlineStartsAfterSlotWait(t *testing.T) {
	dir := t.TempDir()
	fs, err := files.New(files.Options{TmpDir: filepath.Join(dir, "tmp"), SaveDir: filepath.Join(dir, "save")})
	if err != nil {
		t.Fatal(err)
	}
	img := filepath.Join(dir, "a.png")
	if err := os.WriteFile(img, []byte("\x89PNG\r\n\x1a\nxxxx"), 0o644); err != nil {
		t.Fatal(err)
	}
	s := &Service{Files: fs, Provider: shortProvider{}, Store: func() *store.Store { return nil }}
	hold := make(chan struct{})
	started := make(chan struct{}, DefaultInferenceSlots)
	for i := 0; i < DefaultInferenceSlots; i++ {
		go s.Inference(context.Background(), func(context.Context) error { started <- struct{}{}; <-hold; return nil })
	}
	for i := 0; i < DefaultInferenceSlots; i++ {
		<-started
	}
	time.AfterFunc(200*time.Millisecond, func() { close(hold) }) // 4x the provider deadline
	_, err = s.Analyse(context.Background(), "m", []domain.ImagePart{{ImagePath: img}})
	var unreadable *llm.UnreadableError
	if !errors.As(err, &unreadable) {
		t.Fatalf("waiting for a slot must not count as a timeout, got %v", err)
	}
}

func TestInferenceSlotsBoundConcurrency(t *testing.T) {
	s := &Service{}
	var running, peak atomic.Int32
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = s.Inference(context.Background(), func(context.Context) error {
				n := running.Add(1)
				for {
					p := peak.Load()
					if n <= p || peak.CompareAndSwap(p, n) {
						break
					}
				}
				time.Sleep(20 * time.Millisecond)
				running.Add(-1)
				return nil
			})
		}()
	}
	wg.Wait()
	if got := peak.Load(); got != DefaultInferenceSlots {
		t.Fatalf("peak concurrency %d, want %d", got, DefaultInferenceSlots)
	}
}

func TestInferenceGivesUpWhenContextEnds(t *testing.T) {
	s := &Service{}
	hold := make(chan struct{})
	started := make(chan struct{}, DefaultInferenceSlots)
	for i := 0; i < DefaultInferenceSlots; i++ {
		go s.Inference(context.Background(), func(context.Context) error { started <- struct{}{}; <-hold; return nil })
	}
	for i := 0; i < DefaultInferenceSlots; i++ {
		<-started
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	ran := false
	if err := s.Inference(ctx, func(context.Context) error { ran = true; return nil }); err == nil || ran {
		t.Fatalf("waiting for a slot must stop with the context (err=%v ran=%v)", err, ran)
	}
	close(hold)
}

func TestChooseModel(t *testing.T) {
	models := []llm.ModelInfo{{ID: "a"}, {ID: "b"}, {ID: "c"}}
	req := func(s string) *string { return &s }
	cases := []struct {
		requested  *string
		configured string
		want       string
	}{
		{req("c"), "b", "c"},   // an available requested model wins
		{req("zzz"), "b", "b"}, // unknown request falls back to the configured model
		{nil, "b", "b"},
		{nil, "missing", "a"}, // then the first vision model
		{req(""), "missing", "a"},
	}
	for i, c := range cases {
		if got := ChooseModel(models, c.requested, c.configured); got != c.want {
			t.Errorf("case %d: got %q, want %q", i, got, c.want)
		}
	}
	if ChooseModel(nil, nil, "x") != "" {
		t.Error("no models => empty")
	}
}
