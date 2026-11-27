package analysis

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/tendstead/vision-bill/internal/llm"
)

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
