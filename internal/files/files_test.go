package files

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var (
	jpegBytes = append([]byte{0xff, 0xd8, 0xff, 0xe0}, make([]byte, 20)...)
	pngBytes  = append([]byte("\x89PNG\r\n\x1a\n"), make([]byte, 20)...)
	heicBytes = append([]byte{0, 0, 0, 24}, []byte("ftypheic\x00\x00\x00\x00heicmif1")...)
)

func TestDetectMediaType(t *testing.T) {
	cases := map[string][]byte{
		JPEG: jpegBytes, PNG: pngBytes, GIF: []byte("GIF89a......"), WebP: []byte("RIFF\x00\x00\x00\x00WEBPVP8 "),
		HEIC: heicBytes, HEIF: append([]byte{0, 0, 0, 24}, []byte("ftypmif1\x00\x00\x00\x00")...),
		"image/avif":      append([]byte{0, 0, 0, 24}, []byte("ftypavif\x00\x00\x00\x00")...),
		"application/pdf": []byte("%PDF-1.7"), "application/x-empty": nil,
		"application/octet-stream": []byte("hello world, not an image"),
	}
	for want, b := range cases {
		if got := DetectMediaType(b); got != want {
			t.Errorf("want %s, got %s", want, got)
		}
	}
}

func newService(t *testing.T, helper string) *Service {
	t.Helper()
	dir := t.TempDir()
	s, err := New(Options{TmpDir: filepath.Join(dir, "tmp"), SaveDir: filepath.Join(dir, "save"), ThumbMaxEdge: 512, ThumbQuality: 80, HelperPath: helper})
	if err != nil {
		t.Fatal(err)
	}
	if helper == "" {
		s.helper = "" // ignore any vips on PATH
	}
	return s
}

// fakeHelper stands in for the image helper: it copies the input to the output.
func fakeHelper(t *testing.T) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "vips")
	script := `#!/bin/sh
case "$1" in
  thumbnail) out="${3%%\[*}"; cp "$2" "$out" ;;
  autorot) out="${3%%\[*}"; printf '\377\330\377jpeg' > "$out" ;;
  *) exit 2 ;;
esac
`
	if err := os.WriteFile(p, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestInspect(t *testing.T) {
	s := newService(t, "")
	for _, ok := range [][]byte{jpegBytes, pngBytes} {
		if _, err := s.Inspect(ok); err != nil {
			t.Fatal(err)
		}
	}
	var unsupported *UnsupportedTypeError
	if _, err := s.Inspect([]byte("%PDF-1.7")); !errors.As(err, &unsupported) || unsupported.Detected != "application/pdf" {
		t.Fatalf("pdf: %v", err)
	}
	if _, err := s.Inspect(heicBytes); !errors.As(err, new(HEICUnavailableError)) {
		t.Fatalf("heic without helper: %v", err)
	}
	if s.HEICSupported() {
		t.Fatal("heic must be unsupported without helper")
	}
	s2 := newService(t, fakeHelper(t))
	if mt, err := s2.Inspect(heicBytes); err != nil || mt != HEIC || !s2.HEICSupported() {
		t.Fatalf("heic with helper: %v %v", mt, err)
	}
}

func TestStoreTmpPermAndDelete(t *testing.T) {
	s := newService(t, fakeHelper(t))
	ctx := context.Background()
	tmp, err := s.StoreTmp(pngBytes)
	if err != nil || !strings.HasPrefix(filepath.Base(tmp), "temp_") || filepath.Ext(tmp) != ".png" {
		t.Fatalf("%v %v", tmp, err)
	}
	thumb := s.GenerateThumbnail(ctx, tmp)
	if thumb != filepath.Join(filepath.Dir(tmp), "thumbnails", strings.TrimSuffix(filepath.Base(tmp), ".png")+".thumb.webp") {
		t.Fatalf("thumb path %q", thumb)
	}
	if _, err := os.Stat(thumb); err != nil {
		t.Fatal(err)
	}
	orig, newThumb, err := s.StorePerm(tmp, "rid")
	if err != nil || filepath.Base(orig) != "receipt_rid.png" || newThumb != filepath.Join(filepath.Dir(orig), "thumbnails", "receipt_rid.thumb.webp") {
		t.Fatalf("%v %v %v", orig, newThumb, err)
	}
	for _, gone := range []string{tmp, thumb} {
		if _, err := os.Stat(gone); !os.IsNotExist(err) {
			t.Fatalf("%s should have moved", gone)
		}
	}
	// A second image for the same receipt id must not overwrite the first.
	tmp2, _ := s.StoreTmp(jpegBytes)
	orig2, _, err := s.StorePerm(tmp2, "rid")
	if err != nil || orig2 == orig || !strings.HasSuffix(orig2, "_receipt_rid.png") {
		t.Fatalf("collision handling: %v %v", orig2, err)
	}
	// A missing tmp file is a soft no-op...
	if o, th, err := s.StorePerm("/nonexistent/x.png", "other"); o != "" || th != "" || err != nil {
		t.Fatalf("missing: %q %q %v", o, th, err)
	}
	// ...unless an earlier attempt already stored it: report that copy (repeatable move).
	if o, th, err := s.StorePerm("/nonexistent/x.png", "rid"); o != orig || th != newThumb || err != nil {
		t.Fatalf("already moved: %q %q %v", o, th, err)
	}
	if !s.Delete(orig) {
		t.Fatal("delete")
	}
	if _, err := os.Stat(newThumb); !os.IsNotExist(err) {
		t.Fatal("thumbnail should be removed with the original")
	}
	if s.Delete(orig) || s.Delete("") {
		t.Fatal("deleting a missing file reports false")
	}
}

func TestThumbnailFailsSoft(t *testing.T) {
	s := newService(t, "")
	tmp, _ := s.StoreTmp(pngBytes)
	if got := s.GenerateThumbnail(context.Background(), tmp); got != "" {
		t.Fatalf("no helper => no thumbnail, got %q", got)
	}
	failing := filepath.Join(t.TempDir(), "vips")
	_ = os.WriteFile(failing, []byte("#!/bin/sh\nexit 1\n"), 0o755)
	s2 := newService(t, failing)
	if got := s2.GenerateThumbnail(context.Background(), tmp); got != "" {
		t.Fatalf("helper failure => no thumbnail, got %q", got)
	}
}

func TestForModel(t *testing.T) {
	ctx := context.Background()
	s := newService(t, fakeHelper(t))
	b, mt, err := s.ForModel(ctx, pngBytes)
	if err != nil || mt != PNG || string(b) != string(pngBytes) {
		t.Fatalf("passthrough: %v %v", mt, err)
	}
	b, mt, err = s.ForModel(ctx, heicBytes)
	if err != nil || mt != JPEG || DetectMediaType(b) != JPEG {
		t.Fatalf("heic conversion: %v %v", mt, err)
	}
	if _, _, err := newService(t, "").ForModel(ctx, heicBytes); err == nil {
		t.Fatal("heic without helper must fail")
	}
}

func TestValidateRejectsUndecodableHEIC(t *testing.T) {
	ctx := context.Background()
	s := newService(t, fakeHelper(t))
	if mt, err := s.Validate(ctx, heicBytes); err != nil || mt != HEIC {
		t.Fatalf("decodable heic: %v %v", mt, err)
	}
	if _, err := s.Validate(ctx, pngBytes); err != nil {
		t.Fatalf("png needs no helper: %v", err)
	}
	failing := filepath.Join(t.TempDir(), "vips")
	_ = os.WriteFile(failing, []byte("#!/bin/sh\nexit 1\n"), 0o755)
	var undecodable *UndecodableError
	if _, err := newService(t, failing).Validate(ctx, heicBytes); !errors.As(err, &undecodable) {
		t.Fatalf("broken heic must be rejected: %v", err)
	}
	// Formats the helper is not needed for are never sent through it.
	if _, err := newService(t, failing).Validate(ctx, jpegBytes); err != nil {
		t.Fatal(err)
	}
}
