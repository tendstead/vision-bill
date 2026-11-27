// Package files validates uploaded images and manages their storage: the
// temporary area, permanent storage and thumbnails.
package files

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/tendstead/vision-bill/internal/ids"
)

// Allowed media types.
const (
	JPEG = "image/jpeg"
	PNG  = "image/png"
	WebP = "image/webp"
	HEIC = "image/heic"
	HEIF = "image/heif"
	GIF  = "image/gif"
)

// UnsupportedTypeError reports content that is not an accepted image.
type UnsupportedTypeError struct{ Detected string }

func (e *UnsupportedTypeError) Error() string { return "Unsupported image type: " + e.Detected }

// HEICUnavailableError reports a HEIC/HEIF image when no decoder helper exists.
type HEICUnavailableError struct{}

func (HEICUnavailableError) Error() string {
	return "HEIC/HEIF images are not supported on this server because the image decoder is not installed"
}

// Options configure the service.
type Options struct {
	TmpDir, SaveDir string
	ThumbMaxEdge    int
	ThumbQuality    int
	// HelperPath overrides the image helper lookup ("" = find vips in PATH).
	HelperPath string
	Logger     *slog.Logger
}

// Service handles image inspection and storage.
type Service struct {
	tmp, save string
	maxEdge   int
	quality   int
	helper    string // path to the vips binary, "" when unavailable
	sem       chan struct{}
	log       *slog.Logger
}

// New creates the storage directories and locates the optional helper.
func New(o Options) (*Service, error) {
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	for _, d := range []string{o.TmpDir, o.SaveDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return nil, err
		}
	}
	s := &Service{tmp: o.TmpDir, save: o.SaveDir, maxEdge: o.ThumbMaxEdge, quality: o.ThumbQuality,
		sem: make(chan struct{}, 2), log: o.Logger}
	helper := o.HelperPath
	if helper == "" {
		if p, err := exec.LookPath("vips"); err == nil {
			helper = p
		}
	}
	s.helper = helper
	if helper == "" {
		s.log.Warn("image helper (vips) not found: thumbnails are disabled and HEIC/HEIF uploads are rejected")
	}
	return s, nil
}

// HEICSupported reports whether HEIC/HEIF uploads can be processed.
func (s *Service) HEICSupported() bool { return s.helper != "" }

// DetectMediaType sniffs the content type from the leading bytes.
func DetectMediaType(b []byte) string {
	switch {
	case len(b) >= 3 && b[0] == 0xff && b[1] == 0xd8 && b[2] == 0xff:
		return JPEG
	case bytes.HasPrefix(b, []byte("\x89PNG\r\n\x1a\n")):
		return PNG
	case bytes.HasPrefix(b, []byte("GIF87a")), bytes.HasPrefix(b, []byte("GIF89a")):
		return GIF
	case len(b) >= 12 && bytes.Equal(b[0:4], []byte("RIFF")) && bytes.Equal(b[8:12], []byte("WEBP")):
		return WebP
	case len(b) >= 12 && bytes.Equal(b[4:8], []byte("ftyp")):
		switch string(b[8:12]) {
		case "heic", "heix", "hevc", "hevx", "heim", "heis", "hevm", "hevs":
			return HEIC
		case "mif1", "msf1":
			return HEIF
		case "avif", "avis":
			return "image/avif"
		}
		return "application/octet-stream"
	case len(b) >= 2 && bytes.Equal(b[:2], []byte("BM")):
		return "image/bmp"
	case bytes.HasPrefix(b, []byte("%PDF")):
		return "application/pdf"
	}
	if len(b) == 0 {
		return "application/x-empty"
	}
	return "application/octet-stream"
}

// Inspect validates content and returns its media type.
func (s *Service) Inspect(content []byte) (string, error) {
	mt := DetectMediaType(content)
	switch mt {
	case JPEG, PNG, WebP, GIF:
		return mt, nil
	case HEIC, HEIF:
		if !s.HEICSupported() {
			return mt, HEICUnavailableError{}
		}
		return mt, nil
	}
	return mt, &UnsupportedTypeError{Detected: mt}
}

// UndecodableError reports an image whose content cannot be decoded.
type UndecodableError struct{ Err error }

func (e *UndecodableError) Error() string { return "The image could not be decoded" }
func (e *UndecodableError) Unwrap() error { return e.Err }

// Validate inspects content and, for HEIC/HEIF (which only the helper can
// read), proves that it decodes so a broken file is rejected at upload time
// instead of failing later during analysis.
func (s *Service) Validate(ctx context.Context, content []byte) (string, error) {
	mt, err := s.Inspect(content)
	if err != nil {
		return mt, err
	}
	if mt == HEIC || mt == HEIF {
		if _, _, err := s.ForModel(ctx, content); err != nil {
			return mt, &UndecodableError{Err: err}
		}
	}
	return mt, nil
}

// StoreTmp writes content to a unique temporary file and returns its path.
func (s *Service) StoreTmp(content []byte) (string, error) {
	p := filepath.Join(s.tmp, "temp_"+ids.UUID()+".png")
	if err := os.WriteFile(p, content, 0o644); err != nil {
		return "", err
	}
	return p, nil
}

func thumbFor(original string) string {
	base := filepath.Base(original)
	stem := strings.TrimSuffix(base, filepath.Ext(base))
	return filepath.Join(filepath.Dir(original), "thumbnails", stem+".thumb.webp")
}

// ThumbPath is the deterministic thumbnail location of an original.
func ThumbPath(original string) string { return thumbFor(original) }

// StorePerm moves a tmp image (and its thumbnail) to permanent storage and
// returns the new paths; an empty string means "absent".
func (s *Service) StorePerm(tmpPath, receiptID string) (orig, thumb string, err error) {
	ext := filepath.Ext(tmpPath)
	dest := filepath.Join(s.save, "receipt_"+receiptID+ext)
	if _, statErr := os.Stat(tmpPath); statErr != nil {
		// A previous attempt may have moved the file before recording its
		// path: report the already-stored copy so the caller can repair it.
		if _, e := os.Stat(dest); e == nil && ext != "" {
			if _, te := os.Stat(thumbFor(dest)); te == nil {
				return dest, thumbFor(dest), nil
			}
			return dest, "", nil
		}
		s.log.Warn("tmp image does not exist, nothing to store", "path", tmpPath)
		return "", "", nil
	}
	if _, e := os.Stat(dest); e == nil {
		dest = filepath.Join(s.save, ids.Hex(16)+"_receipt_"+receiptID+ext)
	}
	if err := moveFile(tmpPath, dest); err != nil {
		return "", "", err
	}
	src := thumbFor(tmpPath)
	if _, e := os.Stat(src); e == nil {
		td := thumbFor(dest)
		if err := os.MkdirAll(filepath.Dir(td), 0o755); err != nil {
			return dest, "", err
		}
		if err := moveFile(src, td); err != nil {
			return dest, "", err
		}
		return dest, td, nil
	}
	return dest, "", nil
}

func moveFile(src, dst string) error {
	if err := os.Rename(src, dst); err == nil {
		return nil
	}
	// Different filesystems: copy then remove.
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		_ = os.Remove(dst)
		return err
	}
	if err := out.Close(); err != nil {
		_ = os.Remove(dst)
		return err
	}
	return os.Remove(src)
}

// Delete removes an image and its thumbnail; false when the file is missing.
func (s *Service) Delete(path string) bool {
	if path == "" {
		return false
	}
	if err := os.Remove(path); err != nil {
		s.log.Warn("image file could not be deleted", "path", path, "err", err)
		return false
	}
	_ = os.Remove(thumbFor(path))
	return true
}

// run executes the helper with a timeout and bounded concurrency.
func (s *Service) run(ctx context.Context, args ...string) error {
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	case <-ctx.Done():
		return ctx.Err()
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, s.helper, args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("image helper failed: %w: %s", err, strings.TrimSpace(string(out)))
	}
	return nil
}

// GenerateThumbnail writes <dir>/thumbnails/<stem>.thumb.webp for src. It
// fails soft: "" is returned when no thumbnail could be made.
func (s *Service) GenerateThumbnail(ctx context.Context, src string) string {
	if s.helper == "" {
		return ""
	}
	dest := thumbFor(src)
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		s.log.Warn("thumbnail directory unavailable", "err", err)
		return ""
	}
	target := fmt.Sprintf("%s[Q=%d]", dest, s.quality)
	// thumbnail applies the EXIF orientation by default and only shrinks.
	if err := s.run(ctx, "thumbnail", src, target, fmt.Sprint(s.maxEdge), "--height", fmt.Sprint(s.maxEdge), "--size", "down"); err != nil {
		s.log.Warn("thumbnail generation failed", "path", src, "err", err)
		_ = os.Remove(dest)
		return ""
	}
	return dest
}

// ForModel returns the bytes to send to a model: HEIC/HEIF is converted to an
// upright JPEG, everything else is passed through unchanged.
func (s *Service) ForModel(ctx context.Context, content []byte) ([]byte, string, error) {
	mt := DetectMediaType(content)
	if mt != HEIC && mt != HEIF {
		return content, mt, nil
	}
	if s.helper == "" {
		return nil, mt, HEICUnavailableError{}
	}
	in, err := os.CreateTemp(s.tmp, "convert_*.heic")
	if err != nil {
		return nil, mt, err
	}
	defer os.Remove(in.Name())
	if _, err := in.Write(content); err != nil {
		in.Close()
		return nil, mt, err
	}
	in.Close()
	out := in.Name() + ".jpg"
	defer os.Remove(out)
	if err := s.run(ctx, "autorot", in.Name(), out+"[Q=92]"); err != nil {
		return nil, mt, err
	}
	b, err := os.ReadFile(out)
	if err != nil {
		return nil, mt, err
	}
	return b, JPEG, nil
}
