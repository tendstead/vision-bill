//go:build e2e

package suite

import (
	"bytes"
	"encoding/binary"
	"path/filepath"
	"testing"
)

// webpSize returns the pixel size of a WebP file (lossy, lossless or extended).
func webpSize(t *testing.T, d []byte) (int, int) {
	t.Helper()
	if len(d) < 30 || string(d[:4]) != "RIFF" || string(d[8:12]) != "WEBP" {
		t.Fatal("not a WebP file")
	}
	switch string(d[12:16]) {
	case "VP8 ":
		return int(binary.LittleEndian.Uint16(d[26:28]) & 0x3fff), int(binary.LittleEndian.Uint16(d[28:30]) & 0x3fff)
	case "VP8L":
		bits := binary.LittleEndian.Uint32(d[21:25])
		return int(bits&0x3fff) + 1, int((bits>>14)&0x3fff) + 1
	case "VP8X":
		return int(uint32(d[24])|uint32(d[25])<<8|uint32(d[26])<<16) + 1, int(uint32(d[27])|uint32(d[28])<<8|uint32(d[29])<<16) + 1
	}
	t.Fatalf("unknown WebP chunk %q", d[12:16])
	return 0, 0
}

func TestOriginalsAreKeptAndThumbnailsAreUpright(t *testing.T) {
	for _, tc := range []struct {
		name, ctype string
		w, h        int
	}{
		{"synthetic.heic", "image/heic", 200, 100},
		{"synthetic-exif-rotated.jpg", "image/jpeg", 100, 200}, // EXIF orientation 6 applied
	} {
		t.Run(tc.name, func(t *testing.T) {
			newEnv(t)
			c, _ := register(t, "formats")
			data := readFile(t, filepath.Join(dir("assets"), tc.name))
			id := str(c.Upload("", File{Name: tc.name, ContentType: tc.ctype, Data: data}).Is(201, 202).M(), "image_id")

			orig := c.Get("/images/" + id + "/file").Is(200)
			if !bytes.Equal(orig.Body, data) || orig.Header.Get("Content-Type") != tc.ctype {
				t.Fatalf("original not stored byte for byte (%s)", orig.Header.Get("Content-Type"))
			}
			thumb := c.Get("/images/" + id + "/thumb").Is(200)
			if thumb.Header.Get("Content-Type") != "image/webp" {
				t.Fatal(thumb.Header)
			}
			if w, h := webpSize(t, thumb.Body); w != tc.w || h != tc.h {
				t.Fatalf("thumbnail %dx%d, want %dx%d", w, h, tc.w, tc.h)
			}
		})
	}
}

func TestCorruptHEICIsRejectedAndNothingIsStored(t *testing.T) {
	newEnv(t)
	c, _ := register(t, "formats")
	broken := append([]byte("\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heicmif1"), make([]byte, 64)...)
	r := c.Upload("", File{Name: "broken.heic", ContentType: "image/heic", Data: broken}).Is(422)
	if str(r.M(), "detail") != "The image could not be decoded" {
		t.Fatal(r)
	}
	if n := len(c.Get("/images").Is(200).L()); n != 0 {
		t.Fatalf("%d images stored", n)
	}
}
