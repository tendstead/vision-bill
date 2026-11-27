//go:build e2e

package suite

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tendstead/vision-bill/internal/domain"
)

// fixtureCase is a registered image + ground-truth pair (e2e/data, opt-in via fixtures.txt).
type fixtureCase struct {
	Name     string
	Image    string
	Raw      []byte
	Expected *domain.Receipt
}

var imageExts = []string{".png", ".jpg", ".jpeg"}

// loadFixtureCases reads e2e/data/fixtures.txt. Missing or invalid entries are
// skipped with a warning, never fatal.
func loadFixtureCases(logf func(string, ...any)) []fixtureCase {
	data := dir("data")
	f, err := os.Open(filepath.Join(data, "fixtures.txt"))
	if err != nil {
		return nil
	}
	defer f.Close()
	var out []fixtureCase
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		name := strings.TrimSpace(sc.Text())
		if name == "" || strings.HasPrefix(name, "#") {
			continue
		}
		image := ""
		for _, ext := range imageExts {
			if _, err := os.Stat(filepath.Join(data, name+ext)); err == nil {
				image = filepath.Join(data, name+ext)
				break
			}
		}
		if image == "" {
			logf("e2e fixtures: skipping %q: no image found", name)
			continue
		}
		raw, err := os.ReadFile(filepath.Join(data, name+".json"))
		if err != nil {
			logf("e2e fixtures: skipping %q: missing ground-truth JSON", name)
			continue
		}
		expected, issues := domain.ParseReceipt(raw, domain.Options{})
		if issues != nil {
			logf("e2e fixtures: skipping %q: invalid ground truth: %v", name, issues)
			continue
		}
		out = append(out, fixtureCase{Name: name, Image: image, Raw: raw, Expected: expected})
	}
	return out
}

func cases(t testing.TB) []fixtureCase { return loadFixtureCases(t.Logf) }

// firstCase returns the first registered fixture; the test fails when none is registered.
func firstCase(t *testing.T) fixtureCase {
	t.Helper()
	cs := cases(t)
	if len(cs) == 0 {
		t.Fatal("no e2e fixtures registered (e2e/data/fixtures.txt)")
	}
	return cs[0]
}

func (fc fixtureCase) file(t *testing.T) File {
	return File{Name: filepath.Base(fc.Image), ContentType: "image/png", Data: readFile(t, fc.Image)}
}

// analyzeFixture uploads the fixture and returns the analysed image row.
func analyzeFixture(t *testing.T, c *Client, fc fixtureCase) map[string]any {
	t.Helper()
	r := c.Upload("", fc.file(t)).Is(201, 202)
	return waitImage(t, c, str(r.M(), "image_id"), true, "analyzed")
}

// receiptOf returns the receipt detail of an analysed image row.
func receiptOf(t *testing.T, c *Client, row map[string]any) map[string]any {
	t.Helper()
	return c.Get("/receipts/" + str(row, "receipt_id")).Is(200).M()
}
