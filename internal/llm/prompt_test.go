package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestPromptContent(t *testing.T) {
	p := BuildPrompt([]string{"food", "my tag", "deposit"}, 1, PromptOptions{})
	for _, want := range []string{
		"There are 1 input photos.",
		"Prefer tags from this list: food, my tag, deposit.",
		"milk: Dairy milk and plant milk alternatives",
		"deposit: Container deposit or refund",
		`"code": "unreadable"`,
		`"merchant_name"`,
	} {
		if !strings.Contains(p, want) {
			t.Errorf("prompt missing %q", want)
		}
	}
	if strings.Contains(p, "duplicate_candidates") {
		t.Error("single-photo prompt must use the plain receipt schema")
	}
	if !strings.Contains(BuildPrompt(nil, 1, PromptOptions{}), "Add short free-form tags") {
		t.Error("free-form tag instruction expected without a vocabulary")
	}
}

func TestMultiPhotoPrompt(t *testing.T) {
	p := BuildPrompt([]string{"food"}, 3, PromptOptions{})
	for _, want := range []string{"There are 3 input photos.", "source_image", "duplicate_candidates", "Do NOT remove duplicates yourself."} {
		if !strings.Contains(p, want) {
			t.Errorf("multi-photo prompt missing %q", want)
		}
	}
}

func TestExtractionOnlyPromptDropsClassification(t *testing.T) {
	for _, n := range []int{1, 2} {
		p := BuildPrompt([]string{"food"}, n, PromptOptions{ExtractionOnly: true})
		i := strings.Index(p, "matching this schema: ")
		if i < 0 {
			t.Fatal("schema marker missing")
		}
		var schema map[string]any
		if err := json.Unmarshal([]byte(p[i+len("matching this schema: "):]), &schema); err != nil {
			t.Fatalf("schema not valid JSON: %v", err)
		}
		raw := p[i:]
		for _, banned := range []string{`"spending_category"`, `"tags"`} {
			if strings.Contains(raw, banned) {
				t.Errorf("%d photos: %s must be removed from the schema", n, banned)
			}
		}
		if !strings.Contains(raw, `"merchant_name"`) || !strings.Contains(p, "Do not invent missing values") {
			t.Error("extraction prompt lost essential content")
		}
	}
}

func TestEmbeddedSchemasListReceiptFields(t *testing.T) {
	for _, multi := range []bool{false, true} {
		props, _ := schemaDoc(multi)["properties"].(map[string]any)
		for _, f := range []string{"merchant_name", "line_items", "total", "payment_method"} {
			if props[f] == nil {
				t.Errorf("multi=%v: schema lacks %s", multi, f)
			}
		}
		if _, has := props["duplicate_candidates"]; has != multi {
			t.Errorf("multi=%v: duplicate_candidates presence wrong", multi)
		}
	}
}
