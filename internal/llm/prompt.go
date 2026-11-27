package llm

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/tendstead/vision-bill/internal/domain"
)

//go:embed receipt_schema.json
var receiptSchemaJSON []byte

//go:embed multi_schema.json
var multiSchemaJSON []byte

// Prompt options mirror the experiment settings that change the prompt.
type PromptOptions struct {
	// ExtractionOnly selects the literal-extraction prompt (no classification).
	ExtractionOnly bool
}

func schemaDoc(multi bool) map[string]any {
	raw := receiptSchemaJSON
	if multi {
		raw = multiSchemaJSON
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		panic("llm: embedded schema is invalid: " + err.Error())
	}
	return doc
}

func schemaText(multi bool) string {
	b, err := json.Marshal(schemaDoc(multi))
	if err != nil {
		panic(err)
	}
	return string(b)
}

// extractionPrompt is the extraction-only prompt: the schema minus
// category/spending_category/tags.
func extractionPrompt(imageCount int) string {
	schema := schemaDoc(imageCount > 1)
	objs := []map[string]any{schema}
	if defs, ok := schema["$defs"].(map[string]any); ok {
		for _, d := range defs {
			if m, ok := d.(map[string]any); ok {
				objs = append(objs, m)
			}
		}
	}
	dropped := []string{"category", "spending_category", "tags"}
	for _, obj := range objs {
		if props, ok := obj["properties"].(map[string]any); ok {
			for _, name := range dropped {
				delete(props, name)
			}
		}
		if req, ok := obj["required"].([]any); ok {
			kept := req[:0]
			for _, r := range req {
				if s, _ := r.(string); !slices.Contains(dropped, s) {
					kept = append(kept, r)
				}
			}
			obj["required"] = kept
		}
	}
	text, err := json.Marshal(schema)
	if err != nil {
		panic(err)
	}
	overlap := "Keep line_items in printed top-to-bottom order."
	if imageCount > 1 {
		overlap = "Extract each printed row from each photo, including overlaps. Set source_image to its " +
			"zero-based photo index. Group rows by photo and then top-to-bottom. In duplicate_candidates " +
			"identify the complete ordered overlap using zero-based line_items indices. Do not remove " +
			"duplicates yourself. Totals must come from printed totals, not duplicated rows."
	}
	return fmt.Sprintf("Extract the literal contents of ONE receipt from %d ordered photos. %s ", imageCount, overlap) +
		"Preserve genuinely repeated purchases. Copy product names and abbreviations as printed; " +
		"do not expand, translate, categorize or tag them. Read merchant_name from printed text, " +
		"never guess from a logo. Positive prices are purchases; refunds and returned deposits " +
		"have negative prices. Do not invent missing values. If unreadable, return only " +
		`{"error":{"code":"unreadable","message":"Which part needs a clearer photo"}}. ` +
		"Return only JSON matching this schema: " + string(text)
}

// BuildPrompt renders the extraction prompt: instructions, the tag vocabulary,
// spending categories and the receipt JSON schema.
func BuildPrompt(tags []string, imageCount int, opts PromptOptions) string {
	if opts.ExtractionOnly {
		return extractionPrompt(imageCount)
	}
	tagInstruction := "Add short free-form tags to each line_item when useful."
	if len(tags) > 0 {
		tagInstruction = "Add short tags to each line_item when useful. Prefer tags from this list: " +
			strings.Join(tags, ", ") + ". If no tag in the list fits, you may suggest a single new " +
			"short tag (lowercase snake_case, at most three words); suggested tags are " +
			"reviewed by a human before they become standard tags."
	}
	parts := make([]string, len(domain.SpendingCategories))
	for i, code := range domain.SpendingCategories {
		parts[i] = code + ": " + domain.CategoryDefinitions[code]
	}
	categories := strings.Join(parts, "; ")
	overlap := "List line_items in the exact top-to-bottom order on the receipt."
	if imageCount > 1 {
		overlap = "Extract EVERY purchased row separately from EACH photo, including overlap rows. " +
			"Do NOT remove duplicates yourself. Set source_image to its zero-based photo index. " +
			"Group line_items by photo, then top-to-bottom within each photo. " +
			"In duplicate_candidates identify only the SAME printed rows visible at the end " +
			"of one photo and the start of the next. Use zero-based line_items array indices. " +
			"Include the entire ordered overlap, not isolated matching products. " +
			"Use an empty candidate list when there is no clear visual overlap. " +
			"Receipt-level totals must come from the printed receipt, not the duplicated rows."
	}
	schema := schemaText(imageCount > 1)
	return "You are a receipt analyser. You will be given one or more images of ONE receipt.\n" +
		"    Images are ordered from top to bottom and may overlap. Return ONE complete receipt.\n" +
		fmt.Sprintf("    There are %d input photos. %s\n", imageCount, overlap) +
		"    Preserve genuinely repeated purchases, even when descriptions and prices are identical.\n" +
		"    If the receipt is too blurry, cut off, or unreadable to extract reliably, return ONLY\n" +
		"    {\"error\": {\"code\": \"unreadable\", \"message\": \"Explain which part needs a clearer photo\"}}.\n" +
		"    This error response is an alternative to the receipt schema. Do not invent missing values.\n" +
		"    You should analyze the receipt and provide a machine-readable JSON response based on the receipt.\n" +
		"    Determine merchant_name from text that is actually printed on the receipt (store name in the\n" +
		"    header or footer, the address line, a website, or any other written hint).\n" +
		"    Do not guess or infer the company name from a logo, photo, or design; if no printed name exists,\n" +
		"    use the best available written hint.\n" +
		"    " + tagInstruction + "\n" +
		"    Set exactly one spending_category on each line_item using only these codes:\n" +
		"    " + categories + ".\n" +
		"    Use unknown when the text does not support a category; use deposit for Pfand lines.\n" +
		"    Set the top-level category to the single best category for the whole purchase.\n" +
		"    Use positive unit_price and total_price for deposit/Pfand charges; use negative prices\n" +
		"    for refunds, credits, or deposits returned to the customer.\n" +
		"    Tag every deposit charge or return with `deposit`, and set spending_category to deposit.\n" +
		"    Do not add any additional text or commentary. Only provide the JSON response matching this schema:\n" +
		"\n" +
		"    " + schema + "\n" +
		"    "
}

// SchemaSHA256 fingerprints the receipt validation schema.
func SchemaSHA256() string {
	b, err := json.Marshal(schemaDoc(false))
	if err != nil {
		panic(err)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
