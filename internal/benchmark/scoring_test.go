package benchmark

import (
	"encoding/json"
	"testing"

	"github.com/tendstead/vision-bill/internal/domain"
)

func parse(t *testing.T, body string) *domain.Receipt {
	t.Helper()
	r, issues := domain.ParseReceipt([]byte(body), domain.Options{})
	if issues != nil {
		t.Fatal(issues)
	}
	return r
}

const base = `{"confidence": 80, "merchant_name": "Shop", "date": "2025-03-04", "time": "12:30:00", "currency": "EUR", "category": "grocery",
 "payment_method": "cash", "subtotal": "3.00", "total": "3.57", "tax_total": "0.57", "discount_total": "0",
 "line_items": [
  {"description": "Milk", "spending_category": "milk", "quantity": 2, "unit_price": "1.50", "total_price": "3.00", "tags": ["dairy"]},
  {"description": "Bread", "spending_category": "bread_bakery", "quantity": 1, "unit_price": "2.00", "total_price": "2.00", "tags": []}],
 "taxes": [{"name": "VAT", "rate": 0.19, "amount": "0.57"}]}`

func TestIdenticalReceiptsScorePerfectly(t *testing.T) {
	d := DetailedScore(parse(t, base), parse(t, base), true)
	if d.Scores["strict"] != 1.0 || d.Scores["normalized"] != 1.0 || d.Scores["classification"] != 1.0 || d.Scores["ordered_items"] != 1.0 {
		t.Fatalf("%v", d.Scores)
	}
	if len(d.Differences) != 0 || d.Rows["matched"] != 2 || d.Rows["missing"] != 0 {
		t.Fatalf("%+v", d)
	}
	if got := ScoreReceipts(parse(t, base), parse(t, base))["overall"]; got < 0.999 {
		t.Fatalf("legacy overall %v", got)
	}
}

func TestFormattingDifferencesAreEquivalentButOneCentIsNot(t *testing.T) {
	actual := parse(t, replace(base, `"unit_price": "1.50"`, `"unit_price": "1.5"`, `"time": "12:30:00"`, `"time": "12:30"`, `"total": "3.57"`, `"total": "3.58"`))
	d := DetailedScore(parse(t, base), actual, true)
	byPath := map[string]difference{}
	for _, x := range d.Differences {
		byPath[x.Path] = x
	}
	if x, ok := byPath["line_items[0].unit_price"]; !ok || !x.Equivalent {
		t.Fatalf("1.50 vs 1.5 must be equivalent: %+v", d.Differences)
	}
	if x, ok := byPath["time"]; !ok || !x.Equivalent {
		t.Fatalf("12:30:00 vs 12:30 must be equivalent: %+v", d.Differences)
	}
	if x, ok := byPath["total"]; !ok || x.Equivalent {
		t.Fatalf("one cent must not be forgiven: %+v", d.Differences)
	}
	if d.Scores["normalized"].(float64) >= 1 || d.Scores["strict"].(float64) >= d.Scores["normalized"].(float64) {
		t.Fatalf("scores %v", d.Scores)
	}
}

func replace(s string, pairs ...string) string {
	for i := 0; i+1 < len(pairs); i += 2 {
		for j := 0; j+len(pairs[i]) <= len(s); j++ {
			if s[j:j+len(pairs[i])] == pairs[i] {
				s = s[:j] + pairs[i+1] + s[j+len(pairs[i]):]
				break
			}
		}
	}
	return s
}

func TestMissingAndExtraRows(t *testing.T) {
	fewer := parse(t, replace(base, `,
  {"description": "Bread", "spending_category": "bread_bakery", "quantity": 1, "unit_price": "2.00", "total_price": "2.00", "tags": []}`, ``))
	d := DetailedScore(parse(t, base), fewer, true)
	if d.Rows["missing"] != 1 || d.Rows["extra"] != 0 || d.Rows["matched"] != 1 {
		t.Fatalf("%+v", d.Rows)
	}
	var found bool
	for _, x := range d.Differences {
		if x.Path == "line_items[1]" && x.Actual == nil && !x.Equivalent {
			found = true
		}
	}
	if !found {
		t.Fatalf("missing-row difference absent: %+v", d.Differences)
	}
	d = DetailedScore(fewer, parse(t, base), true)
	if d.Rows["extra"] != 1 || d.Rows["missing"] != 0 {
		t.Fatalf("%+v", d.Rows)
	}
	// A repeated purchase is aligned, not collapsed.
	twice := parse(t, replace(base, `{"description": "Bread"`, `{"description": "Milk"`))
	if d := DetailedScore(twice, twice, true); d.Rows["matched"] != 2 {
		t.Fatalf("%+v", d.Rows)
	}
}

func TestClassificationScoring(t *testing.T) {
	wrongCat := parse(t, replace(base, `"spending_category": "milk"`, `"spending_category": "cheese"`, `"tags": ["dairy"]`, `"tags": ["other"]`))
	d := DetailedScore(parse(t, base), wrongCat, true)
	// labels: milk(wrong) , tags(wrong), bread ok, tags ok, category ok => 3 of 5
	if got := d.Scores["classification"].(float64); got != 0.6 {
		t.Fatalf("classification %v", got)
	}
	if d := DetailedScore(parse(t, base), wrongCat, false); d.Scores["classification"] != nil {
		t.Fatal("classification must be null when disabled")
	}
}

func TestChecksAndFingerprint(t *testing.T) {
	d := DetailedScore(parse(t, base), parse(t, base), true)
	if d.Checks["line_totals_sum"] != "5.00" || d.Checks["tax_lines_sum_matches_tax_total"] != true || d.Checks["schema_net_plus_tax_matches_total"] != true {
		t.Fatalf("%v", d.Checks)
	}
	a, b := parse(t, base), parse(t, replace(base, `"confidence": 80`, `"confidence": 10`))
	if DatasetFingerprint([]*domain.Receipt{a}) != DatasetFingerprint([]*domain.Receipt{b}) {
		t.Fatal("confidence must not influence the fingerprint")
	}
	c := parse(t, replace(base, `"merchant_name": "Shop"`, `"merchant_name": "Other"`))
	if DatasetFingerprint([]*domain.Receipt{a}) == DatasetFingerprint([]*domain.Receipt{c}) {
		t.Fatal("content must influence the fingerprint")
	}
	if DatasetFingerprint([]*domain.Receipt{a, c}) != DatasetFingerprint([]*domain.Receipt{c, a}) {
		t.Fatal("order must not influence the fingerprint")
	}
}

func TestLegacyScoreComponents(t *testing.T) {
	other := parse(t, replace(base, `"merchant_name": "Shop"`, `"merchant_name": "Other"`, `"total": "3.57"`, `"total": "9.99"`))
	s := ScoreReceipts(parse(t, base), other)
	// total differs; the absent tip matches on both sides.
	if s["header"] != 7.0/8 || s["totals"] != 4.0/5 || s["line_items"] != 1 || s["spending_categories"] != 1 {
		t.Fatalf("%v", s)
	}
}

func TestDumpIsJSONRoundTrippable(t *testing.T) {
	d := Dump(parse(t, base))
	b, err := json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	r, issues := domain.ParseReceipt(b, domain.Options{})
	if issues != nil {
		t.Fatalf("dump must parse back: %v", issues)
	}
	if Dump(r).Subtotal != "3.00" || Dump(r).LineItems[0].Quantity != 2 {
		t.Fatalf("%+v", Dump(r))
	}
}
