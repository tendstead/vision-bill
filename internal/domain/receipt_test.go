package domain

import (
	"fmt"
	"strings"
	"testing"

	"github.com/shopspring/decimal"
)

func dec(s string) decimal.Decimal { return decimal.RequireFromString(s) }

func TestMoneyRoundsHalfAwayFromZero(t *testing.T) {
	// Approved difference from Python: no float64 intermediate.
	cases := map[string]string{
		"0.015": "0.02", "0.045": "0.05", "0.145": "0.15", "0.285": "0.29", "1.005": "1.01",
		"1.015": "1.02", "1.115": "1.12", "2.675": "2.68", "10.075": "10.08", "-1.005": "-1.01",
		"99999999999999.99": "99999999999999.99", "12.5": "12.50",
	}
	for in, want := range cases {
		if got := Money(dec(in)).StringFixed(2); got != want {
			t.Errorf("Money(%s) = %s, want %s", in, got, want)
		}
	}
	if got := Quantity(dec("0.00015")).StringFixed(4); got != "0.0002" {
		t.Errorf("quantity %s", got)
	}
	for in, want := range map[string]string{"0.19": "0.1900", "0.075": "0.0750", "0.07505": "0.0751"} {
		if got := Rate(dec(in)).StringFixed(4); got != want {
			t.Errorf("rate %s = %s", in, got)
		}
	}
}

const valid = `{
 "confidence": 90, "merchant_name": "Shop", "date": "2025-03-04", "time": "12:30",
 "currency": "USD", "category": "grocery", "payment_method": "cash",
 "subtotal": "10.00", "total": 11.90, "tax_total": 1.90, "discount_total": "0",
 "line_items": [
  {"description": "Milk", "spending_category": "milk", "quantity": 2, "unit_price": "1.5", "total_price": 3, "tags": [" Dairy ", "dairy", "x  y", ""]},
  {"description": "Odd", "spending_category": "nonsense", "quantity": "1.5", "unit_price": 1, "total_price": 1.5}
 ],
 "taxes": [{"name": "VAT", "rate": 0.19, "amount": "1.90"}]
}`

func TestParseValidReceipt(t *testing.T) {
	r, issues := ParseReceipt([]byte(valid), Options{})
	if issues != nil {
		t.Fatal(issues)
	}
	if r.Currency != "USD" || r.Total.String() != "11.9" || r.Date.String() != "2025-03-04" || *r.Time != "12:30" {
		t.Fatalf("%+v", r)
	}
	if got := strings.Join(r.LineItems[0].Tags, "|"); got != "Dairy|x y" {
		t.Fatalf("tags %q", got)
	}
	// Model output with an unknown category is coerced (no spending authority).
	if r.LineItems[1].SpendingCategory != "unknown" || !r.LineItems[1].CategoryExplicit {
		t.Fatalf("category %+v", r.LineItems[1])
	}
	if r.LineItems[1].Quantity.String() != "1.5" {
		t.Fatal("string quantity")
	}
	if r.Taxes[0].Rate.String() != "0.19" {
		t.Fatal("rate")
	}
}

func TestParseDefaults(t *testing.T) {
	r, issues := ParseReceipt([]byte(`{"confidence":1,"merchant_name":"m","date":"2025-01-01","line_items":[],"subtotal":1,"total":1}`), Options{})
	if issues != nil {
		t.Fatal(issues)
	}
	if r.Currency != "EUR" || r.Category != "other" || r.PaymentMethod != "unknown" || !r.DiscountTotal.IsZero() || r.Tip != nil {
		t.Fatalf("%+v", r)
	}
}

func TestParseIssues(t *testing.T) {
	_, issues := ParseReceipt([]byte(`{"confidence": 101, "merchant_name": 5, "date": "nope", "category": "x",
	  "line_items": [{"description": "a", "quantity": 0, "unit_price": "abc", "total_price": 1}], "subtotal": -1, "total": 1, "tip": -1,
	  "taxes": [{"name": "v", "amount": -1}], "discount_total": -5}`), Options{})
	want := map[string]string{
		"confidence": "less_than_equal", "merchant_name": "string_type", "date": "date_from_datetime_parsing",
		"category": "literal_error", "line_items.0.quantity": "greater_than", "line_items.0.unit_price": "decimal_parsing",
		"tip": "greater_than_equal", "taxes.0.amount": "greater_than_equal", "discount_total": "greater_than_equal",
	}
	got := map[string]string{}
	for _, i := range issues {
		parts := make([]string, len(i.Loc))
		for n, l := range i.Loc {
			parts[n] = fmt.Sprint(l)
		}
		got[strings.Join(parts, ".")] = i.Type
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("%s: got %q want %q (all: %v)", k, got[k], v, got)
		}
	}
}

func TestUpdateOptions(t *testing.T) {
	body := `{"confidence":1,"merchant_name":"m","date":"2025-01-01","subtotal":1,"total":1,
	  "collection_ids":["11111111-2222-3333-4444-555555555555"],
	  "line_items":[{"id":"AAAAAAAA-2222-3333-4444-555555555555","description":"a","quantity":1,"unit_price":1,"total_price":1,"spending_category":"milk","remember_category":true}]}`
	r, issues := ParseReceipt([]byte(body), Options{Update: true})
	if issues != nil {
		t.Fatal(issues)
	}
	if r.CollectionIDs == nil || len(*r.CollectionIDs) != 1 || *r.LineItems[0].ID != "aaaaaaaa-2222-3333-4444-555555555555" || !r.LineItems[0].RememberCategory {
		t.Fatalf("%+v", r)
	}
	bad := strings.Replace(body, `"milk"`, `"bogus"`, 1)
	if _, issues := ParseReceipt([]byte(bad), Options{Update: true}); len(issues) != 1 || issues[0].Type != "value_error" {
		t.Fatalf("bogus category must be rejected on PUT: %v", issues)
	}
	// omitted collection_ids stays nil
	r, _ = ParseReceipt([]byte(strings.Replace(body, `"collection_ids":["11111111-2222-3333-4444-555555555555"],`, "", 1)), Options{Update: true})
	if r.CollectionIDs != nil {
		t.Fatal("omitted must be nil")
	}
}

func TestTimeOfDay(t *testing.T) {
	ok := map[string]string{"12": "12:00:00", "12:30": "12:30:00", "12:30:45": "12:30:45", "1230": "12:30:00", "12:30:45.5": "12:30:45.500000", "00:00": "00:00:00"}
	for in, want := range ok {
		us, err := ParseTimeOfDay(in)
		if err != nil || FormatTimeOfDay(us) != want {
			t.Errorf("%q -> %v %v", in, FormatTimeOfDay(us), err)
		}
	}
	for _, in := range []string{"", "abc", "25:00", "12:60", "1:30", "12:3"} {
		if _, err := ParseTimeOfDay(in); err == nil {
			t.Errorf("%q accepted", in)
		}
	}
	if _, err := ParseTimeOfDay("zz"); err == nil || err.Error() != "Invalid isoformat string: 'zz'" {
		t.Errorf("message: %v", err)
	}
}

func TestNormalizeHistoryKey(t *testing.T) {
	if NormalizeHistoryKey("  STRASSE\t Café ") != "strasse café" || NormalizeHistoryKey("Straße") != "strasse" {
		t.Fatal("fold")
	}
}

func TestPyFloatRepr(t *testing.T) {
	cases := map[float64]string{
		2: "2.0", 0.19: "0.19", 1.5: "1.5", 0: "0.0", 100: "100.0", 0.0001: "0.0001",
		0.00001: "1e-05", 1e16: "1e+16", 123456789012345678: "1.2345678901234568e+17", -2.5: "-2.5", 0.075: "0.075",
		1e15: "1000000000000000.0",
	}
	for in, want := range cases {
		if got := pyFloatRepr(in); got != want {
			t.Errorf("%v -> %s, want %s", in, got, want)
		}
	}
}
