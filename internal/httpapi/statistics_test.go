package httpapi

import (
	"strings"
	"testing"
	"time"
)

func receiptJSON(merchant, date, total, qty, unit, lineTotal, category, payment string, tip string) string {
	return `{"confidence": 90, "merchant_name": "` + merchant + `", "date": "` + date + `", "currency": "EUR", "category": "` + category + `",
	 "payment_method": "` + payment + `", "subtotal": "` + total + `", "total": "` + total + `", "tip": ` + tip + `, "tax_total": "0", "discount_total": "0",
	 "line_items": [{"description": "Whole Milk", "spending_category": "milk", "quantity": ` + qty + `, "unit_price": "` + unit + `", "total_price": "` + lineTotal + `"}]}`
}

func TestProductSearch(t *testing.T) {
	e := newEnv(t, true)
	alice, bob := e.user(t, "alice", false), e.user(t, "bob", false)
	e.seed(t, alice, receiptJSON("A", "2025-03-01", "2.50", "1", "2.50", "2.50", "grocery", "cash", "null"), "verified")
	e.seed(t, alice, receiptJSON("B", "2025-03-05", "1.50", "1", "1.50", "1.50", "grocery", "cash", "null"), "verified")
	e.seed(t, alice, receiptJSON("C", "2025-03-09", "-0.25", "1", "-0.25", "-0.25", "grocery", "cash", "null"), "verified")
	e.seed(t, alice, receiptJSON("D", "2025-03-10", "9.99", "1", "9.99", "9.99", "grocery", "cash", "null"), "unverified")
	e.seed(t, bob, receiptJSON("E", "2025-03-11", "0.10", "1", "0.10", "0.10", "grocery", "cash", "null"), "verified")

	r := e.do("GET", "/api/v1/search?query=%20milk%20", nil, alice.cookie)
	if r.Code != 200 {
		t.Fatalf("%d %s", r.Code, r.Body)
	}
	j := r.json(t)
	purchases := j["purchases"].([]any)
	if j["query"] != "milk" || len(purchases) != 3 {
		t.Fatalf("%v", j)
	}
	first := purchases[0].(map[string]any)
	if first["merchant_name"] != "C" || first["unit_price"] != "-0.25" || first["currency"] != "EUR" {
		t.Fatalf("newest first expected: %v", first)
	}
	// Unverified and foreign purchases are excluded; the refund is not a "cheapest price".
	if j["latest_price"] != "-0.25" || j["cheapest_price"] != "1.50" || j["currency"] != "EUR" {
		t.Fatalf("summary %v", j)
	}
	if got := j["average_price"].(string); got != "1.25" {
		t.Fatalf("average %s", got)
	}
	empty := e.do("GET", "/api/v1/search?query=zzz", nil, alice.cookie).json(t)
	if len(empty["purchases"].([]any)) != 0 || empty["latest_price"] != nil || empty["currency"] != nil {
		t.Fatalf("empty %v", empty)
	}
	for q, want := range map[string]int{"": 422, "?query=": 422, "?query=%20%20": 422, "?query=" + strings.Repeat("x", 256): 422} {
		path := "/api/v1/search" + q
		if q == "" {
			path = "/api/v1/search"
		}
		if r := e.do("GET", path, nil, alice.cookie); r.Code != want {
			t.Errorf("%q: %d", q, r.Code)
		}
	}
}

func TestStatistics(t *testing.T) {
	e := newEnv(t, true)
	alice := e.user(t, "alice", false)
	today := time.Now().UTC().Format("2006-01-02")
	e.seed(t, alice, receiptJSON("Shop", today, "10.00", "1", "10.00", "10.00", "grocery", "cash", "1.00"), "verified")
	e.seed(t, alice, receiptJSON("Shop", today, "20.00", "1", "20.00", "20.00", "fuel", "credit_card", "null"), "verified")
	e.seed(t, alice, receiptJSON("Skipped", today, "99.00", "1", "99.00", "99.00", "grocery", "cash", "null"), "unverified")

	r := e.do("GET", "/api/v1/statistics", nil, alice.cookie)
	if r.Code != 200 {
		t.Fatalf("%d %s", r.Code, r.Body)
	}
	j := r.json(t)
	if j["verified_receipt_count"] != float64(2) {
		t.Fatalf("%v", j)
	}
	cur := j["currencies"].([]any)[0].(map[string]any)
	for k, want := range map[string]any{"currency": "EUR", "receipt_count": float64(2), "total": "30.00", "minimum": "10.00", "maximum": "20.00", "tips": "1.00", "median": "15"} {
		if cur[k] != want {
			t.Errorf("%s = %v, want %v", k, cur[k], want)
		}
	}
	if !strings.HasPrefix(cur["average"].(string), "15.0") {
		t.Errorf("average %v", cur["average"])
	}
	merchants := j["merchants"].([]any)
	if len(merchants) != 1 || merchants[0].(map[string]any)["name"] != "Shop" || merchants[0].(map[string]any)["receipt_count"] != float64(2) {
		t.Fatalf("merchants %v", merchants)
	}
	if len(j["categories"].([]any)) != 2 || j["categories"].([]any)[0].(map[string]any)["name"] != "fuel" {
		t.Fatalf("categories %v", j["categories"])
	}
	if len(j["payment_methods"].([]any)) != 2 || len(j["weekdays"].([]any)) != 1 || len(j["weekly_spending"].([]any)) != 1 {
		t.Fatalf("%v", j)
	}
	wk := j["weekly_spending"].([]any)[0].(map[string]any)
	if wk["receipt_count"] != float64(2) || wk["total"] != "30.00" {
		t.Fatalf("weekly %v", wk)
	}
	for path, want := range map[string]int{"?weeks=0": 422, "?weeks=53": 422, "?weeks=x": 422, "?collection_id=nope": 422, "?weeks=52": 200} {
		if r := e.do("GET", "/api/v1/statistics"+path, nil, alice.cookie); r.Code != want {
			t.Errorf("%s: %d", path, r.Code)
		}
	}
	// Scope to a collection.
	cid := e.do("POST", "/api/v1/collections", `{"name": "Trip"}`, alice.cookie).json(t)["id"].(string)
	scoped := e.do("GET", "/api/v1/statistics?collection_id="+cid, nil, alice.cookie).json(t)
	if scoped["verified_receipt_count"] != float64(0) {
		t.Fatalf("empty collection: %v", scoped)
	}
}

func TestItemStatistics(t *testing.T) {
	e := newEnv(t, true)
	alice := e.user(t, "alice", false)
	today := time.Now().UTC().Format("2006-01-02")
	// Milk 6.00, a deposit 0.25 and an unknown item; the receipt total carries an extra 1.00 not on any line.
	body := `{"confidence": 90, "merchant_name": "Shop", "date": "` + today + `", "currency": "EUR", "category": "grocery",
	 "subtotal": "8.25", "total": "9.25", "tax_total": "0", "discount_total": "0",
	 "line_items": [
	  {"description": "Milk", "spending_category": "milk", "quantity": 2, "unit_price": "3.00", "total_price": "6.00"},
	  {"description": "Bottle deposit", "spending_category": "deposit", "quantity": 1, "unit_price": "0.25", "total_price": "0.25"},
	  {"description": "Thing", "spending_category": "unknown", "quantity": 1, "unit_price": "2.00", "total_price": "2.00"}]}`
	e.seed(t, alice, body, "verified")
	r := e.do("GET", "/api/v1/statistics/items?months=3", nil, alice.cookie)
	if r.Code != 200 {
		t.Fatalf("%d %s", r.Code, r.Body)
	}
	j := r.json(t)
	if j["months"] != float64(3) || j["date_to"] != today {
		t.Fatalf("%v", j)
	}
	cats := j["categories"].([]any)
	if len(cats) != 2 {
		t.Fatalf("deposits are separate: %v", cats)
	}
	milk := cats[0].(map[string]any)
	if milk["category"] != "milk" || milk["total"] != "6.00" || milk["share"] != "75.00" {
		t.Fatalf("milk %v", milk)
	}
	if cats[1].(map[string]any)["category"] != "unknown" || cats[1].(map[string]any)["share"] != "25.00" {
		t.Fatalf("unknown stays in the denominator: %v", cats[1])
	}
	if got := len(j["monthly"].([]any)); got != 6 { // 3 months x 2 categories
		t.Fatalf("monthly rows: %d", got)
	}
	rec := j["reconciliation"].([]any)[0].(map[string]any)
	for k, want := range map[string]string{"currency": "EUR", "receipt_total": "9.25", "item_total": "8.25", "deposit_total": "0.25", "unallocated_total": "1.00"} {
		if rec[k] != want {
			t.Errorf("%s = %v, want %s", k, rec[k], want)
		}
	}
	if r := e.do("GET", "/api/v1/statistics/items?months=37", nil, alice.cookie); r.Code != 422 {
		t.Fatalf("months limit: %d", r.Code)
	}
	if r := e.do("GET", "/api/v1/statistics/items", nil, e.user(t, "bob", false).cookie); r.json(t)["categories"] == nil || len(r.json(t)["categories"].([]any)) != 0 {
		t.Fatalf("scoped to caller: %s", r.Body)
	}
}
