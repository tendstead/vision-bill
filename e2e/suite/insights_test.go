//go:build e2e

package suite

import (
	"strings"
	"testing"
	"time"
)

// verifiedFixtureReceipt analyses the first fixture and verifies the receipt.
func verifiedFixtureReceipt(t *testing.T, c *Client) map[string]any {
	t.Helper()
	row := analyzeFixture(t, c, firstCase(t))
	rid := str(row, "receipt_id")
	c.Post("/receipts/"+rid+"/verify", nil).Is(200)
	return c.Get("/receipts/" + rid).Is(200).M()
}

func TestSearchFindsVerifiedLineItems(t *testing.T) {
	newEnv(t)
	c, _ := register(t, "search")
	verifiedFixtureReceipt(t, c)
	desc := firstCase(t).Expected.LineItems[0].Description
	term := strings.Fields(desc)[0]
	body := c.Get("/search?query=" + term).Is(200).M()
	if str(body, "query") != term || body["latest_price"] == nil {
		t.Fatalf("%v", body)
	}
	found := false
	for _, p := range list(body, "purchases") {
		if str(p, "description") == desc {
			found = true
		}
	}
	if !found {
		t.Fatalf("%q not in %v", desc, body)
	}
}

func TestStatisticsReflectVerifiedReceipts(t *testing.T) {
	newEnv(t)
	c, _ := register(t, "stats")
	verifiedFixtureReceipt(t, c)
	fc := firstCase(t)
	stats := c.Get("/statistics").Is(200).M()
	if stats["verified_receipt_count"] != float64(1) {
		t.Fatalf("%v", stats)
	}
	var cur map[string]any
	for _, v := range list(stats, "currencies") {
		if str(v, "currency") == fc.Expected.Currency {
			cur = v.(map[string]any)
		}
	}
	if cur == nil || cur["receipt_count"] != float64(1) || !dec(t, str(cur, "total")).Equal(fc.Expected.Total) {
		t.Fatalf("%v", stats)
	}
	merchant := false
	for _, m := range list(stats, "merchants") {
		if str(m, "name") == fc.Expected.MerchantName {
			merchant = true
		}
	}
	if !merchant {
		t.Fatal("merchant missing from statistics")
	}
}

// saveReceipt writes lines (and field updates) back through PUT.
func saveReceipt(t *testing.T, c *Client, details map[string]any, lines []any, updates map[string]any) map[string]any {
	t.Helper()
	body := map[string]any{}
	for k, v := range mp(details, "receipt") {
		body[k] = v
	}
	body["line_items"], body["taxes"] = lines, list(details, "taxes")
	for k, v := range updates {
		body[k] = v
	}
	id := str(details, "receipt", "id")
	c.Put("/receipts/"+id, body).Is(200)
	return c.Get("/receipts/" + id).Is(200).M()
}

func TestConfirmedCategoryHistorySurvivesRenamesAndIsPrivate(t *testing.T) {
	newEnv(t)
	alice, _ := register(t, "category-alice")
	bob, _ := register(t, "category-bob")
	fc := firstCase(t)
	details := receiptOf(t, alice, analyzeFixture(t, alice, fc))
	first := list(details, "line_items")[0].(map[string]any)
	original, merchant := str(first, "description"), str(details, "receipt", "merchant_name")
	first["description"], first["spending_category"], first["remember_category"] = "My breakfast item", "cereals", true
	saved := saveReceipt(t, alice, details, list(details, "line_items"), nil)
	li := list(saved, "line_items")[0].(map[string]any)
	if str(li, "original_description") != original || str(li, "category_source") != "user" {
		t.Fatalf("%v", li)
	}
	for _, d := range []string{original, "My breakfast item"} {
		q := "/tags/category-suggestion?merchant_name=" + escape(merchant) + "&description=" + escape(d)
		own := alice.Get(q).Is(200).M()
		if str(own, "category") != "cereals" || str(own, "source") != "history" {
			t.Fatalf("own suggestion %v", own)
		}
		if bob.Get(q).Is(200).M()["category"] != nil {
			t.Fatal("history leaked to another user")
		}
	}
	repeated := receiptOf(t, alice, analyzeFixture(t, alice, fc))
	rl := list(repeated, "line_items")[0].(map[string]any)
	if str(rl, "spending_category") != "cereals" || str(rl, "category_source") != "history" {
		t.Fatalf("%v", rl)
	}
	// A normal save never turns an unconfirmed category into a mapping.
	rl["spending_category"] = "milk"
	saveReceipt(t, alice, repeated, []any{rl}, nil)
	q := "/tags/category-suggestion?merchant_name=" + escape(merchant) + "&description=" + escape(original)
	if str(alice.Get(q).Is(200).M(), "category") != "cereals" {
		t.Fatal("remembered mapping changed by an ordinary save")
	}
	stranger := receiptOf(t, bob, analyzeFixture(t, bob, fc))
	if str(list(stranger, "line_items")[0], "category_source") == "history" {
		t.Fatal("another user's history was applied")
	}
}

func TestItemStatisticsScopingDepositsAndCollections(t *testing.T) {
	newEnv(t)
	alice, _ := register(t, "item-stats")
	bob, _ := register(t, "item-stats-other")
	today := time.Now().UTC()
	details := receiptOf(t, alice, analyzeFixture(t, alice, firstCase(t)))
	line := func(desc, amount, cat string) any {
		return map[string]any{"description": desc, "quantity": 1, "unit_price": amount, "total_price": amount, "spending_category": cat}
	}
	lines := []any{line("Milk", "6", "milk"), line("Unclear", "4", "unknown"), line("Pfand", "0.25", "deposit")}
	details = saveReceipt(t, alice, details, lines, map[string]any{"date": today.Format("2006-01-02"),
		"total": "9.25", "subtotal": "10.25", "discount_total": "1", "tax_total": "0", "taxes": []any{}})
	rid := str(details, "receipt", "id")

	if n := len(list(alice.Get("/statistics/items").Is(200).M(), "categories")); n != 0 {
		t.Fatalf("unverified receipts must not count (%d categories)", n)
	}
	alice.Post("/receipts/"+rid+"/verify", nil).Is(200)
	report := alice.Get("/statistics/items?months=3").Is(200).M()
	cats := map[string]map[string]any{}
	for _, c := range list(report, "categories") {
		cats[str(c, "category")] = c.(map[string]any)
	}
	if !dec(t, str(cats["milk"], "share")).Equal(dec(t, "60")) || !dec(t, str(cats["unknown"], "share")).Equal(dec(t, "40")) || cats["deposit"] != nil {
		t.Fatalf("%v", cats)
	}
	if len(list(report, "monthly")) != 6 {
		t.Fatalf("monthly rows: %d", len(list(report, "monthly")))
	}
	rec := list(report, "reconciliation")[0]
	if !dec(t, str(rec, "deposit_total")).Equal(dec(t, "0.25")) || !dec(t, str(rec, "unallocated_total")).Equal(dec(t, "-1")) {
		t.Fatalf("%v", rec)
	}
	if n := len(list(bob.Get("/statistics/items").Is(200).M(), "categories")); n != 0 {
		t.Fatal("statistics leaked to another user")
	}

	cid := str(alice.Post("/collections", map[string]string{"name": "Category test"}).Is(201).M(), "id")
	scope := "/statistics/items?collection_id=" + cid
	if n := len(list(alice.Get(scope).Is(200).M(), "categories")); n != 0 {
		t.Fatal("empty collection must have no categories")
	}
	details = saveReceipt(t, alice, details, list(details, "line_items"), map[string]any{"collection_ids": []string{cid}})
	scoped := alice.Get(scope).Is(200).M()
	if len(list(scoped, "categories")) != len(list(report, "categories")) {
		t.Fatalf("collection scope: %v", scoped)
	}
	if n := len(list(bob.Get(scope).Is(200).M(), "categories")); n != 0 {
		t.Fatal("collection statistics leaked to another user")
	}
	// A future-dated receipt does not inflate the current month.
	saveReceipt(t, alice, details, list(details, "line_items"), map[string]any{"date": today.AddDate(0, 0, 1).Format("2006-01-02")})
	if n := len(list(alice.Get("/statistics/items").Is(200).M(), "categories")); n != 0 {
		t.Fatal("future-dated receipt counted")
	}
}

func TestCategoryEndpointsRequireAuthentication(t *testing.T) {
	c := newEnv(t)
	for _, p := range []string{"/tags/categories", "/tags/category-suggestion?merchant_name=X&description=Milk", "/statistics/items"} {
		c.Get(p).Is(401)
	}
}
