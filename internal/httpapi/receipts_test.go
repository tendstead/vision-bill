package httpapi

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/tendstead/vision-bill/internal/domain"
)

type actor struct {
	id     string
	cookie *http.Cookie
}

func (e *testEnv) user(t *testing.T, name string, admin bool) actor {
	t.Helper()
	u, err := e.st.CreateUser(context.Background(), name, "x", admin)
	if err != nil {
		t.Fatal(err)
	}
	return actor{u.ID, &http.Cookie{Name: "vb_session", Value: mustToken(u.ID)}}
}

const receiptBody = `{
 "confidence": 90, "merchant_name": "Shop", "date": "2025-03-04", "time": "12:30",
 "currency": "EUR", "category": "grocery", "payment_method": "cash",
 "subtotal": "3.00", "total": 3.57, "tax_total": 0.57, "discount_total": "0",
 "line_items": [{"description": "Milk", "spending_category": "milk", "quantity": 2, "unit_price": "1.5", "total_price": 3, "tags": ["Dairy"]}],
 "taxes": [{"name": "VAT", "rate": 0.19, "amount": "0.57"}]
}`

func (e *testEnv) seed(t *testing.T, owner actor, body string, status string) string {
	t.Helper()
	r, issues := domain.ParseReceipt([]byte(body), domain.Options{})
	if issues != nil {
		t.Fatal(issues)
	}
	row, err := e.st.PersistReceipt(context.Background(), r, nil, status, status == "verified", &owner.id)
	if err != nil {
		t.Fatal(err)
	}
	return row.ID
}

func TestReceiptReadEndpoints(t *testing.T) {
	e := newEnv(t, true)
	alice, bob := e.user(t, "alice", false), e.user(t, "bob", false)
	id := e.seed(t, alice, receiptBody, "unverified")

	if r := e.do("GET", "/api/v1/receipts", nil); r.Code != 401 {
		t.Fatalf("anon: %d", r.Code)
	}
	list := e.do("GET", "/api/v1/receipts", nil, alice.cookie)
	if list.Code != 200 || !strings.Contains(list.Body.String(), id) {
		t.Fatalf("list: %d %s", list.Code, list.Body)
	}
	if got := e.do("GET", "/api/v1/receipts", nil, bob.cookie).Body.String(); strings.TrimSpace(got) != "[]" {
		t.Fatalf("bob sees %s", got)
	}
	if r := e.do("GET", "/api/v1/receipts/"+id, nil, bob.cookie); r.Code != 404 || r.json(t)["detail"] != "Receipt not found" {
		t.Fatalf("foreign detail: %d %s", r.Code, r.Body)
	}
	d := e.do("GET", "/api/v1/receipts/"+id, nil, alice.cookie)
	if d.Code != 200 {
		t.Fatalf("detail %d %s", d.Code, d.Body)
	}
	body := d.Body.String()
	for _, want := range []string{`"subtotal":"3.00"`, `"total":"3.57"`, `"time":"12:30:00"`, `"quantity":2.0`, `"rate":0.19`,
		`"unit_price":"1.50"`, `"tags":["Dairy"]`, `"collection_ids":[]`, `"image_path":null`, `"original_description":"Milk"`, `"category_source":"llm"`} {
		if !strings.Contains(body, want) {
			t.Errorf("detail missing %s in %s", want, body)
		}
	}
	// validation
	for path, want := range map[string]string{
		"/api/v1/receipts?limit=0":            "greater_than_equal",
		"/api/v1/receipts?limit=501":          "less_than_equal",
		"/api/v1/receipts?offset=-1":          "greater_than_equal",
		"/api/v1/receipts?limit=abc":          "int_parsing",
		"/api/v1/receipts?date_from=zzz":      "date_from_datetime_parsing",
		"/api/v1/receipts?collection_id=nope": "uuid_parsing",
		"/api/v1/receipts/not-a-uuid":         "uuid_parsing",
	} {
		r := e.do("GET", path, nil, alice.cookie)
		if r.Code != 422 || !strings.Contains(r.Body.String(), want) {
			t.Errorf("%s: %d %s", path, r.Code, r.Body)
		}
	}
	// filters
	for q, n := range map[string]int{"status=verified": 0, "status=unverified,verified": 1, "search=sho": 1, "search=nomatch": 0,
		"date_from=2025-03-05": 0, "date_to=2025-03-04": 1} {
		r := e.do("GET", "/api/v1/receipts?"+q, nil, alice.cookie)
		if got := strings.Count(r.Body.String(), `"merchant_name"`); got != n {
			t.Errorf("%s: %d results, want %d", q, got, n)
		}
	}
}

func TestAdminSeesAllOnlyWhenEnabled(t *testing.T) {
	e := newEnv(t, true)
	alice, admin := e.user(t, "alice", false), e.user(t, "adm", true)
	id := e.seed(t, alice, receiptBody, "unverified")
	if r := e.do("GET", "/api/v1/receipts/"+id, nil, admin.cookie); r.Code != 404 {
		t.Fatalf("admin without flag: %d", r.Code)
	}
	_ = e.cfg.Update(func(c *configT) error { c.Auth.AdminCanSeeAll = true; return nil })
	if r := e.do("GET", "/api/v1/receipts/"+id, nil, admin.cookie); r.Code != 200 {
		t.Fatalf("admin with flag: %d", r.Code)
	}
}

func putBody(mod func(string) string) string { return mod(receiptBody) }

func TestUpdateReceiptRoundingAndQuirk(t *testing.T) {
	e := newEnv(t, true)
	alice := e.user(t, "alice", false)
	id := e.seed(t, alice, receiptBody, "unverified")

	// Exact decimal rounding (approved difference): 1.005 -> 1.01, not 1.00.
	body := strings.Replace(receiptBody, `"subtotal": "3.00"`, `"subtotal": "1.005"`, 1)
	r := e.do("PUT", "/api/v1/receipts/"+id, body, alice.cookie)
	if r.Code != 200 || r.json(t)["subtotal"] != "1.01" {
		t.Fatalf("put: %d %s", r.Code, r.Body)
	}

	// Baseline quirk: the edit commits before collection validation.
	missing := "99999999-9999-9999-9999-999999999999"
	body = strings.Replace(receiptBody, `"merchant_name": "Shop"`, `"merchant_name": "Renamed", "collection_ids": ["`+missing+`"]`, 1)
	r = e.do("PUT", "/api/v1/receipts/"+id, body, alice.cookie)
	if r.Code != 404 || r.json(t)["detail"] != "Collection not found" {
		t.Fatalf("bad collection: %d %s", r.Code, r.Body)
	}
	if got := e.do("GET", "/api/v1/receipts/"+id, nil, alice.cookie).json(t)["receipt"].(map[string]any)["merchant_name"]; got != "Renamed" {
		t.Fatalf("edit should persist, got %v", got)
	}

	// Errors: unknown receipt, validation, bad time, foreign id.
	if r := e.do("PUT", "/api/v1/receipts/"+missing, receiptBody, alice.cookie); r.Code != 404 || r.json(t)["detail"] != "Receipt not found" {
		t.Fatalf("unknown: %d %s", r.Code, r.Body)
	}
	if r := e.do("PUT", "/api/v1/receipts/"+id, `{"confidence": 1}`, alice.cookie); r.Code != 422 {
		t.Fatalf("invalid: %d", r.Code)
	}
	bad := strings.Replace(receiptBody, `"time": "12:30"`, `"time": "noon"`, 1)
	if r := e.do("PUT", "/api/v1/receipts/"+id, bad, alice.cookie); r.Code != 422 || r.json(t)["detail"] != "Invalid isoformat string: 'noon'" {
		t.Fatalf("bad time: %d %s", r.Code, r.Body)
	}
	foreign := strings.Replace(receiptBody, `{"description": "Milk"`, `{"id": "`+missing+`", "description": "Milk"`, 1)
	if r := e.do("PUT", "/api/v1/receipts/"+id, foreign, alice.cookie); r.Code != 422 || r.json(t)["detail"] != "Line item id does not belong to this receipt" {
		t.Fatalf("foreign line id: %d %s", r.Code, r.Body)
	}
}

func TestUpdatePreservesCategoryDecisionsAndHistory(t *testing.T) {
	e := newEnv(t, true)
	alice := e.user(t, "alice", false)
	id := e.seed(t, alice, receiptBody, "unverified")
	d := e.do("GET", "/api/v1/receipts/"+id, nil, alice.cookie).json(t)
	item := d["line_items"].([]any)[0].(map[string]any)

	// Changing the category with remember_category stores user source + alias.
	body := strings.Replace(receiptBody, `{"description": "Milk", "spending_category": "milk"`,
		`{"id": "`+item["id"].(string)+`", "remember_category": true, "description": "Milk", "spending_category": "cheese"`, 1)
	if r := e.do("PUT", "/api/v1/receipts/"+id, body, alice.cookie); r.Code != 200 {
		t.Fatalf("%d %s", r.Code, r.Body)
	}
	item = e.do("GET", "/api/v1/receipts/"+id, nil, alice.cookie).json(t)["line_items"].([]any)[0].(map[string]any)
	if item["spending_category"] != "cheese" || item["category_source"] != "user" || item["id"] == nil {
		t.Fatalf("item %v", item)
	}
	r := e.do("GET", "/api/v1/tags/category-suggestion?merchant_name=SHOP&description=milk", nil, alice.cookie)
	if j := r.json(t); j["category"] != "cheese" || j["source"] != "history" {
		t.Fatalf("suggestion %s", r.Body)
	}
	// A new receipt with the same merchant/description takes the remembered category.
	id2 := e.seed(t, alice, receiptBody, "unverified")
	item2 := e.do("GET", "/api/v1/receipts/"+id2, nil, alice.cookie).json(t)["line_items"].([]any)[0].(map[string]any)
	if item2["spending_category"] != "cheese" || item2["category_source"] != "history" {
		t.Fatalf("history not applied: %v", item2)
	}
}

func TestCollectionsAndTags(t *testing.T) {
	e := newEnv(t, true)
	alice, bob := e.user(t, "alice", false), e.user(t, "bob", false)
	rid := e.seed(t, alice, receiptBody, "verified")

	if r := e.do("GET", "/api/v1/collections/active", nil, alice.cookie); r.Code != 204 || r.Body.Len() != 0 {
		t.Fatalf("active none: %d %q", r.Code, r.Body)
	}
	c := e.do("POST", "/api/v1/collections", `{"name": "  Trip   2025 ", "color": "#abc", "start_date": "2025-01-01", "end_date": "2025-02-01", "active": true}`, alice.cookie)
	if c.Code != 201 {
		t.Fatalf("create: %d %s", c.Code, c.Body)
	}
	cj := c.json(t)
	cid := cj["id"].(string)
	if cj["name"] != "Trip 2025" || cj["active"] != true || cj["start_date"] != "2025-01-01" {
		t.Fatalf("created %v", cj)
	}
	for body, want := range map[string]int{
		`{"name": "trip 2025"}`:         409,
		`{"name": "   "}`:               422,
		`{"name": "x", "color": "red"}`: 422,
		`{"name": "x", "start_date": "2025-05-01", "end_date": "2025-01-01"}`: 422,
		`{}`: 422,
	} {
		if r := e.do("POST", "/api/v1/collections", body, alice.cookie); r.Code != want {
			t.Errorf("%s: %d %s", body, r.Code, r.Body)
		}
	}
	if r := e.do("POST", "/api/v1/collections/"+cid+"/receipts/"+rid, nil, alice.cookie); r.Code != 200 || r.json(t)["assigned"] != true {
		t.Fatalf("assign: %d %s", r.Code, r.Body)
	}
	if r := e.do("POST", "/api/v1/collections/"+cid+"/receipts/"+rid, nil, bob.cookie); r.Code != 404 {
		t.Fatalf("bob assign: %d", r.Code)
	}
	list := e.do("GET", "/api/v1/collections", nil, alice.cookie)
	if !strings.Contains(list.Body.String(), `"receipt_count":1`) || !strings.Contains(list.Body.String(), `"totals":[{"currency":"EUR","total":"3.57"}]`) {
		t.Fatalf("list: %s", list.Body)
	}
	det := e.do("GET", "/api/v1/collections/"+cid, nil, alice.cookie)
	if det.Code != 200 || !strings.Contains(det.Body.String(), rid) {
		t.Fatalf("detail: %d %s", det.Code, det.Body)
	}
	if r := e.do("GET", "/api/v1/collections/"+cid, nil, bob.cookie); r.Code != 404 {
		t.Fatalf("bob detail: %d", r.Code)
	}
	if r := e.do("GET", "/api/v1/receipts/"+rid, nil, alice.cookie); !strings.Contains(r.Body.String(), `"collection_ids":["`+cid+`"]`) {
		t.Fatalf("receipt collections: %s", r.Body)
	}
	if r := e.do("GET", "/api/v1/receipts?collection_id="+cid, nil, alice.cookie); !strings.Contains(r.Body.String(), rid) {
		t.Fatalf("filter: %s", r.Body)
	}
	// second active collection deactivates the first
	c2 := e.do("POST", "/api/v1/collections", `{"name": "Other"}`, alice.cookie).json(t)["id"].(string)
	if r := e.do("POST", "/api/v1/collections/"+c2+"/activate", nil, alice.cookie); r.Code != 200 || r.json(t)["active"] != true {
		t.Fatalf("activate: %d %s", r.Code, r.Body)
	}
	if r := e.do("GET", "/api/v1/collections/active", nil, alice.cookie); r.json(t)["id"] != c2 {
		t.Fatalf("active: %s", r.Body)
	}
	if r := e.do("POST", "/api/v1/collections/"+c2+"/deactivate", nil, alice.cookie); r.json(t)["active"] != false {
		t.Fatalf("deactivate: %s", r.Body)
	}
	if r := e.do("PATCH", "/api/v1/collections/"+cid, `{"name": "Renamed", "color": "#123456"}`, alice.cookie); r.Code != 200 || r.json(t)["name"] != "Renamed" {
		t.Fatalf("patch: %d %s", r.Code, r.Body)
	}
	if r := e.do("PATCH", "/api/v1/collections/"+cid, `{"name": "Other"}`, alice.cookie); r.Code != 409 {
		t.Fatalf("patch conflict: %d", r.Code)
	}
	if r := e.do("DELETE", "/api/v1/collections/"+cid+"/receipts/"+rid, nil, alice.cookie); r.Code != 204 {
		t.Fatalf("unassign: %d", r.Code)
	}
	if r := e.do("DELETE", "/api/v1/collections/"+cid+"/receipts/"+rid, nil, alice.cookie); r.Code != 404 || r.json(t)["detail"] != "Not assigned" {
		t.Fatalf("unassign twice: %d %s", r.Code, r.Body)
	}
	if r := e.do("DELETE", "/api/v1/collections/"+cid, nil, alice.cookie); r.Code != 204 {
		t.Fatalf("delete: %d", r.Code)
	}
	if r := e.do("DELETE", "/api/v1/collections/"+cid, nil, alice.cookie); r.Code != 404 {
		t.Fatalf("delete twice: %d", r.Code)
	}

	// tags
	if r := e.do("GET", "/api/v1/tags", nil, alice.cookie); r.Code != 200 || !strings.Contains(r.Body.String(), `"deposit"`) {
		t.Fatalf("tags: %s", r.Body)
	}
	if r := e.do("POST", "/api/v1/tags", `{"name": "  New   TAG "}`, alice.cookie); r.Code != 201 || r.json(t)["name"] != "new tag" || r.json(t)["created"] != true {
		t.Fatalf("create tag: %d %s", r.Code, r.Body)
	}
	if r := e.do("POST", "/api/v1/tags", `{"name": "new tag"}`, alice.cookie); r.Code != 200 || r.json(t)["created"] != false {
		t.Fatalf("existing tag: %d %s", r.Code, r.Body)
	}
	if r := e.do("POST", "/api/v1/tags", `{"name": "   "}`, alice.cookie); r.Code != 422 || r.json(t)["detail"] != "Tag name must not be blank" {
		t.Fatalf("blank tag: %d %s", r.Code, r.Body)
	}
	if r := e.do("GET", "/api/v1/tags/categories", nil, alice.cookie); !strings.HasPrefix(r.Body.String(), `[{"code":"unknown"},{"code":"other"}`) {
		t.Fatalf("categories: %s", r.Body)
	}
}
