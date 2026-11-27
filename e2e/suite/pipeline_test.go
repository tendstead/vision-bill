//go:build e2e

package suite

import (
	"bytes"
	"strconv"
	"testing"
	"time"

	"github.com/shopspring/decimal"
)

func dec(t *testing.T, s string) decimal.Decimal {
	t.Helper()
	d, err := decimal.NewFromString(s)
	if err != nil {
		t.Fatalf("not a decimal: %q", s)
	}
	return d
}

// TestPipelinePerFixture: upload -> LLM stub -> receipt -> verify, once per registered fixture.
func TestPipelinePerFixture(t *testing.T) {
	cs := cases(t)
	if len(cs) == 0 {
		t.Fatal("no e2e fixtures registered")
	}
	for _, fc := range cs {
		t.Run(fc.Name, func(t *testing.T) {
			newEnv(t)
			c, _ := register(t, "pipe")
			up := c.Upload("", fc.file(t)).Is(201, 202).M()
			imageID := str(up, "image_id")
			row := waitImage(t, c, imageID, true, "analyzed")
			receipt := receiptOf(t, c, row)

			got, want := mp(receipt, "receipt"), fc.Expected
			if str(got, "merchant_name") != want.MerchantName || str(got, "date") != want.Date.String() || str(got, "currency") != want.Currency {
				t.Fatalf("header mismatch: %v", got)
			}
			if !dec(t, str(got, "total")).Equal(want.Total) || !dec(t, str(got, "subtotal")).Equal(want.Subtotal) {
				t.Fatalf("totals mismatch: %v", got)
			}
			type item struct {
				desc          string
				qty           float64
				unit, totalPr string
			}
			var gotItems, wantItems []item
			for _, li := range list(receipt, "line_items") {
				m := li.(map[string]any)
				gotItems = append(gotItems, item{str(m, "description"), m["quantity"].(float64), dec(t, str(m, "unit_price")).String(), dec(t, str(m, "total_price")).String()})
			}
			for _, li := range want.LineItems {
				q, _ := li.Quantity.Float64()
				wantItems = append(wantItems, item{li.Description, q, li.UnitPrice.String(), li.TotalPrice.String()})
			}
			if len(gotItems) != len(wantItems) {
				t.Fatalf("items %v, want %v", gotItems, wantItems)
			}
			for _, w := range wantItems {
				found := false
				for _, g := range gotItems {
					if g == w {
						found = true
					}
				}
				if !found {
					t.Fatalf("item %v missing in %v", w, gotItems)
				}
			}

			// Verify marks the receipt, moves the image and a repeat conflicts.
			rid := str(row, "receipt_id")
			if got := c.Post("/receipts/"+rid+"/verify", nil).Is(200).M(); str(got, "status") != "verified" {
				t.Fatalf("%v", got)
			}
			c.Post("/receipts/"+rid+"/verify", nil).Is(409)
			if f := c.Get("/images/" + imageID + "/file").Is(200); !bytes.Equal(f.Body, readFile(t, fc.Image)) {
				t.Fatal("stored file differs from the upload")
			}
		})
	}
}

func TestBypassReviewVerifiesImmediately(t *testing.T) {
	newEnv(t)
	fc := firstCase(t)
	c, _ := register(t, "bypass")
	rid := str(c.Upload("?bypass_review=true", fc.file(t)).Is(201).M(), "receipt_id")
	detail := c.Get("/receipts/" + rid).Is(200).M()
	if str(detail, "receipt", "status") != "verified" || mp(detail, "receipt")["verified"] != true {
		t.Fatalf("%v", mp(detail, "receipt"))
	}
	c.Post("/receipts/"+rid+"/verify", nil).Is(409)
}

func TestReceiptCRUD(t *testing.T) {
	newEnv(t)
	c, _ := register(t, "crud")
	rid := str(analyzeFixture(t, c, firstCase(t)), "receipt_id")

	current := c.Get("/receipts/" + rid).Is(200).M()
	r := mp(current, "receipt")
	var items, taxes []any
	for _, li := range list(current, "line_items") {
		m := li.(map[string]any)
		items = append(items, map[string]any{"description": m["description"], "quantity": m["quantity"],
			"unit_price": m["unit_price"], "total_price": m["total_price"], "tags": m["tags"]})
	}
	for _, tx := range list(current, "taxes") {
		m := tx.(map[string]any)
		taxes = append(taxes, map[string]any{"name": m["name"], "rate": m["rate"], "amount": m["amount"]})
	}
	update := map[string]any{"confidence": r["confidence"], "merchant_name": "Renamed Merchant", "merchant_address": r["merchant_address"],
		"receipt_number": r["receipt_number"], "date": r["date"], "time": r["time"], "currency": r["currency"], "category": r["category"],
		"line_items": items, "taxes": taxes, "subtotal": r["subtotal"], "discount_total": r["discount_total"], "tax_total": r["tax_total"],
		"tip": r["tip"], "total": r["total"], "payment_method": r["payment_method"]}
	if got := c.Put("/receipts/"+rid, update).Is(200).M(); str(got, "merchant_name") != "Renamed Merchant" {
		t.Fatalf("%v", got)
	}
	if got := c.Delete("/receipts/" + rid).Is(200).M(); str(got, "deleted") != rid {
		t.Fatalf("%v", got)
	}
	c.Get("/receipts/" + rid).Is(404)
	c.Delete("/receipts/" + rid).Is(404)
}

func TestUsersCannotSeeEachOthersData(t *testing.T) {
	newEnv(t)
	fc := firstCase(t)
	alice, _ := register(t, "scope-a")
	bob, _ := register(t, "scope-b")
	rid := str(analyzeFixture(t, alice, fc), "receipt_id")

	bob.Get("/receipts/" + rid).Is(404)
	for _, v := range bob.Get("/receipts").Is(200).L() {
		if str(v, "id") == rid {
			t.Fatal("receipt listed for another user")
		}
	}
	bob.Post("/receipts/"+rid+"/verify", nil).Is(404)
	bob.Delete("/receipts/" + rid).Is(404)
	alice.Get("/receipts/" + rid).Is(200)
}

func TestAdminSeesAllInTheE2EStack(t *testing.T) {
	newEnv(t)
	carol, _ := register(t, "scope-c")
	rid := str(analyzeFixture(t, carol, firstCase(t)), "receipt_id")
	a := admin(t)
	found := false
	for _, v := range a.Get("/receipts").Is(200).L() {
		if str(v, "id") == rid {
			found = true
		}
	}
	if !found {
		t.Fatal("admin (can see all) does not list the receipt")
	}
	a.Get("/receipts/" + rid).Is(200)
}

func TestProviderDownKeepsImagePending(t *testing.T) {
	newEnv(t)
	fc := firstCase(t)
	c, _ := register(t, "errdown")
	stubMode(t, "down")
	r := c.Upload("", fc.file(t)).Is(202).M()
	if r["status"] != "pending" || r["warning"] == nil {
		t.Fatalf("%v", r)
	}
	id := str(r, "image_id")
	c.Post("/images/analyze", nil).Is(200) // a cycle with the model down must leave it queued
	row := c.Get("/images/" + id).Is(200).M()
	if row["status"] != "pending" || row["receipt_id"] != nil {
		t.Fatalf("%v", row)
	}
	// Once the model is back the queue drains.
	stubMode(t, "ok")
	waitImage(t, c, id, true, "analyzed")
}

func TestSelfCorrectionRetryRecovers(t *testing.T) {
	newEnv(t)
	fc := firstCase(t)
	c, _ := register(t, "repair")
	stubMode(t, "repair_first") // the first answer per image is broken JSON
	r := c.Upload("", fc.file(t)).Is(201, 202).M()
	row := waitImage(t, c, str(r, "image_id"), true, "analyzed", "failed")
	if row["status"] != "analyzed" {
		t.Fatalf("%v", row)
	}
	broken, good := 0, 0
	for _, req := range stubRequests(t) {
		if req["image_sha256"] == nil {
			continue
		}
		switch req["served"] {
		case "broken":
			broken++
		case "fixture", "generic":
			good++
		}
	}
	if broken < 1 || good < 1 {
		t.Fatalf("expected a broken first attempt and a corrected retry (broken=%d good=%d)", broken, good)
	}
}

func TestUnsupportedImageTypeIs415(t *testing.T) {
	newEnv(t)
	c, _ := register(t, "err415")
	r := c.Upload("", File{Name: "not-an-image.txt", ContentType: "text/plain", Data: []byte("hello")}).Is(415)
	if str(r.M(), "detail") != "Unsupported image type" {
		t.Fatal(r)
	}
}

func fixtureFromStub(t *testing.T, name string) []byte {
	t.Helper()
	return stubCall(t, "GET", "/__fixtures/"+name, nil).Is(200).Body
}

func TestMultiPhotoSubmissionIsProcessedWithoutAManualTrigger(t *testing.T) {
	for _, bypass := range []bool{false, true} {
		t.Run("bypass="+strconv.FormatBool(bypass), func(t *testing.T) {
			newEnv(t)
			c, _ := register(t, "multi")
			other, _ := register(t, "other-multi")
			photo := fixtureFromStub(t, "a")
			truth := stubCall(t, "GET", "/__fixtures/a.json", nil).Is(200).M()
			r := c.Upload("?bypass_review="+strconv.FormatBool(bypass),
				File{Name: "top.png", ContentType: "image/png", Data: photo}, File{Name: "bottom.png", ContentType: "image/png", Data: photo}).Is(202)
			imageURL := "/images/" + str(r.M(), "image_id")

			// No /analyze call: the upload itself wakes the worker, not the 5-minute timer.
			var job map[string]any
			waitUntil(t, 20*time.Second, "the worker to finish the submission", func() bool {
				job = c.Get(imageURL).Is(200).M()
				return job["status"] != "pending" && job["status"] != "processing"
			})
			if job["status"] != "analyzed" || len(list(job, "additional_images")) != 1 {
				t.Fatalf("%v", job)
			}
			other.Get(imageURL).Is(404)
			other.Get(imageURL + "/file?part=1").Is(404)
			receiptURL := "/receipts/" + str(job, "receipt_id")
			receipt := c.Get(receiptURL).Is(200).M()
			if len(list(receipt, "line_items")) != len(list(truth, "line_items")) {
				t.Fatalf("overlap rows not removed: %d items", len(list(receipt, "line_items")))
			}
			wantStatus := "unverified"
			if bypass {
				wantStatus = "verified"
			} else {
				c.Post(receiptURL+"/verify", nil).Is(200)
			}
			if str(receipt, "receipt", "status") != wantStatus {
				t.Fatalf("status %s", str(receipt, "receipt", "status"))
			}
			if !bytes.Equal(c.Get(imageURL+"/file?part=1").Is(200).Body, photo) {
				t.Fatal("second photo differs")
			}
			c.Delete(receiptURL).Is(200)
			c.Get(imageURL).Is(404)
		})
	}
}

func TestUnreadableIsTerminalAndSurvivesReload(t *testing.T) {
	for _, count := range []int{1, 2} {
		t.Run(strconv.Itoa(count)+" photos", func(t *testing.T) {
			newEnv(t)
			c, _ := register(t, "blur")
			photo := fixtureFromStub(t, "a")
			stubMode(t, "unreadable")
			var files []File
			for i := 0; i < count; i++ {
				files = append(files, File{Name: "part-" + strconv.Itoa(i) + ".png", ContentType: "image/png", Data: photo})
			}
			want := 422
			if count > 1 {
				want = 202
			}
			r := c.Upload("", files...).Is(want)
			imageURL := "/images/" + str(r.M(), "image_id")
			waitUntil(t, 20*time.Second, "status unreadable", func() bool { return str(c.Get(imageURL).Is(200).M(), "status") == "unreadable" })
			if c.Get(imageURL).M()["receipt_id"] != nil {
				t.Fatal("unreadable submissions have no receipt")
			}
			stubMode(t, "ok")
			c.Post("/images/analyze", nil)
			if str(c.Get(imageURL).Is(200).M(), "status") != "unreadable" {
				t.Fatal("unreadable is terminal")
			}
			c.Delete(imageURL).Is(200)
		})
	}
}

func TestLineItemsKeepTheirOrder(t *testing.T) {
	newEnv(t)
	c, _ := register(t, "order")
	r := c.Upload("", File{Name: "a.png", ContentType: "image/png", Data: fixtureFromStub(t, "a")}).Is(201, 202)
	row := waitImage(t, c, str(r.M(), "image_id"), true, "analyzed")
	receipt := receiptOf(t, c, row)
	var descs []string
	var positions []float64
	for _, li := range list(receipt, "line_items") {
		m := li.(map[string]any)
		descs = append(descs, str(m, "description"))
		positions = append(positions, m["position"].(float64))
	}
	if len(descs) != 2 || descs[0] != "Milk" || descs[1] != "Bread" || positions[0] != 0 || positions[1] != 1 {
		t.Fatalf("items %v positions %v", descs, positions)
	}
}

func TestDepositTagIsSeeded(t *testing.T) {
	newEnv(t)
	c, _ := register(t, "tags")
	found := false
	for _, v := range c.Get("/tags").Is(200).L() {
		if v == "deposit" {
			found = true
		}
	}
	if !found {
		t.Fatal("deposit tag missing from the vocabulary")
	}
}
