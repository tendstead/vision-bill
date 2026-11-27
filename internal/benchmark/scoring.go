// Package benchmark scores extraction quality and executes durable
// benchmark runs over verified receipts. It never writes to receipts, line
// items, taxes or images.
package benchmark

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strconv"
	"strings"

	"github.com/shopspring/decimal"
	"golang.org/x/text/unicode/norm"

	"github.com/tendstead/vision-bill/internal/domain"
)

// Versions recorded with every run.
const (
	PromptVersion        = "4"
	LegacyScoringVersion = "2"
	DetailScoringVersion = "3"
)

var weights = []struct {
	name string
	w    float64
}{{"header", 0.20}, {"totals", 0.35}, {"line_items", 0.30}, {"taxes", 0.15}}

// ── receipt dump (JSON shape stored in manifests and results) ────────────

type itemDump struct {
	Description      string   `json:"description"`
	SpendingCategory string   `json:"spending_category"`
	Quantity         float64  `json:"quantity"`
	UnitPrice        string   `json:"unit_price"`
	TotalPrice       string   `json:"total_price"`
	Tags             []string `json:"tags"`
}

type taxDump struct {
	Name   string   `json:"name"`
	Rate   *float64 `json:"rate"`
	Amount string   `json:"amount"`
}

// ReceiptDump is the JSON representation of a receipt.
type ReceiptDump struct {
	Confidence      int        `json:"confidence"`
	MerchantName    string     `json:"merchant_name"`
	MerchantAddress *string    `json:"merchant_address"`
	ReceiptNumber   *string    `json:"receipt_number"`
	Date            string     `json:"date"`
	Time            *string    `json:"time"`
	Currency        string     `json:"currency"`
	Category        string     `json:"category"`
	LineItems       []itemDump `json:"line_items"`
	Taxes           []taxDump  `json:"taxes"`
	Subtotal        string     `json:"subtotal"`
	DiscountTotal   string     `json:"discount_total"`
	TaxTotal        string     `json:"tax_total"`
	Tip             *string    `json:"tip"`
	Total           string     `json:"total"`
	PaymentMethod   string     `json:"payment_method"`
}

// decText prints a decimal keeping its scale ("1.50" stays "1.50").
func decText(d decimal.Decimal) string {
	if e := d.Exponent(); e < 0 {
		return d.StringFixed(-e)
	}
	return d.String()
}

func floatOf(d decimal.Decimal) float64 {
	f, _ := d.Float64()
	return f
}

// Dump converts a receipt to its JSON shape.
func Dump(r *domain.Receipt) ReceiptDump {
	d := ReceiptDump{
		Confidence: r.Confidence, MerchantName: r.MerchantName, MerchantAddress: r.MerchantAddress, ReceiptNumber: r.ReceiptNumber,
		Date: r.Date.String(), Time: r.Time, Currency: r.Currency, Category: r.Category,
		LineItems: []itemDump{}, Taxes: []taxDump{},
		Subtotal: decText(r.Subtotal), DiscountTotal: decText(r.DiscountTotal), TaxTotal: decText(r.TaxTotal),
		Total: decText(r.Total), PaymentMethod: r.PaymentMethod,
	}
	if r.Tip != nil {
		s := decText(*r.Tip)
		d.Tip = &s
	}
	for _, it := range r.LineItems {
		tags := it.Tags
		if tags == nil {
			tags = []string{}
		}
		d.LineItems = append(d.LineItems, itemDump{it.Description, it.SpendingCategory, floatOf(it.Quantity), decText(it.UnitPrice), decText(it.TotalPrice), tags})
	}
	for _, t := range r.Taxes {
		td := taxDump{Name: t.Name, Amount: decText(t.Amount)}
		if t.Rate != nil {
			f := floatOf(*t.Rate)
			td.Rate = &f
		}
		d.Taxes = append(d.Taxes, td)
	}
	return d
}

// DatasetFingerprint is a stable hash of the expected (selected) data set,
// independent of the order of receipts and of their confidence.
func DatasetFingerprint(receipts []*domain.Receipt) string {
	rows := make([]string, len(receipts))
	for i, r := range receipts {
		d := Dump(r)
		d.Confidence = 0
		b, _ := json.Marshal(d)
		rows[i] = string(b)
	}
	sort.Strings(rows)
	sum := sha256.Sum256([]byte(strings.Join(rows, "\n")))
	return hex.EncodeToString(sum[:])
}

// ── legacy scoring (runs created without a manifest) ─────────────────────

var moneyTolerance = decimal.RequireFromString("0.01")

func normMoney(d decimal.Decimal) decimal.Decimal { return d.Round(2) }

// moneyEqual treats two absent amounts as equal and compares present ones
// within one cent.
func moneyEqual(a, b *decimal.Decimal) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return normMoney(*a).Sub(normMoney(*b)).Abs().LessThanOrEqual(moneyTolerance)
}

func fraction(matches, total int) float64 {
	if total == 0 {
		return 1
	}
	return float64(matches) / float64(total)
}

func ptrEq(a, b *string) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return *a == *b
}

func dp(d decimal.Decimal) *decimal.Decimal { return &d }

// ScoreReceipts is the legacy per-field comparison; it returns the score
// components including "overall".
func ScoreReceipts(expected, actual *domain.Receipt) map[string]float64 {
	header := 0
	for _, eq := range []bool{
		expected.MerchantName == actual.MerchantName, ptrEq(expected.MerchantAddress, actual.MerchantAddress),
		ptrEq(expected.ReceiptNumber, actual.ReceiptNumber), expected.Date.String() == actual.Date.String(),
		ptrEq(expected.Time, actual.Time), expected.Currency == actual.Currency,
		expected.Category == actual.Category, expected.PaymentMethod == actual.PaymentMethod,
	} {
		if eq {
			header++
		}
	}
	totalsMatch := 0
	for _, p := range [][2]*decimal.Decimal{
		{dp(expected.Subtotal), dp(actual.Subtotal)}, {dp(expected.DiscountTotal), dp(actual.DiscountTotal)},
		{dp(expected.TaxTotal), dp(actual.TaxTotal)}, {expected.Tip, actual.Tip}, {dp(expected.Total), dp(actual.Total)},
	} {
		if moneyEqual(p[0], p[1]) {
			totalsMatch++
		}
	}
	ei, ai := expected.LineItems, actual.LineItems
	itemMatches := 0
	for i := 0; i < len(ei) && i < len(ai); i++ {
		if ei[i].Description == ai[i].Description && moneyEqual(dp(ei[i].TotalPrice), dp(ai[i].TotalPrice)) {
			itemMatches++
		}
	}
	labelled, catMatches := 0, 0
	for i, it := range ei {
		if it.SpendingCategory == "unknown" {
			continue
		}
		labelled++
		if i < len(ai) && it.Description == ai[i].Description && it.SpendingCategory == ai[i].SpendingCategory {
			catMatches++
		}
	}
	et, at := expected.Taxes, actual.Taxes
	taxMatches := 0
	for i := 0; i < len(et) && i < len(at); i++ {
		if et[i].Name == at[i].Name && moneyEqual(dp(et[i].Amount), dp(at[i].Amount)) {
			taxMatches++
		}
	}
	c := map[string]float64{
		"header":     fraction(header, 8),
		"totals":     fraction(totalsMatch, 5),
		"line_items": fraction(itemMatches, max(len(ei), len(ai))),
		"taxes":      fraction(taxMatches, max(len(et), len(at))),
	}
	if labelled > 0 {
		c["spending_categories"] = fraction(catMatches, labelled)
	}
	overall := 0.0
	for _, w := range weights {
		overall += c[w.name] * w.w
	}
	c["overall"] = overall
	return c
}

// ── detailed scoring ─────────────────────────────────────────────────────

type difference struct {
	Path       string `json:"path"`
	Expected   any    `json:"expected"`
	Actual     any    `json:"actual"`
	Equivalent bool   `json:"equivalent"`
}

// Detail is the versioned diagnostic result of comparing two extractions.
type Detail struct {
	Version     string         `json:"version"`
	Scores      map[string]any `json:"scores"`
	Differences []difference   `json:"differences"`
	Rows        map[string]int `json:"rows"`
	Checks      map[string]any `json:"checks"`
}

func normalizeText(v any) any {
	if s, ok := v.(string); ok {
		return strings.Join(strings.Fields(norm.NFC.String(s)), " ")
	}
	return v
}

func asDecimal(v any) (decimal.Decimal, bool) {
	switch t := v.(type) {
	case string:
		d, err := decimal.NewFromString(strings.TrimSpace(t))
		return d, err == nil
	case float64:
		d, err := decimal.NewFromString(strconv.FormatFloat(t, 'f', -1, 64))
		return d, err == nil
	}
	return decimal.Decimal{}, false
}

// timeEqual compares ISO times by value ("12:30" equals "12:30:00").
func timeEqual(a, b string) (bool, bool) {
	x, err1 := domain.ParseTimeOfDay(a)
	y, err2 := domain.ParseTimeOfDay(b)
	if err1 != nil || err2 != nil {
		return false, false
	}
	return x == y, true
}

func fractionBools(v []bool) float64 {
	if len(v) == 0 {
		return 1
	}
	n := 0
	for _, b := range v {
		if b {
			n++
		}
	}
	return float64(n) / float64(len(v))
}

func jsonValue(v any) any {
	// Round-trip through JSON so struct values compare like decoded documents.
	b, _ := json.Marshal(v)
	var out any
	_ = json.Unmarshal(b, &out)
	return out
}

func equalValues(a, b any) bool {
	return jsonEqual(jsonValue(a), jsonValue(b))
}

func jsonEqual(a, b any) bool {
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return string(x) == string(y)
}

// DetailedScore compares an extraction with the expected receipt. Decimal
// formatting differences ("12.9" vs "12.90") are equivalent; a one-cent
// error is not forgiven.
func DetailedScore(expected, actual *domain.Receipt, classification bool) Detail {
	left, right := Dump(expected), Dump(actual)
	diffs := []difference{}
	groups := []string{"header", "totals", "line_items", "taxes"}
	strict := map[string][]bool{}
	normal := map[string][]bool{}
	var labels []bool

	field := func(group, path string, e, a any, numeric bool) {
		exact := equalValues(e, a)
		var equal bool
		ev, av := jsonValue(e), jsonValue(a)
		if numeric && ev != nil && av != nil {
			d1, ok1 := asDecimal(ev)
			d2, ok2 := asDecimal(av)
			equal = ok1 && ok2 && d1.Equal(d2)
		} else {
			equal = jsonEqual(normalizeText(ev), normalizeText(av))
		}
		if path == "time" {
			if es, ok := ev.(string); ok {
				if as, ok := av.(string); ok {
					if eq, parsed := timeEqual(es, as); parsed {
						equal = eq
					}
				}
			}
		}
		strict[group] = append(strict[group], exact)
		normal[group] = append(normal[group], equal)
		if !exact {
			diffs = append(diffs, difference{path, ev, av, equal})
		}
	}

	field("header", "merchant_name", left.MerchantName, right.MerchantName, false)
	field("header", "merchant_address", left.MerchantAddress, right.MerchantAddress, false)
	field("header", "receipt_number", left.ReceiptNumber, right.ReceiptNumber, false)
	field("header", "date", left.Date, right.Date, false)
	field("header", "time", left.Time, right.Time, false)
	field("header", "currency", left.Currency, right.Currency, false)
	field("header", "payment_method", left.PaymentMethod, right.PaymentMethod, false)
	field("totals", "subtotal", left.Subtotal, right.Subtotal, true)
	field("totals", "discount_total", left.DiscountTotal, right.DiscountTotal, true)
	field("totals", "tax_total", left.TaxTotal, right.TaxTotal, true)
	field("totals", "tip", left.Tip, right.Tip, true)
	field("totals", "total", left.Total, right.Total, true)

	missing, extra, matched := 0, 0, 0

	// line items
	{
		es, ac := left.LineItems, right.LineItems
		ea := make([]string, len(es))
		for i, x := range es {
			ea[i] = normalizeText(x.Description).(string)
		}
		ba := make([]string, len(ac))
		for i, x := range ac {
			ba[i] = normalizeText(x.Description).(string)
		}
		for _, op := range opcodes(ea, ba) {
			i, j, k, l := op.i1, op.i2, op.j1, op.j2
			paired := min(j-i, l-k)
			for off := 0; off < paired; off++ {
				e, a := es[i+off], ac[k+off]
				if normalizeText(e.Description) == normalizeText(a.Description) {
					matched++
				}
				p := func(n string) string { return "line_items[" + strconv.Itoa(i+off) + "]." + n }
				field("line_items", p("description"), e.Description, a.Description, false)
				field("line_items", p("quantity"), e.Quantity, a.Quantity, true)
				field("line_items", p("unit_price"), e.UnitPrice, a.UnitPrice, true)
				field("line_items", p("total_price"), e.TotalPrice, a.TotalPrice, true)
				if classification {
					if e.SpendingCategory != "unknown" {
						labels = append(labels, normalizeText(e.Description) == normalizeText(a.Description) && e.SpendingCategory == a.SpendingCategory)
					}
					labels = append(labels, sameSet(e.Tags, a.Tags))
				}
			}
			for index := i + paired; index < j; index++ {
				pad(strict, normal, "line_items", 4)
				diffs = append(diffs, difference{"line_items[" + strconv.Itoa(index) + "]", jsonValue(es[index]), nil, false})
				missing++
				if classification && es[index].SpendingCategory != "unknown" {
					labels = append(labels, false)
				}
				if classification && len(es[index].Tags) > 0 {
					labels = append(labels, false)
				}
			}
			for index := k + paired; index < l; index++ {
				pad(strict, normal, "line_items", 4)
				diffs = append(diffs, difference{"line_items[extra:" + strconv.Itoa(index) + "]", nil, jsonValue(ac[index]), false})
				extra++
			}
		}
	}
	// taxes
	{
		es, ac := left.Taxes, right.Taxes
		ea := make([]string, len(es))
		for i, x := range es {
			ea[i] = normalizeText(x.Name).(string)
		}
		ba := make([]string, len(ac))
		for i, x := range ac {
			ba[i] = normalizeText(x.Name).(string)
		}
		for _, op := range opcodes(ea, ba) {
			i, j, k, l := op.i1, op.i2, op.j1, op.j2
			paired := min(j-i, l-k)
			for off := 0; off < paired; off++ {
				e, a := es[i+off], ac[k+off]
				p := func(n string) string { return "taxes[" + strconv.Itoa(i+off) + "]." + n }
				field("taxes", p("name"), e.Name, a.Name, false)
				field("taxes", p("rate"), e.Rate, a.Rate, true)
				field("taxes", p("amount"), e.Amount, a.Amount, true)
			}
			for index := i + paired; index < j; index++ {
				pad(strict, normal, "taxes", 3)
				diffs = append(diffs, difference{"taxes[" + strconv.Itoa(index) + "]", jsonValue(es[index]), nil, false})
			}
			for index := k + paired; index < l; index++ {
				pad(strict, normal, "taxes", 3)
				diffs = append(diffs, difference{"taxes[extra:" + strconv.Itoa(index) + "]", nil, jsonValue(ac[index]), false})
			}
		}
	}
	if classification {
		labels = append(labels, left.Category == right.Category)
	}

	components := map[string]float64{}
	strictScore, normalScore := 0.0, 0.0
	for _, w := range weights {
		components[w.name] = fractionBools(normal[w.name])
		strictScore += fractionBools(strict[w.name]) * w.w
		normalScore += components[w.name] * w.w
	}
	_ = groups
	scores := map[string]any{
		"header": components["header"], "totals": components["totals"], "line_items": components["line_items"], "taxes": components["taxes"],
		"strict": strictScore, "normalized": normalScore, "classification": nil, "ordered_items": 0.0,
	}
	if classification {
		scores["classification"] = fractionBools(labels)
	}
	if len(left.LineItems) == len(right.LineItems) {
		var ok []bool
		for i, it := range left.LineItems {
			o := right.LineItems[i]
			ok = append(ok, i < len(right.LineItems) && it.Description == o.Description && it.Quantity == o.Quantity &&
				it.UnitPrice == o.UnitPrice && it.TotalPrice == o.TotalPrice)
		}
		scores["ordered_items"] = fractionBools(ok)
	}

	taxSum := decimal.Zero
	for _, t := range actual.Taxes {
		taxSum = taxSum.Add(t.Amount)
	}
	net := actual.Subtotal.Sub(actual.DiscountTotal).Add(actual.TaxTotal)
	if actual.Tip != nil {
		net = net.Add(*actual.Tip)
	}
	lineSum, lineScale := decimal.Zero, int32(0)
	for _, it := range actual.LineItems {
		lineSum = lineSum.Add(it.TotalPrice)
		lineScale = max(lineScale, -it.TotalPrice.Exponent())
	}
	lineText := "0"
	if len(actual.LineItems) > 0 {
		lineText = lineSum.StringFixed(lineScale)
	}
	return Detail{
		Version: DetailScoringVersion, Scores: scores, Differences: diffs,
		Rows: map[string]int{"missing": missing, "extra": extra, "matched": matched},
		Checks: map[string]any{
			"tax_lines_sum_matches_tax_total":   taxSum.Equal(actual.TaxTotal),
			"schema_net_plus_tax_matches_total": net.Equal(actual.Total),
			"line_totals_sum":                   lineText,
			"note":                              "Arithmetic checks are diagnostics; tax-inclusive receipt conventions may differ from the schema.",
		},
	}
}

func pad(strict, normal map[string][]bool, group string, n int) {
	for i := 0; i < n; i++ {
		strict[group] = append(strict[group], false)
		normal[group] = append(normal[group], false)
	}
}

func sameSet(a, b []string) bool {
	x, y := map[string]bool{}, map[string]bool{}
	for _, s := range a {
		x[s] = true
	}
	for _, s := range b {
		y[s] = true
	}
	if len(x) != len(y) {
		return false
	}
	for s := range x {
		if !y[s] {
			return false
		}
	}
	return true
}
