package domain

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/shopspring/decimal"
)

// Column scales; amounts are rounded half away from zero to these on write.
const (
	MoneyScale    = 2
	QuantityScale = 4
	RateScale     = 4
)

// Date is a calendar date serialised as YYYY-MM-DD.
type Date struct{ time.Time }

// ParseDate parses YYYY-MM-DD.
func ParseDate(s string) (Date, error) {
	t, err := time.Parse("2006-01-02", s)
	return Date{t}, err
}

func (d Date) String() string { return d.Format("2006-01-02") }

// MarshalJSON implements json.Marshaler.
func (d Date) MarshalJSON() ([]byte, error) { return json.Marshal(d.String()) }

// LineItem is one purchased row. ID and RememberCategory only appear in edit requests.
type LineItem struct {
	ID               *string
	Description      string
	SpendingCategory string
	// CategoryExplicit records that the client sent spending_category.
	CategoryExplicit bool
	Quantity         decimal.Decimal
	UnitPrice        decimal.Decimal
	TotalPrice       decimal.Decimal
	Tags             []string
	RememberCategory bool
	// SourceImage is only set for multi-photo extraction replies.
	SourceImage *int
}

// TaxLine is one tax row.
type TaxLine struct {
	Name   string
	Rate   *decimal.Decimal
	Amount decimal.Decimal
}

// Receipt is the validated receipt schema (LLM output and edit requests).
type Receipt struct {
	Confidence      int
	MerchantName    string
	MerchantAddress *string
	ReceiptNumber   *string
	Date            Date
	Time            *string
	Currency        string
	Category        string
	LineItems       []LineItem
	Taxes           []TaxLine
	Subtotal        decimal.Decimal
	DiscountTotal   decimal.Decimal
	TaxTotal        decimal.Decimal
	Tip             *decimal.Decimal
	Total           decimal.Decimal
	PaymentMethod   string
	// CollectionIDs is nil when omitted from an edit request.
	CollectionIDs *[]string
	// DuplicateCandidates is only set for multi-photo extraction replies.
	DuplicateCandidates []DuplicatePair
}

// DuplicatePair names the same printed row in two adjacent photos.
type DuplicatePair struct{ First, Second int }

// Issue is one validation problem in FastAPI/pydantic's 422 entry shape.
type Issue struct {
	Type  string         `json:"type"`
	Loc   []any          `json:"loc"`
	Msg   string         `json:"msg"`
	Input any            `json:"input,omitempty"`
	Ctx   map[string]any `json:"ctx,omitempty"`
}

// Issues is a list of validation problems.
type Issues []Issue

// Error renders the issues for repair prompts and logs.
func (is Issues) Error() string {
	parts := make([]string, 0, len(is))
	for _, i := range is {
		loc := make([]string, len(i.Loc))
		for n, l := range i.Loc {
			loc[n] = fmt.Sprint(l)
		}
		parts = append(parts, strings.Join(loc, ".")+": "+i.Msg+" ["+i.Type+"]")
	}
	return strings.Join(parts, "; ")
}

// Options select the schema variant.
type Options struct {
	// Update accepts line item id/remember_category and collection_ids and
	// rejects unknown spending categories (PUT body rules).
	Update bool
	// MultiImage requires source_image and duplicate_candidates.
	MultiImage bool
}

type parser struct {
	issues Issues
	opts   Options
}

func (p *parser) add(loc []any, typ, msg string, input any, ctx map[string]any) {
	p.issues = append(p.issues, Issue{Type: typ, Loc: loc, Msg: msg, Input: input, Ctx: ctx})
}

func with(loc []any, extra ...any) []any {
	out := make([]any, 0, len(loc)+len(extra))
	return append(append(out, loc...), extra...)
}

// ParseReceipt validates a JSON document against the receipt schema.
// The returned Issues are in a location relative to the document root.
func ParseReceipt(raw []byte, opts Options) (*Receipt, Issues) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var doc any
	if err := dec.Decode(&doc); err != nil {
		return nil, Issues{{Type: "json_invalid", Loc: []any{}, Msg: "Invalid JSON: " + err.Error()}}
	}
	return ParseReceiptValue(doc, opts, nil)
}

// ParseReceiptValue validates an already-decoded document (numbers as json.Number).
func ParseReceiptValue(doc any, opts Options, base []any) (*Receipt, Issues) {
	p := &parser{opts: opts}
	m, ok := doc.(map[string]any)
	if !ok {
		p.add(append([]any{}, base...), "model_attributes_type", "Input should be a valid dictionary or object to extract fields from", doc, nil)
		return nil, p.issues
	}
	r := p.receipt(m, append([]any{}, base...))
	if len(p.issues) > 0 {
		return nil, p.issues
	}
	return r, nil
}

func (p *parser) receipt(m map[string]any, loc []any) *Receipt {
	r := &Receipt{Currency: "EUR", Category: "other", PaymentMethod: "unknown"}
	l := func(k string) []any { return with(loc, k) }
	if v, ok := m["confidence"]; !ok {
		p.missing(l("confidence"))
	} else if n, ok := p.intVal(v, l("confidence")); ok {
		r.Confidence = int(n)
		if n < 0 {
			p.add(l("confidence"), "greater_than_equal", "Input should be greater than or equal to 0", v, map[string]any{"ge": 0})
		} else if n > 100 {
			p.add(l("confidence"), "less_than_equal", "Input should be less than or equal to 100", v, map[string]any{"le": 100})
		}
	}
	r.MerchantName, _ = p.reqString(m, "merchant_name", loc)
	r.MerchantAddress = p.optString(m, "merchant_address", loc)
	r.ReceiptNumber = p.optString(m, "receipt_number", loc)
	if s, ok := p.reqString(m, "date", loc); ok {
		if d, err := parseDateLoose(s); err != nil {
			p.add(l("date"), "date_from_datetime_parsing", "Input should be a valid date or datetime, "+err.Error(), s, nil)
		} else {
			r.Date = d
		}
	}
	r.Time = p.optString(m, "time", loc)
	if v, ok := m["currency"]; ok {
		if s, ok := p.stringVal(v, l("currency")); ok {
			r.Currency = s
		}
	}
	if v, ok := m["category"]; ok {
		r.Category = p.literal(v, ReceiptCategories, l("category"), r.Category)
	}
	if v, ok := m["payment_method"]; ok {
		r.PaymentMethod = p.literal(v, PaymentMethods, l("payment_method"), r.PaymentMethod)
	}
	r.Subtotal, _ = p.reqDecimal(m, "subtotal", loc)
	r.Total, _ = p.reqDecimal(m, "total", loc)
	r.DiscountTotal = p.nonNegDecimal(m, "discount_total", loc)
	r.TaxTotal = p.nonNegDecimal(m, "tax_total", loc)
	if v, ok := m["tip"]; ok && v != nil {
		if d, ok := p.decimalVal(v, l("tip")); ok {
			if d.IsNegative() {
				p.add(l("tip"), "greater_than_equal", "Input should be greater than or equal to 0", v, map[string]any{"ge": 0})
			}
			r.Tip = &d
		}
	}

	r.LineItems = []LineItem{}
	if v, ok := m["line_items"]; !ok {
		p.missing(l("line_items"))
	} else if arr, ok := v.([]any); !ok {
		p.add(l("line_items"), "list_type", "Input should be a valid list", v, nil)
	} else {
		for i, it := range arr {
			if item, ok := p.lineItem(it, with(loc, "line_items", i)); ok {
				r.LineItems = append(r.LineItems, item)
			}
		}
	}
	r.Taxes = []TaxLine{}
	if v, ok := m["taxes"]; ok {
		if arr, ok := v.([]any); !ok {
			p.add(l("taxes"), "list_type", "Input should be a valid list", v, nil)
		} else {
			for i, it := range arr {
				if t, ok := p.taxLine(it, with(loc, "taxes", i)); ok {
					r.Taxes = append(r.Taxes, t)
				}
			}
		}
	}
	if p.opts.Update {
		if v, ok := m["collection_ids"]; ok && v != nil {
			if arr, ok := v.([]any); !ok {
				p.add(l("collection_ids"), "list_type", "Input should be a valid list", v, nil)
			} else {
				ids := []string{}
				for i, it := range arr {
					if s, ok := p.uuidVal(it, with(loc, "collection_ids", i)); ok {
						ids = append(ids, s)
					}
				}
				r.CollectionIDs = &ids
			}
		}
	}
	if p.opts.MultiImage {
		r.DuplicateCandidates = []DuplicatePair{}
		if v, ok := m["duplicate_candidates"]; !ok {
			p.missing(l("duplicate_candidates"))
		} else if arr, ok := v.([]any); !ok {
			p.add(l("duplicate_candidates"), "list_type", "Input should be a valid list", v, nil)
		} else {
			for i, it := range arr {
				base := with(loc, "duplicate_candidates", i)
				pm, ok := it.(map[string]any)
				if !ok {
					p.add(base, "model_attributes_type", "Input should be a valid dictionary or object to extract fields from", it, nil)
					continue
				}
				a, ok1 := p.strictIndex(pm, "first_index", base)
				b, ok2 := p.strictIndex(pm, "second_index", base)
				if ok1 && ok2 {
					r.DuplicateCandidates = append(r.DuplicateCandidates, DuplicatePair{a, b})
				}
			}
		}
	}
	return r
}

// strictIndex reads a non-negative integer without string coercion.
func (p *parser) strictIndex(m map[string]any, key string, loc []any) (int, bool) {
	l := with(loc, key)
	v, ok := m[key]
	if !ok {
		p.missing(l)
		return 0, false
	}
	n, isNum := v.(json.Number)
	if !isNum {
		p.add(l, "int_type", "Input should be a valid integer", v, nil)
		return 0, false
	}
	i, err := strconv.ParseInt(n.String(), 10, 64)
	if err != nil {
		p.add(l, "int_type", "Input should be a valid integer", v, nil)
		return 0, false
	}
	if i < 0 {
		p.add(l, "greater_than_equal", "Input should be greater than or equal to 0", v, map[string]any{"ge": 0})
		return 0, false
	}
	return int(i), true
}

func (p *parser) lineItem(v any, loc []any) (LineItem, bool) {
	start := len(p.issues)
	item := LineItem{SpendingCategory: "unknown", Tags: []string{}}
	m, ok := v.(map[string]any)
	if !ok {
		p.add(loc, "model_attributes_type", "Input should be a valid dictionary or object to extract fields from", v, nil)
		return item, false
	}
	l := func(k string) []any { return with(loc, k) }
	if p.opts.Update {
		// PUT bodies reject unsupported categories outright.
		if cv, present := m["spending_category"]; present {
			if s, isStr := cv.(string); !isStr || !IsSpendingCategory(s) {
				p.add(loc, "value_error", "Value error, Unsupported spending category", m, map[string]any{"error": map[string]any{}})
			}
		}
	}
	item.Description, _ = p.reqString(m, "description", loc)
	if cv, present := m["spending_category"]; present {
		item.CategoryExplicit = true
		// Unsupported model output has no spending authority.
		if s, isStr := cv.(string); isStr && IsSpendingCategory(s) {
			item.SpendingCategory = s
		}
	}
	if qv, present := m["quantity"]; !present {
		p.missing(l("quantity"))
	} else if f, ok := p.decimalFromFloatLike(qv, l("quantity")); ok {
		item.Quantity = f
		if !f.IsPositive() {
			p.add(l("quantity"), "greater_than", "Input should be greater than 0", qv, map[string]any{"gt": 0})
		}
	}
	item.UnitPrice, _ = p.reqDecimal(m, "unit_price", loc)
	item.TotalPrice, _ = p.reqDecimal(m, "total_price", loc)
	if tv, present := m["tags"]; present {
		if arr, ok := tv.([]any); !ok {
			p.add(l("tags"), "list_type", "Input should be a valid list", tv, nil)
		} else {
			var raw []string
			for i, t := range arr {
				if s, ok := p.stringVal(t, with(loc, "tags", i)); ok {
					raw = append(raw, s)
				}
			}
			item.Tags = NormalizeTags(raw)
		}
	}
	if p.opts.Update {
		if iv, present := m["id"]; present && iv != nil {
			if s, ok := p.uuidVal(iv, l("id")); ok {
				item.ID = &s
			}
		}
		if bv, present := m["remember_category"]; present {
			if b, ok := p.boolVal(bv, l("remember_category")); ok {
				item.RememberCategory = b
			}
		}
	}
	if p.opts.MultiImage {
		if sv, present := m["source_image"]; !present {
			p.missing(l("source_image"))
		} else if n, ok := p.strictIndex(m, "source_image", loc); ok {
			item.SourceImage = &n
		} else {
			_ = sv
		}
	}
	return item, len(p.issues) == start
}

func (p *parser) taxLine(v any, loc []any) (TaxLine, bool) {
	start := len(p.issues)
	var t TaxLine
	m, ok := v.(map[string]any)
	if !ok {
		p.add(loc, "model_attributes_type", "Input should be a valid dictionary or object to extract fields from", v, nil)
		return t, false
	}
	t.Name, _ = p.reqString(m, "name", loc)
	if rv, present := m["rate"]; present && rv != nil {
		if d, ok := p.decimalFromFloatLike(rv, with(loc, "rate")); ok {
			t.Rate = &d
		}
	}
	if av, present := m["amount"]; !present {
		p.missing(with(loc, "amount"))
	} else if d, ok := p.decimalVal(av, with(loc, "amount")); ok {
		t.Amount = d
		if d.IsNegative() {
			p.add(with(loc, "amount"), "greater_than_equal", "Input should be greater than or equal to 0", av, map[string]any{"ge": 0})
		}
	}
	return t, len(p.issues) == start
}

// ── field helpers ───────────────────────────────────────────────────

func (p *parser) missing(loc []any) { p.add(loc, "missing", "Field required", nil, nil) }

func (p *parser) stringVal(v any, loc []any) (string, bool) {
	s, ok := v.(string)
	if !ok {
		p.add(loc, "string_type", "Input should be a valid string", v, nil)
	}
	return s, ok
}

func (p *parser) reqString(m map[string]any, key string, loc []any) (string, bool) {
	v, ok := m[key]
	if !ok {
		p.missing(with(loc, key))
		return "", false
	}
	return p.stringVal(v, with(loc, key))
}

func (p *parser) optString(m map[string]any, key string, loc []any) *string {
	v, ok := m[key]
	if !ok || v == nil {
		return nil
	}
	if s, ok := p.stringVal(v, with(loc, key)); ok {
		return &s
	}
	return nil
}

func (p *parser) literal(v any, allowed []string, loc []any, def string) string {
	if s, ok := v.(string); ok {
		for _, a := range allowed {
			if a == s {
				return s
			}
		}
	}
	quoted := make([]string, len(allowed))
	for i, a := range allowed {
		quoted[i] = "'" + a + "'"
	}
	exp := strings.Join(quoted[:len(quoted)-1], ", ") + " or " + quoted[len(quoted)-1]
	p.add(loc, "literal_error", "Input should be "+exp, v, map[string]any{"expected": exp})
	return def
}

func (p *parser) intVal(v any, loc []any) (int64, bool) {
	switch t := v.(type) {
	case json.Number:
		if i, err := strconv.ParseInt(t.String(), 10, 64); err == nil {
			return i, true
		}
		f, err := strconv.ParseFloat(t.String(), 64)
		if err != nil || math.IsInf(f, 0) {
			p.add(loc, "int_parsing", "Input should be a valid integer, unable to parse string as an integer", v, nil)
			return 0, false
		}
		if f != math.Trunc(f) {
			p.add(loc, "int_from_float", "Input should be a valid integer, got a number with a fractional part", v, nil)
			return 0, false
		}
		return int64(f), true
	case string:
		if i, err := strconv.ParseInt(strings.TrimSpace(t), 10, 64); err == nil {
			return i, true
		}
		p.add(loc, "int_parsing", "Input should be a valid integer, unable to parse string as an integer", v, nil)
		return 0, false
	}
	p.add(loc, "int_type", "Input should be a valid integer", v, nil)
	return 0, false
}

func (p *parser) boolVal(v any, loc []any) (bool, bool) {
	switch t := v.(type) {
	case bool:
		return t, true
	case json.Number:
		switch t.String() {
		case "0":
			return false, true
		case "1":
			return true, true
		}
	case string:
		switch strings.ToLower(t) {
		case "true", "t", "yes", "y", "on", "1":
			return true, true
		case "false", "f", "no", "n", "off", "0":
			return false, true
		}
		p.add(loc, "bool_parsing", "Input should be a valid boolean, unable to interpret input", v, nil)
		return false, false
	}
	p.add(loc, "bool_type", "Input should be a valid boolean", v, nil)
	return false, false
}

func (p *parser) uuidVal(v any, loc []any) (string, bool) {
	s, ok := v.(string)
	if !ok {
		p.add(loc, "uuid_type", "UUID input should be a string, bytes or UUID object", v, nil)
		return "", false
	}
	if u, ok := CanonicalUUID(s); ok {
		return u, true
	}
	p.add(loc, "uuid_parsing", "Input should be a valid UUID, invalid character: expected an optional prefix of `urn:uuid:` followed by [0-9a-fA-F-], found `"+firstBad(s)+"` at "+strconv.Itoa(badIndex(s)), v, nil)
	return "", false
}

func firstBad(s string) string {
	if i := badIndex(s); i < len(s) {
		return string(s[i])
	}
	return ""
}

func badIndex(s string) int {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !(c == '-' || c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F') {
			return i
		}
	}
	return len(s)
}

// CanonicalUUID accepts the forms Python's UUID() accepts for strings
// (hyphenated or bare hex, optional braces/urn prefix) and returns the
// lowercase hyphenated form.
func CanonicalUUID(s string) (string, bool) {
	h := strings.ToLower(s)
	h = strings.TrimPrefix(h, "urn:")
	h = strings.TrimPrefix(h, "uuid:")
	h = strings.Trim(h, "{}")
	h = strings.ReplaceAll(h, "-", "")
	if len(h) != 32 {
		return "", false
	}
	for _, c := range h {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return "", false
		}
	}
	return h[0:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:], true
}

func parseDecimalString(s string) (decimal.Decimal, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return decimal.Decimal{}, false
	}
	// shopspring accepts forms Python's Decimal does not (e.g. "1_0"); be strict.
	for i, c := range s {
		switch {
		case c >= '0' && c <= '9', c == '.':
		case (c == '+' || c == '-') && i == 0:
		case c == 'e' || c == 'E' || ((c == '+' || c == '-') && i > 0 && (s[i-1] == 'e' || s[i-1] == 'E')):
		default:
			return decimal.Decimal{}, false
		}
	}
	d, err := decimal.NewFromString(s)
	return d, err == nil
}

func (p *parser) decimalVal(v any, loc []any) (decimal.Decimal, bool) {
	switch t := v.(type) {
	case json.Number:
		if d, ok := parseDecimalString(t.String()); ok {
			return d, true
		}
	case string:
		if d, ok := parseDecimalString(t); ok {
			return d, true
		}
		p.add(loc, "decimal_parsing", "Input should be a valid decimal", v, nil)
		return decimal.Decimal{}, false
	default:
		p.add(loc, "decimal_type", "Decimal input should be an integer, float, string or Decimal object", v, nil)
		return decimal.Decimal{}, false
	}
	p.add(loc, "decimal_parsing", "Input should be a valid decimal", v, nil)
	return decimal.Decimal{}, false
}

// decimalFromFloatLike reads a pydantic float field (number or numeric string)
// as an exact decimal so no binary rounding is introduced.
func (p *parser) decimalFromFloatLike(v any, loc []any) (decimal.Decimal, bool) {
	switch t := v.(type) {
	case json.Number:
		if d, ok := parseDecimalString(t.String()); ok {
			return d, true
		}
	case string:
		if d, ok := parseDecimalString(t); ok {
			return d, true
		}
		p.add(loc, "float_parsing", "Input should be a valid number, unable to parse string as a number", v, nil)
		return decimal.Decimal{}, false
	default:
		p.add(loc, "float_type", "Input should be a valid number", v, nil)
		return decimal.Decimal{}, false
	}
	p.add(loc, "float_parsing", "Input should be a valid number, unable to parse string as a number", v, nil)
	return decimal.Decimal{}, false
}

func (p *parser) reqDecimal(m map[string]any, key string, loc []any) (decimal.Decimal, bool) {
	v, ok := m[key]
	if !ok {
		p.missing(with(loc, key))
		return decimal.Decimal{}, false
	}
	return p.decimalVal(v, with(loc, key))
}

func (p *parser) nonNegDecimal(m map[string]any, key string, loc []any) decimal.Decimal {
	v, ok := m[key]
	if !ok {
		return decimal.Zero
	}
	d, ok := p.decimalVal(v, with(loc, key))
	if ok && d.IsNegative() {
		p.add(with(loc, key), "greater_than_equal", "Input should be greater than or equal to 0", v, map[string]any{"ge": 0})
	}
	return d
}

func parseDateLoose(s string) (Date, error) {
	if d, err := ParseDate(s); err == nil {
		return d, nil
	}
	// pydantic also accepts a datetime whose time part is zero.
	for _, layout := range []string{"2006-01-02T15:04:05", "2006-01-02 15:04:05", time.RFC3339} {
		if t, err := time.Parse(layout, s); err == nil {
			if t.Hour() == 0 && t.Minute() == 0 && t.Second() == 0 && t.Nanosecond() == 0 {
				return Date{time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC)}, nil
			}
			return Date{}, fmt.Errorf("dates provided as datetimes must have zero time")
		}
	}
	return Date{}, fmt.Errorf("input is too short")
}

// ── DB-bound rounding and time parsing ──────────────────────────────

// Money rounds to the money column scale, half away from zero.
func Money(d decimal.Decimal) decimal.Decimal { return d.Round(MoneyScale) }

// Quantity rounds to the quantity column scale.
func Quantity(d decimal.Decimal) decimal.Decimal { return d.Round(QuantityScale) }

// Rate rounds to the tax rate column scale.
func Rate(d decimal.Decimal) decimal.Decimal { return d.Round(RateScale) }

// ParseTimeOfDay mirrors datetime.time.fromisoformat for the time column and
// returns microseconds since midnight. The error text matches Python's.
func ParseTimeOfDay(s string) (int64, error) {
	bad := fmt.Errorf("Invalid isoformat string: '%s'", s) //nolint:staticcheck
	t := s
	var us int64
	if i := strings.IndexByte(t, '.'); i >= 0 {
		frac := t[i+1:]
		t = t[:i]
		if len(frac) == 0 || len(frac) > 6 {
			// Python 3.11+ accepts any number of digits but only uses six.
			if len(frac) == 0 {
				return 0, bad
			}
			frac = frac[:6]
		}
		n, err := strconv.ParseInt(frac, 10, 64)
		if err != nil || strings.ContainsAny(frac, "+- ") {
			return 0, bad
		}
		for k := len(frac); k < 6; k++ {
			n *= 10
		}
		us = n
	}
	var h, m, sec int
	var err error
	parse2 := func(x string) (int, error) {
		if len(x) != 2 || strings.ContainsAny(x, "+- ") {
			return 0, bad
		}
		return strconv.Atoi(x)
	}
	var parts []string
	if strings.Contains(t, ":") {
		parts = strings.Split(t, ":")
	} else {
		switch len(t) {
		case 2:
			parts = []string{t}
		case 4:
			parts = []string{t[:2], t[2:]}
		case 6:
			parts = []string{t[:2], t[2:4], t[4:]}
		default:
			return 0, bad
		}
	}
	if len(parts) < 1 || len(parts) > 3 {
		return 0, bad
	}
	if h, err = parse2(parts[0]); err != nil {
		return 0, bad
	}
	if len(parts) > 1 {
		if m, err = parse2(parts[1]); err != nil {
			return 0, bad
		}
	}
	if len(parts) > 2 {
		if sec, err = parse2(parts[2]); err != nil {
			return 0, bad
		}
	}
	if us != 0 && len(parts) < 3 {
		return 0, bad
	}
	if h > 23 || m > 59 || sec > 59 {
		return 0, fmt.Errorf("%s must be in 0..23", "hour") // Python: "hour must be in 0..23"
	}
	return ((int64(h)*60+int64(m))*60+int64(sec))*1_000_000 + us, nil
}

// FormatTimeOfDay renders microseconds since midnight like time.isoformat().
func FormatTimeOfDay(us int64) string {
	sec := us / 1_000_000
	frac := us % 1_000_000
	s := fmt.Sprintf("%02d:%02d:%02d", sec/3600, (sec/60)%60, sec%60)
	if frac != 0 {
		s += fmt.Sprintf(".%06d", frac)
	}
	return s
}
