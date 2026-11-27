package store

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"time"

	"github.com/jackc/pgx/v5/pgtype"
	"github.com/shopspring/decimal"

	"github.com/tendstead/vision-bill/internal/domain"
)

// SearchProducts finds verified line items whose description contains query
// (case-insensitive, done in PostgreSQL) with price history and a summary.
func (s *Store) SearchProducts(ctx context.Context, query string, sc Scope) (*domain.ProductSearchResponse, error) {
	args := []any{"%" + query + "%"}
	sql := `SELECT li.receipt_id::text, li.description, li.quantity::text, li.unit_price::text,
		r.merchant_name, r.date, r."time", r.currency
		FROM line_items li JOIN receipts r ON r.id = li.receipt_id
		WHERE r.status = 'verified' AND r.verified = TRUE AND li.description ILIKE $1` + scopeClause(sc, &args, "r.user_id") +
		` ORDER BY r.date DESC, r."time" DESC NULLS LAST, r.created_at DESC NULLS LAST, r.id DESC, li.id DESC`
	rows, err := s.Pool.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := &domain.ProductSearchResponse{Query: query, Purchases: []domain.ProductPurchase{}}
	var prices []decimal.Decimal
	for rows.Next() {
		var p domain.ProductPurchase
		var qty, price string
		var date time.Time
		var tod pgtype.Time
		if err := rows.Scan(&p.ReceiptID, &p.Description, &qty, &price, &p.MerchantName, &date, &tod, &p.Currency); err != nil {
			return nil, err
		}
		p.Quantity, _ = strconv.ParseFloat(qty, 64)
		p.UnitPrice = domain.DecimalStr(price)
		p.Date = domain.Date{Time: date}
		if tod.Valid {
			t := domain.FormatTimeOfDay(tod.Microseconds)
			p.Time = &t
		}
		out.Purchases = append(out.Purchases, p)
		d, _ := decimal.NewFromString(price)
		prices = append(prices, d)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(out.Purchases) == 0 {
		return out, nil
	}
	latest := out.Purchases[0].UnitPrice
	out.LatestPrice = &latest
	// The cheapest price ignores negative lines: a returned-deposit refund is
	// money back, not a price a customer pays.
	var cheapest *domain.DecimalStr
	for i, p := range prices {
		if !p.IsNegative() && (cheapest == nil || p.LessThan(mustDec(string(*cheapest)))) {
			c := out.Purchases[i].UnitPrice
			cheapest = &c
		}
	}
	out.CheapestPrice = cheapest
	sum := decimal.Zero
	for _, p := range prices {
		sum = sum.Add(p)
	}
	avg := domain.DecimalStr(sum.Div(decimal.NewFromInt(int64(len(prices)))).String())
	out.AveragePrice = &avg
	cur := out.Purchases[0].Currency
	out.Currency = &cur
	return out, nil
}

func mustDec(s string) decimal.Decimal {
	d, _ := decimal.NewFromString(s)
	return d
}

// ReceiptStatistics aggregates verified receipts.
func (s *Store) ReceiptStatistics(ctx context.Context, sc Scope, weeks int, collectionID *string) (*domain.ReceiptStatistics, error) {
	var scopeArgs []any
	scope := scopeClause(sc, &scopeArgs, "user_id")
	if collectionID != nil {
		scopeArgs = append(scopeArgs, *collectionID)
		scope += fmt.Sprintf(" AND EXISTS (SELECT 1 FROM receipt_collections rc WHERE rc.receipt_id = receipts.id AND rc.collection_id = $%d::uuid)", len(scopeArgs))
	}
	const verified = ` WHERE status = 'verified' AND verified = TRUE`

	st := &domain.ReceiptStatistics{
		Currencies: []domain.CurrencyStatistics{}, Merchants: []domain.NamedStatistics{}, Categories: []domain.NamedStatistics{},
		PaymentMethods: []domain.NamedStatistics{}, Weekdays: []domain.WeekdayStatistics{}, WeeklySpending: []domain.WeeklyStatistics{},
	}

	rows, err := s.Pool.Query(ctx, `SELECT currency, COUNT(*)::int, SUM(total)::text, AVG(total)::text,
		percentile_cont(0.5) WITHIN GROUP (ORDER BY total)::numeric::text, MIN(total)::text, MAX(total)::text,
		SUM(subtotal)::text, SUM(discount_total)::text, SUM(tax_total)::text, SUM(COALESCE(tip, 0))::text
		FROM receipts`+verified+scope+` GROUP BY currency ORDER BY SUM(total) DESC, currency`, scopeArgs...)
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		var c domain.CurrencyStatistics
		var total, avg, median, min, max, sub, disc, tax, tips string
		if err := rows.Scan(&c.Currency, &c.ReceiptCount, &total, &avg, &median, &min, &max, &sub, &disc, &tax, &tips); err != nil {
			rows.Close()
			return nil, err
		}
		c.Total, c.Average, c.Median, c.Minimum, c.Maximum = domain.DecimalStr(total), domain.DecimalStr(avg), domain.DecimalStr(median), domain.DecimalStr(min), domain.DecimalStr(max)
		c.Subtotal, c.Discounts, c.Taxes, c.Tips = domain.DecimalStr(sub), domain.DecimalStr(disc), domain.DecimalStr(tax), domain.DecimalStr(tips)
		st.VerifiedReceiptCount += c.ReceiptCount
		st.Currencies = append(st.Currencies, c)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}

	named := func(field string) ([]domain.NamedStatistics, error) {
		rs, err := s.Pool.Query(ctx, fmt.Sprintf(`SELECT %[1]s AS name, currency, COUNT(*)::int, SUM(total)::text, AVG(total)::text
			FROM receipts`+verified+scope+` GROUP BY %[1]s, currency ORDER BY SUM(total) DESC, 1, currency`, field), scopeArgs...)
		if err != nil {
			return nil, err
		}
		defer rs.Close()
		out := []domain.NamedStatistics{}
		for rs.Next() {
			var n domain.NamedStatistics
			var total, avg string
			if err := rs.Scan(&n.Name, &n.Currency, &n.ReceiptCount, &total, &avg); err != nil {
				return nil, err
			}
			n.Total, n.Average = domain.DecimalStr(total), domain.DecimalStr(avg)
			out = append(out, n)
		}
		return out, rs.Err()
	}
	if st.Merchants, err = named(`COALESCE(NULLIF(merchant_name, ''), 'Unknown merchant')`); err != nil {
		return nil, err
	}
	if st.Categories, err = named("category"); err != nil {
		return nil, err
	}
	if st.PaymentMethods, err = named("payment_method"); err != nil {
		return nil, err
	}

	wd, err := s.Pool.Query(ctx, `SELECT EXTRACT(ISODOW FROM date)::int AS weekday, currency, COUNT(*)::int, SUM(total)::text, AVG(total)::text
		FROM receipts`+verified+scope+` GROUP BY weekday, currency ORDER BY weekday, currency`, scopeArgs...)
	if err != nil {
		return nil, err
	}
	for wd.Next() {
		var w domain.WeekdayStatistics
		var total, avg string
		if err := wd.Scan(&w.Weekday, &w.Currency, &w.ReceiptCount, &total, &avg); err != nil {
			wd.Close()
			return nil, err
		}
		w.Total, w.Average = domain.DecimalStr(total), domain.DecimalStr(avg)
		st.Weekdays = append(st.Weekdays, w)
	}
	wd.Close()
	if err := wd.Err(); err != nil {
		return nil, err
	}

	weeklyArgs := append(append([]any{}, scopeArgs...), float64(max(weeks-1, 0)))
	wk, err := s.Pool.Query(ctx, fmt.Sprintf(`SELECT date_trunc('week', date)::date AS week_start, currency, COUNT(*)::int, SUM(total)::text, AVG(total)::text
		FROM receipts`+verified+` AND date >= date_trunc('week', CURRENT_DATE) - $%d::float8 * INTERVAL '1 week'`+scope+`
		GROUP BY week_start, currency ORDER BY week_start, currency`, len(weeklyArgs)), weeklyArgs...)
	if err != nil {
		return nil, err
	}
	defer wk.Close()
	for wk.Next() {
		var w domain.WeeklyStatistics
		var start time.Time
		var total, avg string
		if err := wk.Scan(&start, &w.Currency, &w.ReceiptCount, &total, &avg); err != nil {
			return nil, err
		}
		w.WeekStart = domain.Date{Time: start}
		w.Total, w.Average = domain.DecimalStr(total), domain.DecimalStr(avg)
		st.WeeklySpending = append(st.WeeklySpending, w)
	}
	return st, wk.Err()
}

// acc accumulates exact amounts; an empty accumulator prints as "0" while a
// non-empty one keeps the money scale.
type acc struct {
	d decimal.Decimal
	n int
}

func (a *acc) add(d decimal.Decimal) { a.d = a.d.Add(d); a.n++ }

func (a acc) str() domain.DecimalStr {
	if a.n == 0 {
		return "0"
	}
	return domain.DecimalStr(a.d.StringFixed(domain.MoneyScale))
}

func monthStart(t time.Time, offset int) time.Time {
	total := t.Year()*12 + int(t.Month()) - 1 + offset
	return time.Date(total/12, time.Month(total%12+1), 1, 0, 0, 0, 0, time.UTC)
}

func share(amount, total decimal.Decimal) *domain.DecimalStr {
	// Net refunds can make a share negative; a non-positive denominator has no
	// useful interpretation, so the share is null.
	if !total.IsPositive() {
		return nil
	}
	v := domain.DecimalStr(amount.Mul(decimal.NewFromInt(100)).DivRound(total, 2).StringFixed(2))
	return &v
}

// ItemStatistics reports category shares and calendar-month trends through
// today. Deposits are separate, unknown items stay in the denominator and
// receipt-wide adjustments surface as "unallocated".
func (s *Store) ItemStatistics(ctx context.Context, sc Scope, months int, collectionID *string, now time.Time) (*domain.ItemStatistics, error) {
	today := time.Date(now.UTC().Year(), now.UTC().Month(), now.UTC().Day(), 0, 0, 0, 0, time.UTC)
	start := monthStart(today, 1-months)
	canSeeAll := sc.CanSeeAll || sc.UserID == ""
	var userID *string
	if sc.UserID != "" {
		userID = &sc.UserID
	}
	rows, err := s.Pool.Query(ctx, `
WITH scoped_receipts AS (
    SELECT r.id, r.date, r.currency, r.total FROM receipts r
    WHERE r.status = 'verified' AND r.verified = TRUE AND r.date >= $1 AND r.date <= $2
      AND ($3::boolean OR r.user_id = $4::uuid)
      AND ($5::uuid IS NULL OR EXISTS (SELECT 1 FROM receipt_collections rc WHERE rc.receipt_id = r.id AND rc.collection_id = $5::uuid))
)
SELECT 'item' AS kind, date_trunc('month', r.date)::date AS month, li.spending_category AS category, r.currency, SUM(li.total_price)::text AS total
FROM scoped_receipts r JOIN line_items li ON li.receipt_id = r.id GROUP BY month, li.spending_category, r.currency
UNION ALL
SELECT 'receipt', NULL::date, NULL::text, currency, SUM(total)::text FROM scoped_receipts GROUP BY currency`,
		start, today, canSeeAll, userID, collectionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	type key struct{ category, currency string }
	type mkey struct {
		month    time.Time
		category string
		currency string
	}
	type mcur struct {
		month    time.Time
		currency string
	}
	totals := map[key]*acc{}
	monthly := map[mkey]*acc{}
	currencyTotals := map[string]decimal.Decimal{}
	monthTotals := map[mcur]decimal.Decimal{}
	receiptTotals := map[string]*acc{}
	deposits := map[string]*acc{}
	itemTotals := map[string]*acc{}
	get := func(m map[string]*acc, k string) *acc {
		if m[k] == nil {
			m[k] = &acc{}
		}
		return m[k]
	}
	for rows.Next() {
		var kind, currency, totalText string
		var month pgtype.Date
		var category *string
		if err := rows.Scan(&kind, &month, &category, &currency, &totalText); err != nil {
			return nil, err
		}
		total := mustDec(totalText)
		if kind == "receipt" {
			get(receiptTotals, currency).add(total)
			continue
		}
		cat := "unknown"
		if category != nil && *category != "" {
			cat = *category
		}
		get(itemTotals, currency).add(total)
		if cat == "deposit" {
			get(deposits, currency).add(total)
			continue
		}
		m := month.Time
		k, mk := key{cat, currency}, mkey{m, cat, currency}
		if totals[k] == nil {
			totals[k] = &acc{}
		}
		if monthly[mk] == nil {
			monthly[mk] = &acc{}
		}
		totals[k].add(total)
		monthly[mk].add(total)
		currencyTotals[currency] = currencyTotals[currency].Add(total)
		monthTotals[mcur{m, currency}] = monthTotals[mcur{m, currency}].Add(total)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	out := &domain.ItemStatistics{Months: months, DateFrom: domain.Date{Time: start}, DateTo: domain.Date{Time: today},
		Categories: []domain.ItemCategoryStatistics{}, Monthly: []domain.MonthlyItemStatistics{}, Reconciliation: []domain.ItemReconciliation{}}
	keys := make([]key, 0, len(totals))
	for k := range totals {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := keys[i], keys[j]
		if a.currency != b.currency {
			return a.currency < b.currency
		}
		if c := totals[a].d.Cmp(totals[b].d); c != 0 {
			return c > 0
		}
		return a.category < b.category
	})
	for _, k := range keys {
		out.Categories = append(out.Categories, domain.ItemCategoryStatistics{
			Category: k.category, Currency: k.currency, Total: totals[k].str(), Share: share(totals[k].d, currencyTotals[k.currency])})
	}
	pairs := append([]key(nil), keys...)
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].category != pairs[j].category {
			return pairs[i].category < pairs[j].category
		}
		return pairs[i].currency < pairs[j].currency
	})
	for offset := 0; offset < months; offset++ {
		m := monthStart(start, offset)
		for _, k := range pairs {
			a := monthly[mkey{m, k.category, k.currency}]
			if a == nil {
				a = &acc{}
			}
			out.Monthly = append(out.Monthly, domain.MonthlyItemStatistics{
				Month: domain.Date{Time: m}, Category: k.category, Currency: k.currency, Total: a.str(),
				Share: share(a.d, monthTotals[mcur{m, k.currency}])})
		}
	}
	currencies := map[string]bool{}
	for c := range receiptTotals {
		currencies[c] = true
	}
	for c := range itemTotals {
		currencies[c] = true
	}
	var names []string
	for c := range currencies {
		names = append(names, c)
	}
	sort.Strings(names)
	for _, c := range names {
		r, i, d := get(receiptTotals, c), get(itemTotals, c), get(deposits, c)
		un := acc{d: r.d.Sub(i.d), n: r.n + i.n}
		out.Reconciliation = append(out.Reconciliation, domain.ItemReconciliation{
			Currency: c, ReceiptTotal: r.str(), ItemTotal: i.str(), DepositTotal: d.str(), UnallocatedTotal: un.str()})
	}
	return out, nil
}
