package store

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/shopspring/decimal"

	"github.com/tendstead/vision-bill/internal/domain"
)

// Scope restricts queries to the caller's rows unless CanSeeAll is set.
type Scope struct {
	UserID    string
	CanSeeAll bool
}

// ValidationError is a client error raised below the HTTP layer (a 422).
type ValidationError struct{ Msg string }

func (e *ValidationError) Error() string { return e.Msg }

// ErrReceiptReferenced is returned when a benchmark task still references the receipt.
var ErrReceiptReferenced = errors.New("receipt is referenced by a benchmark run")

const foreignKeyViolation = "23503"

// receiptCols selects a receipt with the API's conversions (exact numerics as
// text, time as microseconds, created_at as the UTC date).
const receiptCols = `r.id::text, r.confidence, r.merchant_name, r.merchant_address, r.receipt_number,
	r.date, r."time", r.currency, r.category, r.subtotal::text, r.discount_total::text, r.tax_total::text,
	r.tip::text, r.total::text, r.payment_method, (r.created_at AT TIME ZONE 'UTC')::date, r.status,
	r.image_id::text, r.verified, r.user_id::text`

const thumbSubquery = `(SELECT i.thumbnail_path FROM images i WHERE i.id = r.image_id)`

func scanReceipt(row pgx.Row, withThumb bool) (*domain.ReceiptRow, error) {
	var (
		r       domain.ReceiptRow
		date    time.Time
		tod     pgtype.Time
		created pgtype.Date
		tip     *string
		thumb   *string
	)
	var sub, disc, tax, total string
	dest := []any{&r.ID, &r.Confidence, &r.MerchantName, &r.MerchantAddress, &r.ReceiptNumber,
		&date, &tod, &r.Currency, &r.Category, &sub, &disc, &tax, &tip, &total, &r.PaymentMethod,
		&created, &r.Status, &r.ImageID, &r.Verified, &r.UserID}
	if withThumb {
		dest = append(dest, &thumb)
	}
	if err := row.Scan(dest...); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	r.Date = domain.Date{Time: date}
	if tod.Valid {
		s := domain.FormatTimeOfDay(tod.Microseconds)
		r.Time = &s
	}
	if created.Valid {
		d := domain.Date{Time: created.Time}
		r.CreatedAt = &d
	}
	r.Subtotal, r.DiscountTotal, r.TaxTotal, r.Total = domain.DecimalStr(sub), domain.DecimalStr(disc), domain.DecimalStr(tax), domain.DecimalStr(total)
	if tip != nil {
		t := domain.DecimalStr(*tip)
		r.Tip = &t
	}
	r.ThumbnailPath = thumb
	return &r, nil
}

// receiptArgs are the values shared by insert and update, rounded to the
// column scales (half away from zero; money never passes through float64).
type receiptArgs struct {
	confidence                     int
	merchant                       string
	address, number                *string
	date                           time.Time
	tod                            *string // formatted time, nil when absent
	currency, category             string
	subtotal, discount, tax, total string
	tip                            *string
	payment                        string
}

func prepareReceipt(r *domain.Receipt) (receiptArgs, error) {
	a := receiptArgs{
		confidence: r.Confidence, merchant: r.MerchantName, address: r.MerchantAddress, number: r.ReceiptNumber,
		date: r.Date.Time, currency: r.Currency, category: r.Category, payment: r.PaymentMethod,
		subtotal: domain.Money(r.Subtotal).StringFixed(domain.MoneyScale),
		discount: domain.Money(r.DiscountTotal).StringFixed(domain.MoneyScale),
		tax:      domain.Money(r.TaxTotal).StringFixed(domain.MoneyScale),
		total:    domain.Money(r.Total).StringFixed(domain.MoneyScale),
	}
	if r.Time != nil && *r.Time != "" {
		us, err := domain.ParseTimeOfDay(*r.Time)
		if err != nil {
			return a, &ValidationError{err.Error()}
		}
		s := domain.FormatTimeOfDay(us)
		a.tod = &s
	}
	if r.Tip != nil {
		s := domain.Money(*r.Tip).StringFixed(domain.MoneyScale)
		a.tip = &s
	}
	return a, nil
}

func (a receiptArgs) values() []any {
	return []any{a.confidence, a.merchant, a.address, a.number, a.date, a.tod, a.currency, a.category,
		a.subtotal, a.discount, a.tax, a.tip, a.total, a.payment}
}

const insertReceiptSQL = `
INSERT INTO receipts
  (confidence, merchant_name, merchant_address, receipt_number, date, "time", currency, category,
   subtotal, discount_total, tax_total, tip, total, payment_method, status, image_id, verified, user_id)
VALUES ($1,$2,$3,$4,$5,$6::time,$7,$8,$9::numeric,$10::numeric,$11::numeric,$12::numeric,$13::numeric,$14,$15,$16::uuid,$17,$18::uuid)
RETURNING id::text`

const updateByImageSQL = `
UPDATE receipts SET confidence=$1, merchant_name=$2, merchant_address=$3, receipt_number=$4, date=$5,
  "time"=$6::time, currency=$7, category=$8, subtotal=$9::numeric, discount_total=$10::numeric,
  tax_total=$11::numeric, tip=$12::numeric, total=$13::numeric, payment_method=$14, status=$15,
  verified=$16, user_id=$17::uuid
WHERE id = (SELECT id FROM receipts WHERE image_id = $18::uuid ORDER BY created_at ASC, id ASC LIMIT 1)
RETURNING id::text`

const insertLineItemSQL = `
INSERT INTO line_items (receipt_id, description, quantity, unit_price, total_price, tags, position, id,
  spending_category, category_source, original_description)
VALUES ($1::uuid,$2,$3::numeric,$4::numeric,$5::numeric,$6,$7,COALESCE($8::uuid, gen_random_uuid()),$9,$10,$11)`

type itemDecision struct {
	id       *string
	original string
	category string
	source   string
}

func insertChildren(ctx context.Context, tx pgx.Tx, receiptID string, r *domain.Receipt, decisions []itemDecision) error {
	for i, item := range r.LineItems {
		d := decisions[i]
		tags := item.Tags
		if tags == nil {
			tags = []string{}
		}
		if _, err := tx.Exec(ctx, insertLineItemSQL, receiptID, item.Description,
			domain.Quantity(item.Quantity).StringFixed(domain.QuantityScale),
			domain.Money(item.UnitPrice).StringFixed(domain.MoneyScale),
			domain.Money(item.TotalPrice).StringFixed(domain.MoneyScale),
			tags, i, d.id, d.category, d.source, d.original); err != nil {
			return err
		}
	}
	for _, t := range r.Taxes {
		var rate *string
		if t.Rate != nil {
			s := domain.Rate(*t.Rate).StringFixed(domain.RateScale)
			rate = &s
		}
		if _, err := tx.Exec(ctx, `INSERT INTO taxes (receipt_id, name, rate, amount) VALUES ($1::uuid,$2,$3::numeric,$4::numeric)`,
			receiptID, t.Name, rate, domain.Money(t.Amount).StringFixed(domain.MoneyScale)); err != nil {
			return err
		}
	}
	return nil
}

func historyForItems(ctx context.Context, tx pgx.Tx, userID *string, merchant string, descriptions []string) (map[string]string, error) {
	out := map[string]string{}
	if userID == nil || len(descriptions) == 0 {
		return out, nil
	}
	merchantKey := domain.NormalizeHistoryKey(merchant)
	seen := map[string]bool{}
	var keys []string
	for _, d := range descriptions {
		if strings.TrimSpace(d) == "" {
			continue
		}
		k := domain.NormalizeHistoryKey(d)
		if !seen[k] {
			seen[k] = true
			keys = append(keys, k)
		}
	}
	if merchantKey == "" || len(keys) == 0 {
		return out, nil
	}
	rows, err := tx.Query(ctx, `SELECT description_key, category FROM item_category_aliases
		WHERE user_id = $1::uuid AND merchant_key = $2 AND description_key = ANY($3::text[])`, *userID, merchantKey, keys)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var k, c string
		if err := rows.Scan(&k, &c); err != nil {
			return nil, err
		}
		if domain.IsSpendingCategory(c) && c != "unknown" {
			out[k] = c
		}
	}
	return out, rows.Err()
}

// PersistReceipt stores an extracted receipt. When imageID has an existing
// receipt it is updated in place (idempotent re-analysis); otherwise inserted.
func (s *Store) PersistReceipt(ctx context.Context, r *domain.Receipt, imageID *string, status string, verified bool, userID *string) (*domain.ReceiptRow, error) {
	a, err := prepareReceipt(r)
	if err != nil {
		return nil, err
	}
	var id string
	err = s.inTx(ctx, func(tx pgx.Tx) error {
		found := false
		if imageID != nil {
			args := append(a.values(), status, verified, userID, *imageID)
			e := tx.QueryRow(ctx, updateByImageSQL, args...).Scan(&id)
			switch {
			case e == nil:
				found = true
			case errors.Is(e, pgx.ErrNoRows):
			default:
				return e
			}
		}
		if !found {
			args := append(a.values(), status, imageID, verified, userID)
			if e := tx.QueryRow(ctx, insertReceiptSQL, args...).Scan(&id); e != nil {
				return e
			}
		}
		descs := make([]string, len(r.LineItems))
		for i, it := range r.LineItems {
			descs[i] = it.Description
		}
		history, e := historyForItems(ctx, tx, userID, r.MerchantName, descs)
		if e != nil {
			return e
		}
		decisions := make([]itemDecision, len(r.LineItems))
		for i, it := range r.LineItems {
			cat, ok := history[domain.NormalizeHistoryKey(it.Description)]
			d := itemDecision{original: it.Description, category: it.SpendingCategory, source: "unknown"}
			switch {
			case ok:
				d.category, d.source = cat, "history"
			case it.SpendingCategory != "unknown":
				d.source = "llm"
			}
			decisions[i] = d
		}
		if _, e := tx.Exec(ctx, `DELETE FROM line_items WHERE receipt_id = $1::uuid`, id); e != nil {
			return e
		}
		if _, e := tx.Exec(ctx, `DELETE FROM taxes WHERE receipt_id = $1::uuid`, id); e != nil {
			return e
		}
		return insertChildren(ctx, tx, id, r, decisions)
	})
	if err != nil {
		return nil, err
	}
	s.captureToActive(ctx, id, userID)
	return s.ReceiptByID(ctx, id, Scope{CanSeeAll: true})
}

// captureToActive assigns a freshly persisted receipt to its owner's active
// collection. It is best effort: a failure never breaks persistence.
func (s *Store) captureToActive(ctx context.Context, receiptID string, userID *string) {
	if userID == nil {
		return
	}
	active, err := s.ActiveCollection(ctx, Scope{UserID: *userID})
	if err != nil || active == nil {
		return
	}
	_ = s.AssignReceipt(ctx, active.ID, receiptID)
}

func (s *Store) inTx(ctx context.Context, fn func(pgx.Tx) error) error {
	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if err := fn(tx); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

func scopeClause(sc Scope, args *[]any, col string) string {
	if sc.CanSeeAll || sc.UserID == "" {
		return ""
	}
	*args = append(*args, sc.UserID)
	return fmt.Sprintf(" AND %s = $%d::uuid", col, len(*args))
}

// ReceiptByID returns a receipt row visible to the caller, or nil.
func (s *Store) ReceiptByID(ctx context.Context, id string, sc Scope) (*domain.ReceiptRow, error) {
	args := []any{id}
	sql := `SELECT ` + receiptCols + ` FROM receipts r WHERE r.id = $1::uuid` + scopeClause(sc, &args, "r.user_id")
	return scanReceipt(s.Pool.QueryRow(ctx, sql, args...), false)
}

// ListFilter are the optional receipt list filters.
type ListFilter struct {
	Limit, Offset int
	Status        []string
	DateFrom      *time.Time
	DateTo        *time.Time
	Search        string
	CollectionID  *string
}

// ListReceipts returns receipts ordered by date descending (id breaks ties so paging is stable).
func (s *Store) ListReceipts(ctx context.Context, f ListFilter, sc Scope) ([]domain.ReceiptRow, error) {
	var where []string
	var args []any
	if !sc.CanSeeAll && sc.UserID != "" {
		args = append(args, sc.UserID)
		where = append(where, fmt.Sprintf("r.user_id = $%d::uuid", len(args)))
	}
	if len(f.Status) > 0 {
		args = append(args, f.Status)
		where = append(where, fmt.Sprintf("r.status = ANY($%d)", len(args)))
	}
	if f.DateFrom != nil {
		args = append(args, *f.DateFrom)
		where = append(where, fmt.Sprintf("r.date >= $%d", len(args)))
	}
	if f.DateTo != nil {
		args = append(args, *f.DateTo)
		where = append(where, fmt.Sprintf("r.date <= $%d", len(args)))
	}
	if f.Search != "" {
		args = append(args, "%"+f.Search+"%")
		where = append(where, fmt.Sprintf("(r.merchant_name ILIKE $%d OR r.receipt_number ILIKE $%d)", len(args), len(args)))
	}
	if f.CollectionID != nil {
		args = append(args, *f.CollectionID)
		where = append(where, fmt.Sprintf("EXISTS (SELECT 1 FROM receipt_collections rc WHERE rc.receipt_id = r.id AND rc.collection_id = $%d::uuid)", len(args)))
	}
	sql := `SELECT ` + receiptCols + `, ` + thumbSubquery + ` AS thumbnail_path FROM receipts r`
	if len(where) > 0 {
		sql += " WHERE " + strings.Join(where, " AND ")
	}
	args = append(args, f.Limit, f.Offset)
	sql += " ORDER BY r.date DESC, r.id DESC LIMIT $" + strconv.Itoa(len(args)-1) + " OFFSET $" + strconv.Itoa(len(args))
	rows, err := s.Pool.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []domain.ReceiptRow{}
	for rows.Next() {
		r, err := scanReceipt(rows, true)
		if err != nil {
			return nil, err
		}
		out = append(out, *r)
	}
	return out, rows.Err()
}

// ReceiptDetails returns a receipt with line items, taxes, image path and collections.
func (s *Store) ReceiptDetails(ctx context.Context, id string, sc Scope) (*domain.ReceiptWithDetails, error) {
	args := []any{id}
	sql := `SELECT ` + receiptCols + `, i.thumbnail_path, i.image_path FROM receipts r
		LEFT JOIN images i ON i.id = r.image_id WHERE r.id = $1::uuid` + scopeClause(sc, &args, "r.user_id")
	var imagePath *string
	var row *domain.ReceiptRow
	{
		// scanReceipt appends only the thumbnail; scan image_path via a wrapper row.
		rows, err := s.Pool.Query(ctx, sql, args...)
		if err != nil {
			return nil, err
		}
		if rows.Next() {
			var wrapper = &extraScan{rows: rows, extra: []any{&imagePath}}
			r, err := scanReceipt(wrapper, true)
			if err != nil {
				rows.Close()
				return nil, err
			}
			row = r
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			return nil, err
		}
	}
	if row == nil {
		return nil, nil
	}
	d := &domain.ReceiptWithDetails{Receipt: *row, ImagePath: imagePath, LineItems: []domain.LineItemRow{}, Taxes: []domain.TaxLineRow{}, CollectionIDs: []string{}}

	li, err := s.Pool.Query(ctx, `SELECT id::text, receipt_id::text, description, original_description, spending_category,
		category_source, quantity::text, unit_price::text, total_price::text, tags, position
		FROM line_items WHERE receipt_id = $1::uuid ORDER BY position`, id)
	if err != nil {
		return nil, err
	}
	for li.Next() {
		var r domain.LineItemRow
		var q, up, tp string
		var tags []string
		if err := li.Scan(&r.ID, &r.ReceiptID, &r.Description, &r.OriginalDescription, &r.SpendingCategory,
			&r.CategorySource, &q, &up, &tp, &tags, &r.Position); err != nil {
			li.Close()
			return nil, err
		}
		f, _ := strconv.ParseFloat(q, 64)
		r.Quantity, r.UnitPrice, r.TotalPrice = f, domain.DecimalStr(up), domain.DecimalStr(tp)
		if tags == nil {
			tags = []string{}
		}
		r.Tags = tags
		d.LineItems = append(d.LineItems, r)
	}
	li.Close()
	if err := li.Err(); err != nil {
		return nil, err
	}

	tx, err := s.Pool.Query(ctx, `SELECT id::text, receipt_id::text, name, rate::text, amount::text FROM taxes WHERE receipt_id = $1::uuid ORDER BY id`, id)
	if err != nil {
		return nil, err
	}
	for tx.Next() {
		var t domain.TaxLineRow
		var rate *string
		var amount string
		if err := tx.Scan(&t.ID, &t.ReceiptID, &t.Name, &rate, &amount); err != nil {
			tx.Close()
			return nil, err
		}
		if rate != nil {
			f, _ := strconv.ParseFloat(*rate, 64)
			pf := f
			t.Rate = &pf
		}
		t.Amount = domain.DecimalStr(amount)
		d.Taxes = append(d.Taxes, t)
	}
	tx.Close()
	if err := tx.Err(); err != nil {
		return nil, err
	}

	cr, err := s.Pool.Query(ctx, `SELECT collection_id::text FROM receipt_collections WHERE receipt_id = $1::uuid`, id)
	if err != nil {
		return nil, err
	}
	for cr.Next() {
		var c string
		if err := cr.Scan(&c); err != nil {
			cr.Close()
			return nil, err
		}
		d.CollectionIDs = append(d.CollectionIDs, c)
	}
	cr.Close()
	return d, cr.Err()
}

// extraScan lets scanReceipt read trailing columns it does not know about.
type extraScan struct {
	rows  pgx.Rows
	extra []any
}

func (e *extraScan) Scan(dest ...any) error { return e.rows.Scan(append(dest, e.extra...)...) }

// UpdateReceipt replaces a receipt's fields, line items and taxes. It returns
// (nil, nil) when the receipt does not exist for the caller.
func (s *Store) UpdateReceipt(ctx context.Context, id string, r *domain.Receipt, sc Scope) (*domain.ReceiptRow, error) {
	a, err := prepareReceipt(r)
	if err != nil {
		return nil, err
	}
	found := false
	err = s.inTx(ctx, func(tx pgx.Tx) error {
		args := append(a.values(), id)
		sql := `UPDATE receipts SET confidence=$1, merchant_name=$2, merchant_address=$3, receipt_number=$4, date=$5,
			"time"=$6::time, currency=$7, category=$8, subtotal=$9::numeric, discount_total=$10::numeric,
			tax_total=$11::numeric, tip=$12::numeric, total=$13::numeric, payment_method=$14
			WHERE id = $15::uuid` + scopeClause(sc, &args, "user_id") + ` RETURNING user_id::text`
		var owner *string
		if e := tx.QueryRow(ctx, sql, args...).Scan(&owner); e != nil {
			if errors.Is(e, pgx.ErrNoRows) {
				return nil
			}
			return e
		}
		found = true

		type oldItem struct {
			id, description, original, category, source string
			position                                    int
		}
		rows, e := tx.Query(ctx, `SELECT id::text, description, COALESCE(original_description, ''), spending_category,
			category_source, position FROM line_items WHERE receipt_id = $1::uuid ORDER BY position`, id)
		if e != nil {
			return e
		}
		var oldRows []oldItem
		byID := map[string]*oldItem{}
		byPos := map[int]*oldItem{}
		for rows.Next() {
			var o oldItem
			if e := rows.Scan(&o.id, &o.description, &o.original, &o.category, &o.source, &o.position); e != nil {
				rows.Close()
				return e
			}
			oldRows = append(oldRows, o)
		}
		rows.Close()
		if e := rows.Err(); e != nil {
			return e
		}
		for i := range oldRows {
			byID[oldRows[i].id] = &oldRows[i]
			byPos[oldRows[i].position] = &oldRows[i]
		}

		used := map[string]bool{}
		decisions := make([]itemDecision, len(r.LineItems))
		merchantKey := domain.NormalizeHistoryKey(r.MerchantName)
		legacy := true
		for _, it := range r.LineItems {
			if it.ID != nil {
				legacy = false
			}
		}
		for pos, it := range r.LineItems {
			var old *oldItem
			if it.ID != nil {
				o, ok := byID[*it.ID]
				if !ok {
					return &ValidationError{"Line item id does not belong to this receipt"}
				}
				old = o
			}
			if old == nil && legacy && len(oldRows) == len(r.LineItems) {
				if c := byPos[pos]; c != nil && c.description == it.Description {
					old = c
				}
			}
			if old != nil && used[old.id] {
				return &ValidationError{"Duplicate line item id"}
			}
			if old != nil {
				used[old.id] = true
			}
			original := it.Description
			oldCategory, oldSource := "unknown", "unknown"
			if old != nil {
				original = old.original
				if original == "" {
					original = old.description
				}
				oldCategory, oldSource = old.category, old.source
			}
			changed := it.CategoryExplicit && it.SpendingCategory != oldCategory
			category := oldCategory
			if changed || old == nil {
				category = it.SpendingCategory
			}
			source := oldSource
			if changed || it.RememberCategory {
				source = "user"
			}
			if old == nil && it.CategoryExplicit && category != "unknown" {
				source = "user"
			}
			dec := itemDecision{original: original, category: category, source: source}
			if old != nil {
				oid := old.id
				dec.id = &oid
			}
			decisions[pos] = dec
			if it.RememberCategory && owner != nil && merchantKey != "" {
				for _, key := range uniqueKeys(domain.NormalizeHistoryKey(original), domain.NormalizeHistoryKey(it.Description)) {
					if key == "" {
						continue
					}
					if category == "unknown" {
						_, e = tx.Exec(ctx, `DELETE FROM item_category_aliases WHERE user_id = $1::uuid AND merchant_key = $2 AND description_key = $3`, *owner, merchantKey, key)
					} else {
						_, e = tx.Exec(ctx, `INSERT INTO item_category_aliases (user_id, merchant_key, description_key, category)
							VALUES ($1::uuid,$2,$3,$4) ON CONFLICT (user_id, merchant_key, description_key) DO UPDATE SET category = EXCLUDED.category`,
							*owner, merchantKey, key, category)
					}
					if e != nil {
						return e
					}
				}
			}
		}
		if _, e := tx.Exec(ctx, `DELETE FROM line_items WHERE receipt_id = $1::uuid`, id); e != nil {
			return e
		}
		if _, e := tx.Exec(ctx, `DELETE FROM taxes WHERE receipt_id = $1::uuid`, id); e != nil {
			return e
		}
		return insertChildren(ctx, tx, id, r, decisions)
	})
	if err != nil || !found {
		return nil, err
	}
	return s.ReceiptByID(ctx, id, Scope{CanSeeAll: true})
}

func uniqueKeys(a, b string) []string {
	if a == b {
		return []string{a}
	}
	return []string{a, b}
}

// VerifyReceipt marks a receipt verified; (nil, nil) when not visible.
func (s *Store) VerifyReceipt(ctx context.Context, id string, sc Scope) (*domain.ReceiptRow, error) {
	args := []any{id}
	sql := `UPDATE receipts SET status = 'verified', verified = TRUE WHERE id = $1::uuid` + scopeClause(sc, &args, "user_id") + ` RETURNING id::text`
	var got string
	if err := s.Pool.QueryRow(ctx, sql, args...).Scan(&got); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	return s.ReceiptByID(ctx, got, Scope{CanSeeAll: true})
}

// DeleteReceipt deletes a receipt and returns the removed row, or nil.
func (s *Store) DeleteReceipt(ctx context.Context, id string, sc Scope) (*domain.ReceiptRow, error) {
	row, err := s.ReceiptByID(ctx, id, sc)
	if err != nil || row == nil {
		return nil, err
	}
	args := []any{id}
	sql := `DELETE FROM receipts WHERE id = $1::uuid` + scopeClause(sc, &args, "user_id")
	tag, err := s.Pool.Exec(ctx, sql, args...)
	if err != nil {
		if pgErrCode(err) == foreignKeyViolation {
			return nil, ErrReceiptReferenced
		}
		return nil, err
	}
	if tag.RowsAffected() == 0 {
		return nil, nil
	}
	return row, nil
}

// ReceiptByImageID returns the first receipt linked to an image.
func (s *Store) ReceiptByImageID(ctx context.Context, imageID string) (*domain.ReceiptRow, error) {
	return scanReceipt(s.Pool.QueryRow(ctx, `SELECT `+receiptCols+` FROM receipts r WHERE r.image_id = $1::uuid
		ORDER BY r.created_at ASC, r.id ASC LIMIT 1`, imageID), false)
}

// ListTags returns the tag vocabulary ordered by name.
func (s *Store) ListTags(ctx context.Context) ([]string, error) {
	rows, err := s.Pool.Query(ctx, `SELECT name FROM tags ORDER BY name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []string{}
	for rows.Next() {
		var n string
		if err := rows.Scan(&n); err != nil {
			return nil, err
		}
		out = append(out, n)
	}
	return out, rows.Err()
}

// CreateTag inserts a tag, reporting whether it was newly created.
func (s *Store) CreateTag(ctx context.Context, name string) (bool, error) {
	var got string
	err := s.Pool.QueryRow(ctx, `INSERT INTO tags (name) VALUES ($1) ON CONFLICT (name) DO NOTHING RETURNING name`, name).Scan(&got)
	if errors.Is(err, pgx.ErrNoRows) {
		return false, nil
	}
	return err == nil, err
}

var _ = decimal.Zero

// CategorySuggestion returns only this user's confirmed exact mapping.
func (s *Store) CategorySuggestion(ctx context.Context, userID, merchant, description string) (*string, error) {
	mk, dk := domain.NormalizeHistoryKey(merchant), domain.NormalizeHistoryKey(description)
	if mk == "" || dk == "" {
		return nil, nil
	}
	var c string
	err := s.Pool.QueryRow(ctx, `SELECT category FROM item_category_aliases
		WHERE user_id = $1::uuid AND merchant_key = $2 AND description_key = $3`, userID, mk, dk).Scan(&c)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if domain.IsSpendingCategory(c) && c != "unknown" {
		return &c, nil
	}
	return nil, nil
}
