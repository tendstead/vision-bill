package store

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"

	"github.com/tendstead/vision-bill/internal/domain"
)

const collectionCols = `c.id::text, c.name, c.color, c.start_date, c.end_date, c.active,
	(c.created_at AT TIME ZONE 'UTC')::date`

func scanCollection(row pgx.Row, extra ...any) (*domain.Collection, error) {
	var c domain.Collection
	var start, end, created pgtype.Date
	dest := append([]any{&c.ID, &c.Name, &c.Color, &start, &end, &c.Active, &created}, extra...)
	if err := row.Scan(dest...); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	c.StartDate, c.EndDate, c.CreatedAt = optDate(start), optDate(end), optDate(created)
	return &c, nil
}

func optDate(d pgtype.Date) *domain.Date {
	if !d.Valid {
		return nil
	}
	return &domain.Date{Time: d.Time}
}

// ErrNotFound reports a missing or inaccessible referenced entity.
var ErrNotFound = errors.New("not found")

// CollectionsFind returns the user's collection with this name (case-insensitive).
func (s *Store) FindCollectionByName(ctx context.Context, userID, name string) (*domain.Collection, error) {
	return scanCollection(s.Pool.QueryRow(ctx, `SELECT `+collectionCols+` FROM collections c
		WHERE c.user_id = $1::uuid AND lower(c.name) = lower($2)`, userID, name))
}

// CreateCollection inserts a collection, deactivating siblings first when active.
func (s *Store) CreateCollection(ctx context.Context, userID, name string, color *string, start, end *time.Time, active bool) (*domain.Collection, error) {
	var out *domain.Collection
	err := s.inTx(ctx, func(tx pgx.Tx) error {
		if active {
			if _, err := tx.Exec(ctx, `UPDATE collections SET active = FALSE WHERE user_id = $1::uuid`, userID); err != nil {
				return err
			}
		}
		var id string
		if err := tx.QueryRow(ctx, `INSERT INTO collections (user_id, name, color, start_date, end_date, active)
			VALUES ($1::uuid,$2,$3,$4,$5,$6) RETURNING id::text`, userID, name, color, start, end, active).Scan(&id); err != nil {
			return err
		}
		c, err := scanCollection(tx.QueryRow(ctx, `SELECT `+collectionCols+` FROM collections c WHERE c.id = $1::uuid`, id))
		out = c
		return err
	})
	return out, err
}

func (s *Store) totalsFor(ctx context.Context, q interface {
	Query(context.Context, string, ...any) (pgx.Rows, error)
}, id string) ([]domain.CurrencyTotal, error) {
	rows, err := q.Query(ctx, `SELECT r.currency, SUM(r.total)::text AS total
		FROM receipt_collections rc JOIN receipts r ON r.id = rc.receipt_id
		WHERE rc.collection_id = $1::uuid GROUP BY r.currency ORDER BY SUM(r.total) DESC`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []domain.CurrencyTotal{}
	for rows.Next() {
		var t domain.CurrencyTotal
		var total string
		if err := rows.Scan(&t.Currency, &total); err != nil {
			return nil, err
		}
		t.Total = domain.DecimalStr(total)
		out = append(out, t)
	}
	return out, rows.Err()
}

// ListCollections returns collections ordered by name with counts and totals.
func (s *Store) ListCollections(ctx context.Context, sc Scope) ([]domain.CollectionSummary, error) {
	args := []any{}
	sql := `SELECT ` + collectionCols + `, (SELECT COUNT(*)::int FROM receipt_collections rc WHERE rc.collection_id = c.id) AS cnt
		FROM collections c WHERE 1=1` + scopeClause(sc, &args, "c.user_id") + ` ORDER BY c.name`
	rows, err := s.Pool.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	out := []domain.CollectionSummary{}
	for rows.Next() {
		var cnt int
		c, err := scanCollection(rows, &cnt)
		if err != nil {
			rows.Close()
			return nil, err
		}
		out = append(out, domain.CollectionSummary{Collection: *c, ReceiptCount: cnt, Totals: []domain.CurrencyTotal{}})
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}
	for i := range out {
		if out[i].ReceiptCount > 0 {
			t, err := s.totalsFor(ctx, s.Pool, out[i].ID)
			if err != nil {
				return nil, err
			}
			out[i].Totals = t
		}
	}
	return out, nil
}

// GetCollection returns a collection visible to the caller, or nil.
func (s *Store) GetCollection(ctx context.Context, id string, sc Scope) (*domain.Collection, error) {
	args := []any{id}
	return scanCollection(s.Pool.QueryRow(ctx, `SELECT `+collectionCols+` FROM collections c WHERE c.id = $1::uuid`+scopeClause(sc, &args, "c.user_id"), args...))
}

// CollectionDetail returns a collection with totals and member receipts.
func (s *Store) CollectionDetail(ctx context.Context, id string, sc Scope) (*domain.CollectionDetail, error) {
	c, err := s.GetCollection(ctx, id, sc)
	if err != nil || c == nil {
		return nil, err
	}
	totals, err := s.totalsFor(ctx, s.Pool, id)
	if err != nil {
		return nil, err
	}
	args := []any{id}
	rows, err := s.Pool.Query(ctx, `SELECT `+receiptCols+` FROM receipts r JOIN receipt_collections rc ON rc.receipt_id = r.id
		WHERE rc.collection_id = $1::uuid`+scopeClause(sc, &args, "r.user_id")+` ORDER BY r.date DESC`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	receipts := []domain.ReceiptRow{}
	for rows.Next() {
		r, err := scanReceipt(rows, false)
		if err != nil {
			return nil, err
		}
		receipts = append(receipts, *r)
	}
	return &domain.CollectionDetail{Collection: *c, Totals: totals, Receipts: receipts}, rows.Err()
}

// ActiveCollection returns the (first) active collection visible to the caller.
func (s *Store) ActiveCollection(ctx context.Context, sc Scope) (*domain.Collection, error) {
	args := []any{}
	return scanCollection(s.Pool.QueryRow(ctx, `SELECT `+collectionCols+` FROM collections c WHERE c.active = TRUE`+scopeClause(sc, &args, "c.user_id")+` LIMIT 1`, args...))
}

// UpdateCollection writes all four editable fields; nil means NULL.
func (s *Store) UpdateCollection(ctx context.Context, id string, sc Scope, name string, color *string, start, end *time.Time) (*domain.Collection, error) {
	args := []any{id, name, color, start, end}
	var got string
	err := s.Pool.QueryRow(ctx, `UPDATE collections c SET name = $2, color = $3, start_date = $4, end_date = $5
		WHERE c.id = $1::uuid`+scopeClause(sc, &args, "c.user_id")+` RETURNING c.id::text`, args...).Scan(&got)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return s.GetCollection(ctx, got, Scope{CanSeeAll: true})
}

// DeleteCollection removes a collection; false when not visible.
func (s *Store) DeleteCollection(ctx context.Context, id string, sc Scope) (bool, error) {
	args := []any{id}
	tag, err := s.Pool.Exec(ctx, `DELETE FROM collections c WHERE c.id = $1::uuid`+scopeClause(sc, &args, "c.user_id"), args...)
	return tag.RowsAffected() > 0, err
}

// CollectionExists reports whether the caller can see the collection.
func (s *Store) CollectionExists(ctx context.Context, id string, sc Scope) (bool, error) {
	args := []any{id}
	var x int
	err := s.Pool.QueryRow(ctx, `SELECT 1 FROM collections c WHERE c.id = $1::uuid`+scopeClause(sc, &args, "c.user_id"), args...).Scan(&x)
	if errors.Is(err, pgx.ErrNoRows) {
		return false, nil
	}
	return err == nil, err
}

// ActivateCollection marks one collection active, deactivating the caller's others.
func (s *Store) ActivateCollection(ctx context.Context, id string, sc Scope) (*domain.Collection, error) {
	var ok bool
	err := s.inTx(ctx, func(tx pgx.Tx) error {
		if !sc.CanSeeAll && sc.UserID != "" {
			var x int
			if err := tx.QueryRow(ctx, `SELECT 1 FROM collections WHERE id = $1::uuid AND user_id = $2::uuid`, id, sc.UserID).Scan(&x); err != nil {
				if errors.Is(err, pgx.ErrNoRows) {
					return nil
				}
				return err
			}
			if _, err := tx.Exec(ctx, `UPDATE collections SET active = FALSE WHERE user_id = $1::uuid`, sc.UserID); err != nil {
				return err
			}
		}
		args := []any{id}
		var got string
		if err := tx.QueryRow(ctx, `UPDATE collections c SET active = TRUE WHERE c.id = $1::uuid`+scopeClause(sc, &args, "c.user_id")+` RETURNING c.id::text`, args...).Scan(&got); err != nil {
			if errors.Is(err, pgx.ErrNoRows) {
				return nil
			}
			return err
		}
		ok = true
		return nil
	})
	if err != nil || !ok {
		return nil, err
	}
	return s.GetCollection(ctx, id, Scope{CanSeeAll: true})
}

// DeactivateCollection marks a collection inactive.
func (s *Store) DeactivateCollection(ctx context.Context, id string, sc Scope) (*domain.Collection, error) {
	args := []any{id}
	var got string
	err := s.Pool.QueryRow(ctx, `UPDATE collections c SET active = FALSE WHERE c.id = $1::uuid`+scopeClause(sc, &args, "c.user_id")+` RETURNING c.id::text`, args...).Scan(&got)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return s.GetCollection(ctx, got, Scope{CanSeeAll: true})
}

// AssignReceipt links a receipt to a collection (idempotent).
func (s *Store) AssignReceipt(ctx context.Context, collectionID, receiptID string) error {
	_, err := s.Pool.Exec(ctx, `INSERT INTO receipt_collections (receipt_id, collection_id) VALUES ($1::uuid,$2::uuid)
		ON CONFLICT (receipt_id, collection_id) DO NOTHING`, receiptID, collectionID)
	return err
}

// UnassignReceipt removes a link; false when it did not exist.
func (s *Store) UnassignReceipt(ctx context.Context, collectionID, receiptID string) (bool, error) {
	tag, err := s.Pool.Exec(ctx, `DELETE FROM receipt_collections WHERE receipt_id = $1::uuid AND collection_id = $2::uuid`, receiptID, collectionID)
	return tag.RowsAffected() > 0, err
}

// SetReceiptCollections replaces a receipt's memberships after verifying that
// the receipt and every collection are visible to the caller. It returns
// ErrNotFound with a message naming the missing entity.
func (s *Store) SetReceiptCollections(ctx context.Context, receiptID string, collectionIDs []string, sc Scope) error {
	r, err := s.ReceiptByID(ctx, receiptID, sc)
	if err != nil {
		return err
	}
	if r == nil {
		return fmt.Errorf("Receipt not found: %w", ErrNotFound)
	}
	for _, c := range collectionIDs {
		ok, err := s.CollectionExists(ctx, c, sc)
		if err != nil {
			return err
		}
		if !ok {
			return fmt.Errorf("Collection not found: %w", ErrNotFound)
		}
	}
	return s.inTx(ctx, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, `DELETE FROM receipt_collections WHERE receipt_id = $1::uuid`, receiptID); err != nil {
			return err
		}
		for _, c := range collectionIDs {
			if _, err := tx.Exec(ctx, `INSERT INTO receipt_collections (receipt_id, collection_id) VALUES ($1::uuid,$2::uuid)
				ON CONFLICT (receipt_id, collection_id) DO NOTHING`, receiptID, c); err != nil {
				return err
			}
		}
		return nil
	})
}
