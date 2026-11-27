package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"

	"github.com/tendstead/vision-bill/internal/domain"
)

const imageCols = `id::text, original_filename, media_type, size_bytes, image_path, thumbnail_path, status, error,
	receipt_id::text, bypass_review, user_id::text, created_at, analyzed_at, processing_at, additional_images, model_id`

func scanImage(row pgx.Row) (*domain.ImageRow, error) {
	var r domain.ImageRow
	var created, analyzed, processing *time.Time
	var parts []byte
	if err := row.Scan(&r.ID, &r.OriginalFilename, &r.MediaType, &r.SizeBytes, &r.ImagePath, &r.ThumbnailPath,
		&r.Status, &r.Error, &r.ReceiptID, &r.BypassReview, &r.UserID, &created, &analyzed, &processing, &parts, &r.ModelID); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}
	r.CreatedAt, r.AnalyzedAt, r.ProcessingAt = ts(created), ts(analyzed), ts(processing)
	r.AdditionalImages = []domain.ImagePart{}
	if len(parts) > 0 {
		if err := json.Unmarshal(parts, &r.AdditionalImages); err != nil {
			return nil, fmt.Errorf("decode additional_images: %w", err)
		}
	}
	return &r, nil
}

func ts(t *time.Time) *domain.Timestamp {
	if t == nil {
		return nil
	}
	return &domain.Timestamp{Time: *t}
}

// NewImage are the values of a newly stored submission.
type NewImage struct {
	ImagePath        string
	OriginalFilename *string
	MediaType        *string
	SizeBytes        *int64
	UserID           string
	BypassReview     bool
	ThumbnailPath    *string
	Additional       []domain.ImagePart
	ModelID          *string
}

// StoreImage inserts a pending submission.
func (s *Store) StoreImage(ctx context.Context, n NewImage) (*domain.ImageRow, error) {
	parts := n.Additional
	if parts == nil {
		parts = []domain.ImagePart{}
	}
	raw, err := json.Marshal(parts)
	if err != nil {
		return nil, err
	}
	return scanImage(s.Pool.QueryRow(ctx, `INSERT INTO images
		(original_filename, media_type, size_bytes, image_path, thumbnail_path, status, user_id, bypass_review, additional_images, model_id)
		VALUES ($1,$2,$3,$4,$5,'pending',$6::uuid,$7,$8::jsonb,$9) RETURNING `+imageCols,
		n.OriginalFilename, n.MediaType, n.SizeBytes, n.ImagePath, n.ThumbnailPath, nullIfEmpty(n.UserID), n.BypassReview, string(raw), n.ModelID))
}

func nullIfEmpty(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// ImageByID returns a submission visible to the caller, or nil.
func (s *Store) ImageByID(ctx context.Context, id string, sc Scope) (*domain.ImageRow, error) {
	args := []any{id}
	return scanImage(s.Pool.QueryRow(ctx, `SELECT `+imageCols+` FROM images WHERE id = $1::uuid`+scopeClause(sc, &args, "user_id"), args...))
}

// ListImages lists submissions newest first, optionally filtered by status.
func (s *Store) ListImages(ctx context.Context, statuses []string, limit, offset int, sc Scope) ([]domain.ImageRow, error) {
	var where []string
	var args []any
	if !sc.CanSeeAll && sc.UserID != "" {
		args = append(args, sc.UserID)
		where = append(where, fmt.Sprintf("user_id = $%d::uuid", len(args)))
	}
	if len(statuses) > 0 {
		args = append(args, statuses)
		where = append(where, fmt.Sprintf("status = ANY($%d)", len(args)))
	}
	sql := `SELECT ` + imageCols + ` FROM images`
	if len(where) > 0 {
		sql += " WHERE " + strings.Join(where, " AND ")
	}
	args = append(args, limit, offset)
	sql += " ORDER BY created_at DESC, id DESC LIMIT $" + strconv.Itoa(len(args)-1) + " OFFSET $" + strconv.Itoa(len(args))
	return s.queryImages(ctx, sql, args...)
}

func (s *Store) queryImages(ctx context.Context, sql string, args ...any) ([]domain.ImageRow, error) {
	rows, err := s.Pool.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []domain.ImageRow{}
	for rows.Next() {
		r, err := scanImage(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, *r)
	}
	return out, rows.Err()
}

// claimable matches images that still need analysis, are being retried,
// timed out and whose retry delay has passed, or whose previous worker lease
// (15 minutes) expired.
const claimable = `(status IN ('pending', 'failed')
	OR (status = 'timed_out' AND (retry_after IS NULL OR retry_after <= CURRENT_TIMESTAMP))
	OR (status = 'processing' AND (processing_at IS NULL OR processing_at < CURRENT_TIMESTAMP - INTERVAL '15 minutes')))`

// ListPendingImages returns the analysis queue, oldest first.
func (s *Store) ListPendingImages(ctx context.Context) ([]domain.ImageRow, error) {
	return s.queryImages(ctx, `SELECT `+imageCols+` FROM images WHERE `+claimable+` ORDER BY created_at ASC, id ASC`)
}

// ClaimImage atomically claims an image so only one worker analyses it; nil
// when it is no longer claimable.
func (s *Store) ClaimImage(ctx context.Context, id string) (*domain.ImageRow, error) {
	return scanImage(s.Pool.QueryRow(ctx, `UPDATE images SET status = 'processing', processing_at = CURRENT_TIMESTAMP, error = NULL
		WHERE id = $1::uuid AND `+claimable+` RETURNING `+imageCols, id))
}

// MarkImageAnalyzed links the image to its receipt.
func (s *Store) MarkImageAnalyzed(ctx context.Context, imageID, receiptID string) error {
	_, err := s.Pool.Exec(ctx, `UPDATE images SET status = 'analyzed', receipt_id = $2::uuid, error = NULL,
		analyzed_at = CURRENT_TIMESTAMP, processing_at = NULL WHERE id = $1::uuid`, imageID, receiptID)
	return err
}

// MarkImageFailed records a retryable failure.
func (s *Store) MarkImageFailed(ctx context.Context, imageID, msg string) error {
	_, err := s.Pool.Exec(ctx, `UPDATE images SET status = 'failed', error = $2, processing_at = NULL WHERE id = $1::uuid`, imageID, msg)
	return err
}

// MarkImageTerminal records a final outcome (only "unreadable": the model
// says the photo cannot be read, so retrying would not help).
func (s *Store) MarkImageTerminal(ctx context.Context, imageID, status, msg string) error {
	if status != "unreadable" {
		return errors.New("unsupported terminal analysis status")
	}
	_, err := s.Pool.Exec(ctx, `UPDATE images SET status = $2, error = $3, processing_at = NULL WHERE id = $1::uuid`, imageID, status, msg)
	return err
}

// MarkImageTimedOut records a model timeout. It is never final: the image is
// claimed again once retry_after has passed. The delay doubles with every
// timeout (10 min, 20 min, 40 min, ... capped at 6 h) so a photo the model
// cannot handle in time costs little instead of occupying an inference slot
// for the full deadline every cycle.
func (s *Store) MarkImageTimedOut(ctx context.Context, imageID, msg string) error {
	_, err := s.Pool.Exec(ctx, `UPDATE images SET status = 'timed_out', error = $2, processing_at = NULL,
		analysis_attempts = analysis_attempts + 1,
		retry_after = CURRENT_TIMESTAMP + LEAST(INTERVAL '10 minutes' * power(2, LEAST(analysis_attempts, 10)), INTERVAL '6 hours')
		WHERE id = $1::uuid`, imageID, msg)
	return err
}

// UpdateImagePath sets the on-disk path of the primary photo.
func (s *Store) UpdateImagePath(ctx context.Context, imageID, path string) error {
	_, err := s.Pool.Exec(ctx, `UPDATE images SET image_path = $2 WHERE id = $1::uuid`, imageID, path)
	return err
}

// UpdateImageThumbnailPath sets the primary photo's thumbnail path.
func (s *Store) UpdateImageThumbnailPath(ctx context.Context, imageID, path string) error {
	_, err := s.Pool.Exec(ctx, `UPDATE images SET thumbnail_path = $2 WHERE id = $1::uuid`, imageID, path)
	return err
}

// UpdateAdditionalImages replaces the additional photo list.
func (s *Store) UpdateAdditionalImages(ctx context.Context, imageID string, parts []domain.ImagePart) error {
	if parts == nil {
		parts = []domain.ImagePart{}
	}
	raw, err := json.Marshal(parts)
	if err != nil {
		return err
	}
	_, err = s.Pool.Exec(ctx, `UPDATE images SET additional_images = $2::jsonb WHERE id = $1::uuid`, imageID, string(raw))
	return err
}

// DeleteImage removes the row (files are removed by the caller).
func (s *Store) DeleteImage(ctx context.Context, imageID string) error {
	_, err := s.Pool.Exec(ctx, `DELETE FROM images WHERE id = $1::uuid`, imageID)
	return err
}
