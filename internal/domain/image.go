package domain

import "time"

// Timestamp serialises as RFC 3339 in UTC with microsecond precision when
// non-zero ("2025-03-04T12:34:56.123456Z").
type Timestamp struct{ time.Time }

// MarshalJSON implements json.Marshaler.
func (t Timestamp) MarshalJSON() ([]byte, error) {
	return []byte(`"` + t.UTC().Format("2006-01-02T15:04:05.999999Z07:00") + `"`), nil
}

// ImagePart is one stored photo of a submission.
type ImagePart struct {
	ImagePath        string  `json:"image_path"`
	ThumbnailPath    *string `json:"thumbnail_path"`
	OriginalFilename *string `json:"original_filename"`
	MediaType        *string `json:"media_type"`
	SizeBytes        *int64  `json:"size_bytes"`
}

// ImageRow is a submission: the first photo plus any additional ones, with
// its analysis workflow state (pending -> processing -> analyzed | failed |
// timed_out | unreadable).
type ImageRow struct {
	ID               string      `json:"id"`
	OriginalFilename *string     `json:"original_filename"`
	MediaType        *string     `json:"media_type"`
	SizeBytes        *int64      `json:"size_bytes"`
	ImagePath        *string     `json:"image_path"`
	ThumbnailPath    *string     `json:"thumbnail_path"`
	Status           string      `json:"status"`
	Error            *string     `json:"error"`
	ReceiptID        *string     `json:"receipt_id"`
	BypassReview     bool        `json:"bypass_review"`
	UserID           *string     `json:"user_id"`
	CreatedAt        *Timestamp  `json:"created_at"`
	AnalyzedAt       *Timestamp  `json:"analyzed_at"`
	ProcessingAt     *Timestamp  `json:"processing_at"`
	AdditionalImages []ImagePart `json:"additional_images"`
	ModelID          *string     `json:"model_id"`
}

// Parts returns the primary photo followed by the additional photos.
func (r ImageRow) Parts() []ImagePart {
	first := ImagePart{
		ThumbnailPath: r.ThumbnailPath, OriginalFilename: r.OriginalFilename,
		MediaType: r.MediaType, SizeBytes: r.SizeBytes,
	}
	if r.ImagePath != nil {
		first.ImagePath = *r.ImagePath
	}
	return append([]ImagePart{first}, r.AdditionalImages...)
}
