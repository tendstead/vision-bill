package domain

import (
	"regexp"
	"strings"
)

// Collection is a per-user group a whole receipt may belong to. Field order
// lists editable fields first.
type Collection struct {
	Name      string  `json:"name"`
	Color     *string `json:"color"`
	StartDate *Date   `json:"start_date"`
	EndDate   *Date   `json:"end_date"`
	ID        string  `json:"id"`
	Active    bool    `json:"active"`
	CreatedAt *Date   `json:"created_at"`
}

// CurrencyTotal is a per-currency sum of receipt totals.
type CurrencyTotal struct {
	Currency string     `json:"currency"`
	Total    DecimalStr `json:"total"`
}

// CollectionSummary is a list entry.
type CollectionSummary struct {
	Collection
	ReceiptCount int             `json:"receipt_count"`
	Totals       []CurrencyTotal `json:"totals"`
}

// CollectionDetail includes member receipts.
type CollectionDetail struct {
	Collection
	Totals   []CurrencyTotal `json:"totals"`
	Receipts []ReceiptRow    `json:"receipts"`
}

var hexColor = regexp.MustCompile(`^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$`)

// IsValidHexColor reports whether v is a CSS 3- or 6-digit hex colour.
func IsValidHexColor(v string) bool { return hexColor.MatchString(v) }

// NormalizeCollectionName collapses and trims whitespace.
func NormalizeCollectionName(raw string) string {
	return strings.TrimSpace(strings.Join(pyFields(raw), " "))
}
