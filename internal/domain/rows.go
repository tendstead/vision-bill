package domain

import (
	"math"
	"strconv"
	"strings"
)

// DecimalStr carries a NUMERIC value exactly as PostgreSQL printed it
// (preserving scale, like Python's Decimal) and serialises as a JSON string.
type DecimalStr string

// PyFloat serialises like Python's repr(float): "2.0", "0.19", "1e-05".
type PyFloat float64

// MarshalJSON implements json.Marshaler.
func (f PyFloat) MarshalJSON() ([]byte, error) {
	v := float64(f)
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return []byte("null"), nil
	}
	return []byte(pyFloatRepr(v)), nil
}

func pyFloatRepr(v float64) string {
	if v == 0 {
		if math.Signbit(v) {
			return "-0.0"
		}
		return "0.0"
	}
	s := strconv.FormatFloat(v, 'e', -1, 64) // d.ddde±XX, shortest digits
	mant, expS, _ := strings.Cut(s, "e")
	exp, _ := strconv.Atoi(expS)
	neg := strings.HasPrefix(mant, "-")
	mant = strings.TrimPrefix(mant, "-")
	digits := strings.Replace(mant, ".", "", 1)
	var out string
	if exp < -4 || exp >= 16 {
		m := digits[:1]
		if len(digits) > 1 {
			m += "." + digits[1:]
		}
		sign := "+"
		if exp < 0 {
			sign = "-"
		}
		e := exp
		if e < 0 {
			e = -e
		}
		out = m + "e" + sign + leftPad2(strconv.Itoa(e))
	} else if exp >= 0 {
		if len(digits) <= exp+1 {
			out = digits + strings.Repeat("0", exp+1-len(digits)) + ".0"
		} else {
			out = digits[:exp+1] + "." + digits[exp+1:]
		}
	} else {
		out = "0." + strings.Repeat("0", -exp-1) + digits
	}
	if neg {
		out = "-" + out
	}
	return out
}

func leftPad2(s string) string {
	if len(s) < 2 {
		return "0" + s
	}
	return s
}

// ReceiptRow mirrors the receipts table as returned by the API.
type ReceiptRow struct {
	ID              string      `json:"id"`
	Confidence      int         `json:"confidence"`
	MerchantName    string      `json:"merchant_name"`
	MerchantAddress *string     `json:"merchant_address"`
	ReceiptNumber   *string     `json:"receipt_number"`
	Date            Date        `json:"date"`
	Time            *string     `json:"time"`
	Currency        string      `json:"currency"`
	Category        string      `json:"category"`
	Subtotal        DecimalStr  `json:"subtotal"`
	DiscountTotal   DecimalStr  `json:"discount_total"`
	TaxTotal        DecimalStr  `json:"tax_total"`
	Tip             *DecimalStr `json:"tip"`
	Total           DecimalStr  `json:"total"`
	PaymentMethod   string      `json:"payment_method"`
	CreatedAt       *Date       `json:"created_at"`
	Status          string      `json:"status"`
	ImageID         *string     `json:"image_id"`
	ThumbnailPath   *string     `json:"thumbnail_path"`
	Verified        bool        `json:"verified"`
	UserID          *string     `json:"user_id"`
}

// LineItemRow mirrors the line_items table.
type LineItemRow struct {
	ID                  string     `json:"id"`
	ReceiptID           string     `json:"receipt_id"`
	Description         string     `json:"description"`
	OriginalDescription *string    `json:"original_description"`
	SpendingCategory    string     `json:"spending_category"`
	CategorySource      string     `json:"category_source"`
	Quantity            PyFloat    `json:"quantity"`
	UnitPrice           DecimalStr `json:"unit_price"`
	TotalPrice          DecimalStr `json:"total_price"`
	Tags                []string   `json:"tags"`
	Position            int        `json:"position"`
}

// TaxLineRow mirrors the taxes table.
type TaxLineRow struct {
	ID        string     `json:"id"`
	ReceiptID string     `json:"receipt_id"`
	Name      string     `json:"name"`
	Rate      *PyFloat   `json:"rate"`
	Amount    DecimalStr `json:"amount"`
}

// ReceiptWithDetails is the receipt detail response.
type ReceiptWithDetails struct {
	Receipt       ReceiptRow    `json:"receipt"`
	LineItems     []LineItemRow `json:"line_items"`
	Taxes         []TaxLineRow  `json:"taxes"`
	ImagePath     *string       `json:"image_path"`
	CollectionIDs []string      `json:"collection_ids"`
}
