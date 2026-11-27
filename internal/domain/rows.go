package domain

// DecimalStr carries a NUMERIC value exactly as PostgreSQL printed it
// (preserving scale) and serialises as a JSON string.
type DecimalStr string

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
	Quantity            float64    `json:"quantity"`
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
	Rate      *float64   `json:"rate"`
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
