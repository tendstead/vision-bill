package domain

// ProductPurchase is one verified line-item purchase matching a product search.
type ProductPurchase struct {
	ReceiptID    string     `json:"receipt_id"`
	Description  string     `json:"description"`
	MerchantName string     `json:"merchant_name"`
	Date         Date       `json:"date"`
	Time         *string    `json:"time"`
	Quantity     float64    `json:"quantity"`
	UnitPrice    DecimalStr `json:"unit_price"`
	Currency     string     `json:"currency"`
}

// ProductSearchResponse is purchase history plus a unit-price summary.
type ProductSearchResponse struct {
	Query         string            `json:"query"`
	Purchases     []ProductPurchase `json:"purchases"`
	LatestPrice   *DecimalStr       `json:"latest_price"`
	CheapestPrice *DecimalStr       `json:"cheapest_price"`
	AveragePrice  *DecimalStr       `json:"average_price"`
	Currency      *string           `json:"currency"`
}

// CurrencyStatistics aggregates verified receipts of one currency.
type CurrencyStatistics struct {
	Currency     string     `json:"currency"`
	ReceiptCount int        `json:"receipt_count"`
	Total        DecimalStr `json:"total"`
	Average      DecimalStr `json:"average"`
	Median       DecimalStr `json:"median"`
	Minimum      DecimalStr `json:"minimum"`
	Maximum      DecimalStr `json:"maximum"`
	Subtotal     DecimalStr `json:"subtotal"`
	Discounts    DecimalStr `json:"discounts"`
	Taxes        DecimalStr `json:"taxes"`
	Tips         DecimalStr `json:"tips"`
}

// NamedStatistics aggregates by merchant, category or payment method.
type NamedStatistics struct {
	Name         string     `json:"name"`
	Currency     string     `json:"currency"`
	ReceiptCount int        `json:"receipt_count"`
	Total        DecimalStr `json:"total"`
	Average      DecimalStr `json:"average"`
}

// WeekdayStatistics aggregates by ISO weekday (1 = Monday).
type WeekdayStatistics struct {
	Weekday      int        `json:"weekday"`
	Currency     string     `json:"currency"`
	ReceiptCount int        `json:"receipt_count"`
	Total        DecimalStr `json:"total"`
	Average      DecimalStr `json:"average"`
}

// WeeklyStatistics aggregates by week start date.
type WeeklyStatistics struct {
	WeekStart    Date       `json:"week_start"`
	Currency     string     `json:"currency"`
	ReceiptCount int        `json:"receipt_count"`
	Total        DecimalStr `json:"total"`
	Average      DecimalStr `json:"average"`
}

// ReceiptStatistics is the verified-only spending summary.
type ReceiptStatistics struct {
	VerifiedReceiptCount int                  `json:"verified_receipt_count"`
	Currencies           []CurrencyStatistics `json:"currencies"`
	Merchants            []NamedStatistics    `json:"merchants"`
	Categories           []NamedStatistics    `json:"categories"`
	PaymentMethods       []NamedStatistics    `json:"payment_methods"`
	Weekdays             []WeekdayStatistics  `json:"weekdays"`
	WeeklySpending       []WeeklyStatistics   `json:"weekly_spending"`
}

// ItemCategoryStatistics is one category's spending and share.
type ItemCategoryStatistics struct {
	Category string      `json:"category"`
	Currency string      `json:"currency"`
	Total    DecimalStr  `json:"total"`
	Share    *DecimalStr `json:"share"`
}

// MonthlyItemStatistics is a category's spending in one calendar month.
type MonthlyItemStatistics struct {
	Category string      `json:"category"`
	Currency string      `json:"currency"`
	Total    DecimalStr  `json:"total"`
	Share    *DecimalStr `json:"share"`
	Month    Date        `json:"month"`
}

// ItemReconciliation reconciles receipt totals with item spending.
type ItemReconciliation struct {
	Currency         string     `json:"currency"`
	ReceiptTotal     DecimalStr `json:"receipt_total"`
	ItemTotal        DecimalStr `json:"item_total"`
	DepositTotal     DecimalStr `json:"deposit_total"`
	UnallocatedTotal DecimalStr `json:"unallocated_total"`
}

// ItemStatistics are category shares and monthly trends per currency.
type ItemStatistics struct {
	Months         int                      `json:"months"`
	DateFrom       Date                     `json:"date_from"`
	DateTo         Date                     `json:"date_to"`
	Categories     []ItemCategoryStatistics `json:"categories"`
	Monthly        []MonthlyItemStatistics  `json:"monthly"`
	Reconciliation []ItemReconciliation     `json:"reconciliation"`
}
