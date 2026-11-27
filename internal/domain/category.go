// Package domain holds the receipt schema, validation and exact money maths.
package domain

import (
	"strings"
	"unicode"

	"golang.org/x/text/cases"
)

// SpendingCategories are the stable per-line-item category codes, in the
// order of the Python definition.
var SpendingCategories = []string{
	"unknown", "other", "milk", "cheese", "yogurt", "eggs", "butter_spreads",
	"bread_bakery", "cereals", "pasta_rice_grains", "vegetables", "fruit",
	"meat", "fish", "ready_meals", "snacks_sweets", "coffee_tea", "water",
	"soft_drinks", "alcohol", "sauces_spices", "cleaning", "household",
	"personal_care", "pet", "health", "clothing", "electronics", "hardware",
	"restaurant", "fuel", "services", "deposit",
}

// CategoryDefinitions describe each spending category (used in prompts).
var CategoryDefinitions = map[string]string{
	"unknown":           "Could not tell from the receipt",
	"other":             "Known item outside the other categories",
	"milk":              "Dairy milk and plant milk alternatives",
	"cheese":            "Cheese",
	"yogurt":            "Yogurt and quark",
	"eggs":              "Eggs",
	"butter_spreads":    "Butter, margarine, and spreads",
	"bread_bakery":      "Bread and bakery items",
	"cereals":           "Breakfast cereals and muesli",
	"pasta_rice_grains": "Pasta, rice, and grains",
	"vegetables":        "Vegetables",
	"fruit":             "Fruit",
	"meat":              "Meat",
	"fish":              "Fish and seafood",
	"ready_meals":       "Prepared meals",
	"snacks_sweets":     "Snacks and sweets",
	"coffee_tea":        "Coffee and tea",
	"water":             "Water",
	"soft_drinks":       "Soft drinks and juice",
	"alcohol":           "Alcoholic drinks",
	"sauces_spices":     "Sauces, spices, and condiments",
	"cleaning":          "Cleaning products",
	"household":         "Household supplies",
	"personal_care":     "Personal care and hygiene",
	"pet":               "Pet products",
	"health":            "Medicines and health products",
	"clothing":          "Clothing",
	"electronics":       "Electronics",
	"hardware":          "Tools and hardware",
	"restaurant":        "Restaurant food and drinks",
	"fuel":              "Fuel",
	"services":          "Services",
	"deposit":           "Container deposit or refund",
}

// ReceiptCategories are the whole-receipt categories.
var ReceiptCategories = []string{"grocery", "electronics", "clothing", "restaurant", "fuel", "pharmacy", "entertainment", "other"}

// PaymentMethods are the allowed payment method values.
var PaymentMethods = []string{"cash", "credit_card", "debit_card", "mobile_payment", "check", "other", "unknown"}

var categorySet = func() map[string]bool {
	m := map[string]bool{}
	for _, c := range SpendingCategories {
		m[c] = true
	}
	return m
}()

// IsSpendingCategory reports whether code is a known spending category.
func IsSpendingCategory(code string) bool { return categorySet[code] }

var folder = cases.Fold()

// Fold is Unicode case folding (Python's str.casefold).
func Fold(s string) string { return folder.String(s) }

// pyFields splits on Unicode whitespace like Python's str.split().
func pyFields(s string) []string {
	return strings.FieldsFunc(s, unicode.IsSpace)
}

// NormalizeHistoryKey collapses whitespace and case-folds (exact-history key).
func NormalizeHistoryKey(s string) string {
	return Fold(strings.Join(pyFields(s), " "))
}

// maxTagLength caps a single tag in characters.
const maxTagLength = 100

// NormalizeTags trims, collapses whitespace, drops empties and over-long
// values, and dedupes case-insensitively, keeping first spellings.
func NormalizeTags(tags []string) []string {
	seen := map[string]bool{}
	clean := []string{}
	for _, tag := range tags {
		name := strings.Join(pyFields(tag), " ")
		if name == "" || len([]rune(name)) > maxTagLength {
			continue
		}
		key := Fold(name)
		if seen[key] {
			continue
		}
		seen[key] = true
		clean = append(clean, name)
	}
	return clean
}
