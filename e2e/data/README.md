# E2E test fixtures

Drop receipt image + ground-truth pairs here, then opt them in via
`fixtures.txt`.

Rules:

- Pair = `<name>.png|jpg|jpeg` + `<name>.json` (same basename).
- `<name>.json` must validate as a receipt (`domain.ParseReceipt` in
  `internal/domain/receipt.go`): required `confidence` (0-100),
  `merchant_name`, `date` (YYYY-MM-DD), `line_items`, `subtotal`, `total`;
  `payment_method` must be one of cash, credit_card, debit_card,
  mobile_payment, check, other, unknown.
- Add the basename (no extension) on its own line in `fixtures.txt`.
- Your own receipts are gitignored (PII stays local). A synthetic, PII-free
  seed pair (`seed_receipt.png` / `seed_receipt.json`) is tracked so the suite
  always has >=1 case on a fresh clone.
- Every registered, valid pair becomes one parametrized e2e test case
  (`TestPipelinePerFixture` in `e2e/suite/pipeline_test.go`). Missing or invalid entries are skipped
  with a warning, never fatal.

The stub never parses the image — it maps `sha256(image bytes)` to the JSON —
so the image content only needs to be a valid image type (for the app's
content-type sniffing); the JSON is what the "extraction" returns.

Example (add your own):

    fixtures.txt:   one basename per line, e.g. "seed_receipt" and "my-receipt"
    my-receipt.jpg  (your photo, gitignored)
    my-receipt.json (what a perfect extraction would return, gitignored)
