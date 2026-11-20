# E2E test fixtures

Drop receipt image + ground-truth pairs here, then opt them in via
`fixtures.toml`.

Rules:

- Pair = `<name>.png|jpg|jpeg` + `<name>.json` (same basename).
- `<name>.json` must validate as a `Receipt` (see
  `src/vision_bill/model/receipt.py`): required `confidence` (0-100),
  `merchant_name`, `date` (YYYY-MM-DD), `line_items`, `subtotal`, `total`;
  `payment_method` must be one of cash, credit_card, debit_card,
  mobile_payment, check, other, unknown.
- Add the basename (no extension) to `fixtures = [...]` in `fixtures.toml`.
- Your own receipts are gitignored (PII stays local). A synthetic, PII-free
  seed pair (`seed_receipt.png` / `seed_receipt.json`) and deposit/refund pair
  (`deposit_refund.png` / `deposit_refund.json`) are tracked mandatory fixtures.
- Every registered, valid pair becomes one parametrized e2e test case
  (tests/e2e/test_pipeline_e2e.py). Optional invalid private entries are skipped with a warning;
  invalid mandatory fixtures fail preparation.

The strict stub maps explicitly registered original/normalized image hashes to
prepared synthetic replies and rejects unknown inputs. Preparation retains original
bytes, creates aliases with actual production EXIF/white-RGB conversion, and builds
separate complete/partial/unreadable scenarios. Test-only synthetic defaults belong
to these explicit replies, not production inference. Classification is a separate
stub stage; primary printed observations carry no prefilled enrichment. Stubs and
fixture scores do not qualify real OCR hardware or model quality. Each E2E run uses
private copies and a unique app/DB/stub project. The runner seeds synthetic app
settings into a project-owned configuration volume and preserves test edits
through restart. Cleanup is scoped to the runner's resources.

Example (add your own):

    fixtures.toml:  fixtures = ["seed_receipt", "deposit_refund", "my-receipt"]
    my-receipt.jpg  (your photo, gitignored)
    my-receipt.json (what a perfect extraction would return, gitignored)
