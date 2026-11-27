# Multi-photo receipts

Upload one photo for immediate extraction, or select/capture up to 10 ordered
photos (20 MB each) of the same receipt. Include the header and total, with a few
overlapping rows between adjacent photos. Photos can be reordered before upload.

`POST /api/v1/images` accepts repeated multipart `receipt` fields. One photo
retains synchronous processing when a model is reachable. Multiple photos return
202 with `image_id` (the submission UUID) and a `Location` header; the existing
analysis scheduler is signalled immediately. Unreachable providers leave uploads
pending for periodic recovery. The chosen model and review preference are stored.

The frontend waits at `/upload?job=<UUID>` and polls the authenticated image
resource. That URL survives reloads and browser closure. Pending/active/error
submissions are linked from Queue, and completed receipts appear in Receipts.
The receipt review screen includes every source photo. Verification and deletion
handle the additional image files as well as the original.

## Extraction and duplicate verification

Both providers send all ordered photos in one call. For multiple photos, the
extraction schema asks for all line items, their zero-based `source_image`, and
`duplicate_candidates` pairs of `first_index`/`second_index` into `line_items`.
Only model-proposed, ordered suffix/prefix overlaps between adjacent photos are
eligible for removal. Repeated purchases elsewhere remain intact.

`llm.Reconcile` (`internal/llm`) verifies each proposed pair by description,
quantity and total_price: all three must be equal, with case and whitespace
normalization for names. If any pair in an overlap fails verification, that
overlap is retained. Visual identification of
overlap still depends on the model; verification cannot recover omitted rows.

The persisted Receipt schema remains unchanged. The model can instead return
`{"error":{"code":"unreadable","message":"Photo 2 is too blurry"}}`.
This ends extraction without JSON-repair retries or automatic reprocessing.

## Timeouts

`LLM__ANALYSIS_TIMEOUT_SECONDS` (or YAML `llm.analysis_timeout_seconds`) defaults
to 600 seconds and accepts values greater than 0 up to 600. This deadline covers
the entire extraction including JSON-repair attempts, and stays below the
existing 15-minute processing lease. Discovery is bounded to five seconds.
Transport timeouts and deadline expiry mark the submission `timed_out`.
Single-photo uploads return HTTP 504 for timeouts or 422 for unreadable input,
including `image_id`, `status`, and `detail`; asynchronous jobs expose these
outcomes through GET. Neither terminal outcome is automatically retried.
Uploading again is an explicit retry. Cancelling the client request cannot
guarantee that an inference server stops its own computation.

Apply migration 0009 before running the updated app (Docker applies it on startup).
Its parent is the existing working-tree migration 0008; neither earlier migration
is modified by this feature.
