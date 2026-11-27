# 4. Cherry-picks from PR #16 (OCR prototype)

To be scoped in its own session. Notes so it starts cleanly:

- PR #16 ("Use external LLM endpoints with optional PaddleOCR-VL 1.6") is about
  115k added lines in 567 files and is based on the Python backend, which no
  longer exists on `main`. It cannot be merged or rebased; useful parts must be
  re-implemented in Go.
- Do not read the diff as a whole. Start with `git diff --stat` and classify the
  files: application code, the OCR/formatter services, configuration, docs and
  design notes, and generated/vendored/data files (the majority is probably the
  last group).
- Ask per feature whether it is worth porting: external LLM endpoints (overlaps
  with plan 1, step D "connections"), OCR as an optional pipeline, the
  structured-extraction template/formatter. The benchmark capability flag
  `ocr_available` is currently `false` and `pipeline: "ocr"` is rejected;
  schema revision 0020 from the prototype is not part of `main`.
- Keep OCR optional and off by default; any new schema change must be additive.
- A running prototype stack (`vision-bill-pr16-rocm`: database, ROCm formatter,
  llama.cpp, paddleocr) exists on the development machine; its app container is
  stopped.
- The OCR port would be a feature release (minor bump).
