# 1. Comparable benchmarks

## Goals

- Add a new LLM (or a new connection) later and compare it with earlier runs.
- Re-run an old benchmark after updating a model and see whether it improved.
- Measure how much a setting changes the result: temperature, `think`, and
  inference-server settings such as GPU offloading or context size.
- Show the differences as charts and tables where every receipt is a link.
- Run efficiently: all receipts of one model before the next model is loaded.

## Done

- **Model-major execution** (commit `90921adc`, branch `feat/benchmark-compare`):
  tasks are leased run by run, then model by model (in the requested order),
  then by receipt, so the inference server loads each model once per run.

## What already exists (reuse, do not rebuild)

- A run freezes its inputs: expected receipts, image SHA-256, prompt hashes,
  validation-schema hash, endpoint fingerprint, options (`experiment_manifest`).
- Runs record `dataset_fingerprint`, `model_digests`, `prompt_version`,
  `scoring_version`, council policy. The original benchmark design
  (local `docs/llm-benchmark-plan.md`, not tracked in Git) defines two runs as **directly comparable
  only when dataset fingerprint, model digest, prompt/schema version, scoring
  version and council policy match**. The compare feature uses exactly that
  rule and explains which part differs when it does not match (for example a
  changed model digest = "the model was updated", which is the interesting case).
- Per-task results with scores (`strict`, `normalized`, `classification`,
  `ordered_items`), latency, attempts and Ollama telemetry (`load_ms`,
  `prompt_eval_ms`, `eval_ms`, token counts).
- Transient re-evaluation of a single receipt.

## Design

### A. Data (migration `003_benchmark_compare.sql`, additive)

- `benchmark_runs.parent_run_id uuid NULL REFERENCES benchmark_runs(id) ON DELETE SET NULL`
  (the run this one was derived from).
- `benchmark_runs.label text NULL` (free text, e.g. "ngl 20, ctx 8k, GPU only").
- `benchmark_runs.sweep_id uuid NULL`, `variant_label text NULL`,
  `variant_index int NULL` (runs created together by a sweep).
- No change to the task primary key. A sweep is a set of runs that share the
  frozen inputs and differ in options; the lease order becomes
  *sweep (or run) first, then model position, then variant index, then receipt*,
  so each model is still loaded once per sweep.

### B. API

- `POST /benchmarks/{id}/rerun`: body overrides `model_ids`, `experiment`
  options, `label`, `provider_options`. Reuses the frozen receipt set and
  expected values of the parent (receipts that were deleted or changed since are
  reported, not silently dropped) and records `parent_run_id`.
- `POST /benchmarks` accepts `variants: [{label, experiment: {...}}]` and
  creates the sweep (for example temperature 0.0 / 0.1 / 0.7).
- `GET /benchmarks/compare?runs=a,b,c`: per receipt and per `model + variant`:
  scores, latency, success, attempts, model digest; summary deltas against the
  baseline (the first run or `parent_run_id`); a `comparable` block with the
  reasons when the strict rule is not met.
- Admin only, like the rest of `/benchmarks`.

### C. Provider options

- Ollama: whitelisted pass-through of per-request `options` (`num_gpu`,
  `num_ctx`, `num_thread`, `top_p`, `top_k`, `seed`, `repeat_penalty`). Stored
  in the manifest and part of the comparability fingerprint.
- OpenAI-compatible endpoints: only the whitelisted standard fields (`seed`,
  `top_p`); everything else is reported as "not supported by this provider".
- Settings that live in the inference server's start flags (llama.cpp `-ngl`,
  Ollama environment) cannot be changed by vision-bill; the run `label` records
  them so runs stay comparable by hand.

### D. Connections (separate step, after A-C and the UI)

- Named LLM endpoints in config (`llm.connections: [{name, provider, host,
  api_key_env}]`; secrets only through environment variables, never in the YAML
  file or the manifest, which stores only the endpoint fingerprint).
- A run entry is `connection/model`; the digests and fingerprints are recorded
  per connection.
- Everything stays optional: without `llm.connections` behaviour is unchanged.

### E. Frontend

- **SVG charts, no chart library.** Write a small `BenchmarkChart` component set
  (see licensing note below). Charts:
  1. *Score per receipt*: one row per receipt, a coloured dot per model (and
     variant); each dot is a link to the receipt/image; hover/focus pins the
     values; keyboard accessible; description text for screen readers.
  2. *Model summary bars*: average score, latency, failures.
  3. *Run-to-run difference* (dumbbell): previous and current score for the same
     model on one line per receipt, green if better, red if worse; totals table
     with deltas.
  4. *Latency strip*: per-receipt dots so slow outliers stand out (separate
     load time from generation time when telemetry is available).
- Tables link every receipt id to its receipt/image page, in the results,
  compare and re-evaluation views.
- Compare page: pick runs (defaults to the parent chain), shows the
  comparability result first.
- The existing Chart.js statistics page is left alone.

### Licensing note for the chart code

Both projects are under the Elastic License 2.0 (vision-bill from 0.2.1), so
roadbook's `web/src/lib/Chart.svelte` and `chart-layout.ts` may be ported with
their notices kept. The dot plot differs from roadbook's time series, so expect
a rewrite of the layout parts; a shared component in `@tendstead/ui` is an
option if more apps need it.

## Test plan

- Store: lease order for several models, several runs, sweeps and retries.
- Compare: comparable vs. not comparable (digest change, dataset change,
  prompt-version change), missing receipts, rerun with the same model.
- Provider options: whitelist, rejected keys, manifest/fingerprint effect.
- Playwright: charts render, a dot link navigates to the right receipt.
- API e2e: create run, rerun, compare through the stub LLM.

## Order of work

1. Migration + rerun + compare API (no UI) with tests.
2. Provider options + sweeps.
3. Frontend: tables with links, then charts, then the compare page.
4. Connections (D), if still wanted.

## Open questions

- Is a sweep one object in the UI or just several linked runs? (Plan assumes
  linked runs sharing `sweep_id`.)
- Should rerun keep the *old expected values* (fair comparison) or re-read the
  receipts as they are now (the plan keeps the old ones and reports changes)?
