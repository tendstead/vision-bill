# 3. Optional integrations (shopping list, roadbook)

Blocked until the other services are ready. This plan fixes the direction so
nothing is built twice. It builds on roadbook's planning record
(`roadbook/docs/vision-bill-v1.1-plan.md` and
`roadbook/docs/superpowers/plans/2026-11-08-vision-bill-connection.md`), which
already agreed most of the boundary; where this plan differs it says so.

## Goals

- A public, authenticated API: another app uploads a receipt, follows the
  analysis, reads the result and the images.
- Shopping list: upload the receipt of a shopping trip; show the receipts of a
  trip and, from a receipt, the trips/lists that reference it; compare planned
  items with purchased line items (comparison logic lives in the consumer).
- Roadbook: upload fuel receipts the same way; keep the reviewed vehicle record
  in roadbook.
- Navigation entries to the other apps appear when a connection is configured.
- **Everything optional and off by default**, easy to enable by configuration.
  With nothing configured: no routes, no UI, no outbound calls.

## PR #16 overlap (David Hupperich) - reuse for the contract

Three parts of PR #16 are building blocks for the integration API. Credit him
(see the README).

| PR #16 feature | Where | Use here |
| --- | --- | --- |
| **`Idempotency-Key` on upload**, scoped per user, with a payload-conflict error when the same key is used with different content | `src/vision_bill/api/images.py`, `ux_images_owner_upload_key` (`0011`) | The duplicate-safe retry required by roadbook's plan. Port as is and expose it on the integration upload. |
| **`SubmissionStatus`**: an explicit public projection of an upload (status, `terminal`, `draft_available`, `latest_extraction_id`, attempts, `retry_at`, `result_code`, `wait_reason`) | `src/vision_bill/model/workflow.py` | Basis of the integration status shape; internal fields never leave the server. |
| **Review drafts**: extraction ends in a draft that a human finalizes (`DraftPatch` with `expected_revision`, `DraftFinalize` with a `request_key`); status `needs_review` | `src/vision_bill/api/drafts.py`, `src/vision_bill/service/receipt_review.py`, migrations `0012`, `0013`, `0015`, `0018` | Matches the `review` state in roadbook's plan (extracted values are proposals until reviewed). Decide in plan 4 whether to port the draft flow; if yes, the integration status exposes it. |

Plan adjustments that follow from this:

- Step 0 (contract document) starts from his status projection and upload
  idempotency.
- Roadbook's status list (`queued`, `processing`, `review`, `completed`,
  `failed`) maps to his statuses; align the names in the contract.

## Principles (agreed in roadbook's plan)

- Each app stays independently deployable and fully usable without the others.
  No shared database, credentials, cookies or filesystem.
- vision-bill owns the receipt images and extracted data of receipts it
  analysed; consumers keep their own reviewed records and only references.
- vision-bill never calls the consumers (no outbound integration traffic, no
  SSRF surface). Consumers call vision-bill. A later webhook/SSE feature needs
  its own review.
- Consumers mediate browser access: the browser only gets the consumer's own
  URLs, never a vision-bill token.

## Authentication: restricted integration tokens

Rejected idea: "profiles without password" (trust the other app's profile).
A profile id is not an identity; accepting it would let anyone who can reach the
consumer skip vision-bill's authentication, and roadbook's own plan forbids it
(`X-Roadbook-Profile` is never proof of identity). Shopping list has no
authentication at all today.

Chosen design:

1. A signed-in vision-bill user creates an **integration token** in settings:
   name, scopes, optional expiry. The secret is shown once.
2. The user pastes it into the consumer's connection settings; the consumer
   stores it server-side (encrypted).
3. The consumer calls vision-bill with `Authorization: Bearer vbt_...`.
4. Tokens are opaque random values; vision-bill stores only a hash, owner,
   connection id, scopes, created/expires/revoked/last-used. Revocation takes
   effect immediately. Never logged, never returned again.
5. Minimum scopes: `receipts:upload`, `receipts:read` (connection-owned
   receipts only), `images:read` (thumbnails and explicit originals). No
   account-wide listing, no deletion, no administration. Every endpoint checks
   token owner **and** the connection grant; knowing a receipt id is not enough.

A consumer without its own authentication (shopping list) holds one token for
the whole installation: everyone who can open that app sees the linked
receipts. That must be stated in its settings page and in the docs.

## API (additive, versioned under `/api/v1/integrations`)

- `POST /receipts`: multipart upload; `Idempotency-Key` header and a caller
  reference (`{app, kind, ref}`), so a retry after a lost response never creates
  a second receipt. `201`/`202` with `Location`.
- `GET /receipts/{id}`: status (`queued`, `processing`, `review`, `completed`,
  `failed`, `gave_up`), extracted receipt with **stable line-item ids**, source
  revision, caller references.
- `GET /receipts/{id}/thumb` and `/file`.
- References table `receipt_links(receipt_id, connection_id, app, ref_kind,
  ref_id, line_item_ids[])` so several apps can link one receipt, and a
  receipt with fuel and groceries can be linked per line.
- Change detection: `revision` on receipts; consumers re-read, vision-bill never
  pushes. Deleting a linked receipt requires an explicit confirmation that
  lists the linked consumers; consumers keep their own records.
- Rate limits, body/size limits and media-type limits as for normal uploads;
  the analysis goes through the existing queue (no second worker).

## Navigation and "embedding"

- Config lists the connected apps: `integrations.links: [{name, url, icon}]`.
  `GET /system/ui-config` returns them; the navigation shows an entry per link
  only when configured. Deep links: receipt page ↔ the consumer's trip/list
  page using the stored references (`{base_url}/trips/{ref_id}`).
- Real embedding (one Svelte app loading another at runtime, or an iframe) is
  not planned: the apps run on separate origins and versions, the session
  cookie is `SameSite=Lax` and is not sent in cross-site frames, and runtime
  sharing would force identical versions of every app. Navigation plus the
  consumer's own pages (fed by its backend through the API) gives the same
  result without that coupling. Revisit only if a shared embed route with token
  exchange is explicitly designed.

## Configuration

```
INTEGRATIONS__ENABLED=false            # default; routes answer 404 when off
INTEGRATIONS__LINKS=Shopping list|http://host:8080,Roadbook|http://host:8090
```

Tokens are managed in the UI (admin or owner), not in the environment.

## Order of work

0. Contract document `plans/integration-api.md` agreed with the roadbook and
   shopping-list sessions (do not implement before this).
1. Token model + auth middleware + scopes + settings UI, behind
   `INTEGRATIONS__ENABLED`.
2. Upload/status/read endpoints with idempotency and links.
3. Navigation links and deep links.
4. Consumer-side work happens in their own repositories.

## Test plan

- Token lifecycle (hash only stored, revoke, expiry, last-use), cross-user and
  cross-connection access denied, disabled feature returns 404, idempotent
  upload retries, linked-delete confirmation, no outbound traffic when
  disabled, API e2e with a stub consumer.

## Open questions

- Who can create tokens: any user for their own receipts (planned) or admin only?
- Token lifetime default (none / 1 year)?
- Should consumers see receipts uploaded through the normal UI if the user
  grants it (explicit share), or only receipts uploaded through their token
  (planned)?
- Transport: HTTP on the LAN is possible; document HTTPS/reverse proxy as the
  recommended setup, never disable certificate verification.
