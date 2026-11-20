# Application package

FastAPI entrypoint: `main.py`. Routers enforce authentication/ownership, services
coordinate durable submission/review/benchmark workflows, providers own database
and configured inference endpoint I/O, and models define exact contracts.
Vision Bill uses an operator-selected LLM and can call an independent OCR service.
Alembic owns schema changes. The current head is `0020`, which adds durable
observation checkpoints for completion-only recovery.

See the [architecture guide](../../AGENTS.md),
[upload/review contract](../../MULTI_IMAGE_RECEIPTS.md) and
[upgrade/verification runbook](../../UPGRADE_AND_VERIFY.md).
