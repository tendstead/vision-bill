"""Completion boundary shared by typed formatting, uploads and experiments."""

from ..model.extraction import InferenceEnvelope, PersistedInferenceEnvelope
from ..model.receipt import Receipt
from .receipt_review import receipt_from_draft


def admit_observations(envelope: InferenceEnvelope | PersistedInferenceEnvelope) -> Receipt:
    """Require complete raw observations and preserve the producer's reconciled receipt."""
    if envelope.draft is None:
        raise ValueError("draft_incomplete")
    receipt_from_draft(envelope.draft)
    if envelope.receipt is None:
        raise ValueError("receipt_incomplete")
    return envelope.receipt
