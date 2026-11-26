import json
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, computed_field, field_validator

ClaimPurpose = Literal["inference", "completion"]


class ImagePart(BaseModel):
    image_path: str
    # Assigned only by image_source_parts; file allocation is not a part identity.
    part_id: UUID | None = None
    source_sha256: str | None = None
    photo_index: int | None = Field(default=None, ge=0)
    thumbnail_path: str | None = None
    original_filename: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None


class ImageRow(BaseModel):
    """Represents a single row in the `images` table.

    An image owns its on-disk path and its analysis workflow
    (pending -> processing -> analyzed | failed). A successfully analyzed image is linked
    to a receipt via ``receipt_id``.
    """

    id: UUID
    original_filename: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    image_path: str | None = None
    thumbnail_path: str | None = None
    status: str = "pending"
    error: str | None = None
    receipt_id: UUID | None = None
    bypass_review: bool = False
    user_id: UUID | None = None
    created_at: datetime | None = None
    analyzed_at: datetime | None = None
    processing_at: datetime | None = None
    additional_images: list[ImagePart] = Field(default_factory=list)
    model_id: str | None = None
    generation: int = Field(default=1, ge=1)
    attempt_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1, le=3)
    retry_at: datetime | None = None
    result_code: str | None = None
    wait_reason: str | None = None
    draft_available: bool = False
    effective_options: dict[str, Any] = Field(default_factory=dict, repr=False, exclude=True)

    @field_validator("effective_options", mode="before")
    @classmethod
    def decode_options(cls, value: object) -> object:
        return json.loads(value) if isinstance(value, str) else value

    # Internal fencing/lease fields. Public responses use SubmissionStatus.
    claim_token: UUID | None = Field(default=None, repr=False, exclude=True)
    lease_expires_at: datetime | None = Field(default=None, repr=False, exclude=True)
    claim_purpose: ClaimPurpose | None = Field(default=None, repr=False, exclude=True)
    observation_checkpoint: dict[str, Any] | None = Field(default=None, repr=False, exclude=True)
    unmanaged_classification_unsettled: bool = Field(default=False, repr=False, exclude=True)

    @field_validator("observation_checkpoint", mode="before")
    @classmethod
    def decode_observation_checkpoint(cls, value: object) -> object:
        return json.loads(value) if isinstance(value, str) else value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def observations_available(self) -> bool:
        checkpoint = self.observation_checkpoint
        return checkpoint is not None and checkpoint.get("eligible_generation") == self.generation

    @field_validator("additional_images", mode="before")
    @classmethod
    def decode_parts(cls, value: object) -> object:
        return json.loads(value) if isinstance(value, str) else value

    def parts(self) -> list[ImagePart]:
        """The primary image followed by additional photos, in capture order."""
        first = ImagePart(
            image_path=self.image_path or "",
            thumbnail_path=self.thumbnail_path,
            original_filename=self.original_filename,
            media_type=self.media_type,
            size_bytes=self.size_bytes,
        )
        return [first, *self.additional_images]
