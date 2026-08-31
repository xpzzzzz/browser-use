"""Claim models for the WebEvidence verification pipeline."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints
from uuid_extensions import uuid7str

# A claim is only worth verifying if it says something, so blank text is rejected at the model boundary.
NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Claim(BaseModel):
	"""One atomic, externally verifiable factual statement taken from the final answer."""

	claim_id: str = Field(default_factory=uuid7str, description='Unique id of this claim')
	text: NonBlankString = Field(description='The claim itself, verifiable on its own')
	order: int = Field(ge=1, description='1-based position of the claim in the answer')


class ClaimSet(BaseModel):
	"""The atomic claims extracted from one answer of one agent task.

	This intentionally carries no evidence linkage yet: alignment, support scores and
	relations belong to later phases.
	"""

	task_id: str = Field(description='Agent task the answer belongs to')
	task: str = Field(description='Original task prompt')
	answer: str = Field(description='Final natural-language answer the claims were extracted from')
	claims: list[Claim] = Field(default_factory=list, description='Atomic claims, in the order the answer makes them')
