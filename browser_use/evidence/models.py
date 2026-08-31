"""Pydantic models for the WebEvidence data layer."""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from uuid_extensions import uuid7str


class EvidenceNode(BaseModel):
	"""A single unit of web evidence captured while the agent worked on a step.

	It records what the page looked like (url, title, text, screenshot) at the
	moment the agent acted, so later stages can trace a claim back to its source.
	"""

	# identity
	evidence_id: str = Field(default_factory=uuid7str, description='Unique id of this evidence node')
	task_id: str = Field(description='Agent task id this evidence belongs to')
	step_number: int = Field(ge=1, description='1-based agent step that produced this evidence')

	# observed page content
	url: str = Field(description='Page url the evidence was captured from')
	title: str = Field(default='', description='Page title at capture time')
	text: str = Field(default='', description='Extracted text content of the page')
	screenshot_path: str | None = Field(default=None, description='Local path to the page screenshot, if captured')

	# provenance
	action_names: list[str] = Field(default_factory=list, description='Actions the agent performed on that step')
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description='UTC capture timestamp')
	metadata: dict[str, Any] = Field(default_factory=dict, description='Free-form extra information about the evidence')
