"""Turn the agent's final answer into atomic, externally verifiable claims."""

from pydantic import BaseModel, Field

from browser_use.evidence.claims import Claim, ClaimSet
from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import SystemMessage, UserMessage


class RawClaim(BaseModel):
	"""One claim as returned by the model: text only, because ids and ordering stay ours."""

	text: str = Field(description='A single atomic factual claim, stated so it can be checked without the original answer')


class RawClaimExtraction(BaseModel):
	"""Structured output schema the model must fill in."""

	claims: list[RawClaim] = Field(default_factory=list, description='Every atomic factual claim in the answer, in answer order')


class ClaimExtractionError(RuntimeError):
	"""Raised when claims cannot be extracted.

	Unlike evidence collection, claim extraction feeds the verification pipeline, so a failure
	is reported loudly instead of degrading to an empty claim set.
	"""


_SYSTEM_PROMPT = """You extract the factual claims from a research answer so that each one can later be checked against independent web evidence.

Return every atomic, externally verifiable factual claim the answer makes.

Rules:
1. One claim states exactly one independently verifiable fact. Split parts joined by "and", "but" or commas whenever the parts can be verified separately. For example, "Browser Use is open source and has 100k stars" becomes two claims: "Browser Use is open source." and "Browser Use has 100k stars."
2. Keep every detail that changes the fact: numbers, dates, units, entity names, versions, comparisons, quantities, and qualifying conditions. State each claim self-contained, replacing pronouns with the entity they refer to.
3. Do not extract opinions, subjective advice, greetings, section headings, reasoning steps, plans, action suggestions, or self-referential framing such as "I think" or "you should". None of those can be checked against web evidence.
4. If the answer contains no externally verifiable fact, return an empty claims list.
5. Never add a fact the answer does not state, and never reword a fact into a stronger or weaker one.
6. Do not judge whether a claim is true. Report only what the answer asserts.
7. Keep claims in the order the answer presents them.
"""


class ClaimExtractor:
	"""Split a final answer into atomic claims using a structured-output LLM call.

	The model only supplies claim text. ``claim_id`` and ``order`` are assigned here, because
	identifiers and ordering are bookkeeping the pipeline must control rather than trust the
	model to reproduce. This extractor never reads the evidence store: what a claim asserts is
	independent of what the collected evidence happens to support.
	"""

	def __init__(self, llm: BaseChatModel) -> None:
		self.llm = llm

	async def extract(self, *, task_id: str, task: str, answer: str) -> ClaimSet:
		"""Extract atomic claims from ``answer``.

		Raises:
			ClaimExtractionError: if the model call fails or returns an unusable completion.
		"""
		if not answer.strip():
			return ClaimSet(task_id=task_id, task=task, answer=answer, claims=[])

		user_prompt = f'<task>\n{task}\n</task>\n\n<answer>\n{answer}\n</answer>'
		messages = [SystemMessage(content=_SYSTEM_PROMPT), UserMessage(content=user_prompt)]

		try:
			response = await self.llm.ainvoke(messages, output_format=RawClaimExtraction)
		except Exception as e:
			# Deliberately omits str(e): provider errors often echo the request, which carries the answer and any credentials.
			raise ClaimExtractionError(f'Claim extraction failed: {type(e).__name__}') from e

		extraction = getattr(response, 'completion', None)
		if not isinstance(extraction, RawClaimExtraction):
			raise ClaimExtractionError(
				f'Claim extraction expected {RawClaimExtraction.__name__}, got {type(extraction).__name__}'
			)

		return ClaimSet(task_id=task_id, task=task, answer=answer, claims=self._to_claims(extraction.claims))

	@staticmethod
	def _to_claims(raw_claims: list[RawClaim]) -> list[Claim]:
		"""Assign generated ids and contiguous 1-based order, dropping blank entries."""
		claims: list[Claim] = []
		for raw_claim in raw_claims:
			text = (raw_claim.text or '').strip()
			if not text:
				continue
			claims.append(Claim(text=text, order=len(claims) + 1))
		return claims
