"""LLM semantic reranking of the Phase 4A lexical candidates.

This stage asks one question per claim: how semantically relevant is each candidate to this
claim? It is still retrieval, never verification. The model must not decide whether a claim is
true, false, supported, contradicted or partial, and evidence that disagrees with the claim may
legitimately be just as relevant as evidence that agrees with it.

The reranker is model agnostic: it depends only on the ``BaseChatModel`` abstraction and one
Pydantic structured-output schema, so the same wiring works for whatever model a deployment is
configured with.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from browser_use.evidence.alignment import AlignmentResult, ClaimAlignment, EvidenceMatch
from browser_use.evidence.claims import Claim, ClaimSet
from browser_use.evidence.models import EvidenceNode

if TYPE_CHECKING:
	from browser_use.llm.base import BaseChatModel

# Deterministic prompt budget marker, mirroring the Phase 4A clipping convention.
_TRUNCATION_MARKER = ' [...truncated for length]'


class EvidenceRerankingError(RuntimeError):
	"""Raised when semantic reranking cannot produce a trustworthy result.

	Semantic scores are only meaningful if they line up one-to-one with the lexical candidates they
	were computed for, so an id or claim mismatch is an error rather than something to paper over.
	Falling back to the plain lexical result is an orchestration decision and belongs to the caller.
	"""


class RawSemanticEvidenceScore(BaseModel):
	"""One model-supplied relevance score. The model supplies text and score; identifiers stay ours."""

	evidence_id: str = Field(description='The evidence_id of a candidate, copied verbatim from the prompt')
	relevance_score: float = Field(ge=0.0, le=1.0, description='Semantic relevance of this candidate to the claim')


class RawSemanticReranking(BaseModel):
	"""Structured output schema: exactly one relevance score per candidate of one claim."""

	scores: list[RawSemanticEvidenceScore] = Field(
		default_factory=list,
		description='One entry for every candidate in the prompt, same ids, none dropped, none invented',
	)


class RerankedEvidenceMatch(BaseModel):
	"""One candidate after semantic rescoring, with both signals kept visible side by side."""

	evidence_id: str = Field(description='EvidenceNode.evidence_id of the candidate')
	lexical_score: float = Field(ge=0.0, le=1.0, description='Phase 4A relevance score, preserved unchanged')
	semantic_score: float = Field(ge=0.0, le=1.0, description='Model-judged semantic relevance')
	combined_score: float = Field(ge=0.0, le=1.0, description='Weighted blend of the lexical and semantic scores')
	rank: int = Field(ge=1, description='1-based rank within this claim, assigned after the final sort')


class ClaimReranking(BaseModel):
	"""Reranked candidates for one claim; an empty list means the claim had no lexical candidate."""

	claim_id: str = Field(description='Claim.claim_id this reranking belongs to, copied unchanged')
	matches: list[RerankedEvidenceMatch] = Field(default_factory=list, description='Candidates ordered by combined score')


class RerankingResult(BaseModel):
	"""Semantic reranking outcome for a whole ClaimSet."""

	task_id: str = Field(description='Task the claims and evidence belong to')
	rerankings: list[ClaimReranking] = Field(default_factory=list, description='One entry per claim, in claim order')


_SEMANTIC_RERANKING_SYSTEM_PROMPT = """You are a semantic evidence reranker. For one factual claim you are given several evidence candidates, and you score how semantically relevant each candidate is to that claim.

Score meaning:
- 1.0: the candidate directly addresses the same fact, entity, attribute, event, quantity or comparison as the claim.
- 0.5: the candidate is about the same entity or topic, but not about the specific fact the claim states.
- 0.0: the candidate is about something else.

This is relevance, not verification:
- Do not decide whether the claim is true, false, supported, contradicted, partially supported or in conflict. Those judgements belong to a later stage, and no candidate may be dropped because of them.
- Contradicting evidence can be highly relevant. Against the claim "Browser Use has 100,000 GitHub stars", both "Browser Use has 111,799 GitHub stars" and "Browser Use has 30,000 GitHub stars" are highly relevant, because both state that repository's GitHub star count.
- Evidence that shares only a few words with the claim while stating a different fact is only weakly relevant. Against that same claim, "Browser Use is primarily written in Python" names the same project but not the star count, so its relevance is clearly lower.

Output rules:
- Return exactly one score for every candidate you are given, and no score for anything else.
- Copy each evidence_id verbatim from the prompt. Never invent, rename, shorten, drop or duplicate an id.
- Output only what the schema asks for: a "scores" list where each item has an evidence_id and a relevance_score.
- Do not explain, and do not show your reasoning.
"""


class SemanticEvidenceReranker:
	"""Rescore the Phase 4A candidates of each claim with one structured-output call per claim.

	One call per claim instead of one per pair keeps the cost proportional to the number of claims,
	and putting all candidates of a claim in a single prompt lets the model weigh them against each
	other rather than scoring each in a vacuum.

	Inputs are the products of the earlier phases: the ``ClaimSet``, its lexical ``AlignmentResult``,
	and the ``EvidenceNode`` list. The reranker reads no store, no screenshot and no metadata, never
	mutates its inputs, and always carries ``lexical_score`` through untouched so a later stage can
	still see what the deterministic baseline concluded.
	"""

	def __init__(
		self,
		llm: 'BaseChatModel',
		*,
		semantic_weight: float = 0.7,
		max_evidence_chars: int = 6000,
	) -> None:
		"""Args:
		llm: chat model used to judge candidate relevance.
		semantic_weight: share of the combined score given to the model; the rest goes to lexical.
		max_evidence_chars: prompt budget for one candidate, clipped deterministically.
		"""
		if not 0.0 <= semantic_weight <= 1.0:
			raise ValueError(f'semantic_weight must be between 0.0 and 1.0, got {semantic_weight}')
		if max_evidence_chars < 1:
			raise ValueError(f'max_evidence_chars must be at least 1, got {max_evidence_chars}')

		self.llm = llm
		self.semantic_weight = semantic_weight
		self.lexical_weight = 1.0 - semantic_weight
		self.max_evidence_chars = max_evidence_chars

	async def rerank(
		self,
		*,
		claim_set: ClaimSet,
		alignment_result: AlignmentResult,
		evidence_nodes: Sequence[EvidenceNode],
	) -> RerankingResult:
		"""Rerank the lexical candidates of every claim.

		Raises:
			EvidenceRerankingError: when the inputs disagree with each other, when the completion is
				not a RawSemanticReranking, or when the model call fails. Nothing is silently
				dropped, because a partial score set would quietly change what gets verified.
		"""
		if alignment_result.task_id != claim_set.task_id:
			raise EvidenceRerankingError(
				f'Task mismatch: alignment result is for {alignment_result.task_id!r}, claim set is for {claim_set.task_id!r}'
			)

		claims_by_id = self._index_claims(claim_set)
		alignments_by_id = self._index_alignments(alignment_result, claims_by_id)
		evidence_by_id = self._index_evidence(evidence_nodes)

		rerankings: list[ClaimReranking] = []
		for claim in claim_set.claims:
			alignment = alignments_by_id[claim.claim_id]
			if not alignment.matches:
				# Nothing to ask about, and no reason to spend a call on it.
				rerankings.append(ClaimReranking(claim_id=claim.claim_id, matches=[]))
				continue
			rerankings.append(await self._rerank_claim(claim, alignment, evidence_by_id))

		return RerankingResult(task_id=claim_set.task_id, rerankings=rerankings)

	@staticmethod
	def _index_claims(claim_set: ClaimSet) -> dict[str, Claim]:
		claims_by_id: dict[str, Claim] = {}
		for claim in claim_set.claims:
			if claim.claim_id in claims_by_id:
				raise EvidenceRerankingError(f'Claim set contains claim_id {claim.claim_id!r} more than once')
			claims_by_id[claim.claim_id] = claim
		return claims_by_id

	@staticmethod
	def _index_alignments(alignment_result: AlignmentResult, claims_by_id: dict[str, Claim]) -> dict[str, ClaimAlignment]:
		"""Require exactly one alignment per claim, so no claim quietly vanishes from verification."""
		alignments_by_id: dict[str, ClaimAlignment] = {}
		for alignment in alignment_result.alignments:
			if alignment.claim_id in alignments_by_id:
				raise EvidenceRerankingError(f'Alignment result contains claim_id {alignment.claim_id!r} more than once')
			if alignment.claim_id not in claims_by_id:
				raise EvidenceRerankingError(f'Alignment references unknown claim_id {alignment.claim_id!r}')
			alignments_by_id[alignment.claim_id] = alignment

		missing = [claim_id for claim_id in claims_by_id if claim_id not in alignments_by_id]
		if missing:
			raise EvidenceRerankingError(f'{len(missing)} claim(s) have no alignment entry, first one: {missing[0]!r}')
		return alignments_by_id

	@staticmethod
	def _index_evidence(evidence_nodes: Sequence[EvidenceNode]) -> dict[str, EvidenceNode]:
		evidence_by_id: dict[str, EvidenceNode] = {}
		for node in evidence_nodes:
			if node.evidence_id in evidence_by_id:
				raise EvidenceRerankingError(f'Evidence list contains evidence_id {node.evidence_id!r} more than once')
			evidence_by_id[node.evidence_id] = node
		return evidence_by_id

	async def _rerank_claim(
		self, claim: Claim, alignment: ClaimAlignment, evidence_by_id: dict[str, EvidenceNode]
	) -> ClaimReranking:
		"""Ask for the relevance of all candidates of one claim in a single call."""
		candidates: list[tuple[EvidenceMatch, EvidenceNode]] = []
		for match in alignment.matches:
			node = evidence_by_id.get(match.evidence_id)
			if node is None:
				raise EvidenceRerankingError(
					f'Alignment for claim {claim.claim_id!r} references unknown evidence_id {match.evidence_id!r}'
				)
			candidates.append((match, node))

		try:
			response = await self.llm.ainvoke(self._messages(claim, candidates), output_format=RawSemanticReranking)
		except Exception as e:
			# No str(e): provider errors often echo the request, which carries the claim and the evidence.
			raise EvidenceRerankingError(f'Semantic reranking failed for claim {claim.order}: {type(e).__name__}') from e

		completion = getattr(response, 'completion', None)
		if not isinstance(completion, RawSemanticReranking):
			raise EvidenceRerankingError(
				f'Semantic reranking for claim {claim.order} expected RawSemanticReranking, got {type(completion).__name__}'
			)

		semantic_scores = self._match_scores(completion.scores, [node.evidence_id for _, node in candidates], claim)
		matches = [
			RerankedEvidenceMatch(
				evidence_id=node.evidence_id,
				lexical_score=match.score,
				semantic_score=semantic_scores[node.evidence_id],
				combined_score=self._combine(match.score, semantic_scores[node.evidence_id]),
				rank=1,
			)
			for match, node in candidates
		]
		matches.sort(key=lambda match: (-match.combined_score, -match.semantic_score, -match.lexical_score, match.evidence_id))
		for rank, match in enumerate(matches, start=1):
			match.rank = rank
		return ClaimReranking(claim_id=claim.claim_id, matches=matches)

	@staticmethod
	def _match_scores(scores: Sequence[RawSemanticEvidenceScore], candidate_ids: Sequence[str], claim: Claim) -> dict[str, float]:
		"""Pin the model output to the candidate set: same ids, same count, each exactly once."""
		returned_ids = [score.evidence_id for score in scores]

		unknown_ids = sorted(set(returned_ids) - set(candidate_ids))
		if unknown_ids:
			raise EvidenceRerankingError(
				f'Semantic reranking for claim {claim.order} returned {len(unknown_ids)} unknown evidence_id(s), first one: {unknown_ids[0]!r}'
			)

		seen: set[str] = set()
		duplicate_ids = sorted({evidence_id for evidence_id in returned_ids if evidence_id in seen or seen.add(evidence_id)})
		if duplicate_ids:
			raise EvidenceRerankingError(
				f'Semantic reranking for claim {claim.order} returned {len(duplicate_ids)} duplicate evidence_id(s), first one: {duplicate_ids[0]!r}'
			)

		omitted_ids = sorted(set(candidate_ids) - set(returned_ids))
		if omitted_ids:
			raise EvidenceRerankingError(
				f'Semantic reranking for claim {claim.order} omitted {len(omitted_ids)} candidate(s), first one: {omitted_ids[0]!r}'
			)

		return {score.evidence_id: score.relevance_score for score in scores}

	def _combine(self, lexical_score: float, semantic_score: float) -> float:
		"""``combined = (1 - semantic_weight) * lexical_score + semantic_weight * semantic_score``, clamped."""
		combined = self.lexical_weight * lexical_score + self.semantic_weight * semantic_score
		return max(0.0, min(1.0, combined))

	def _messages(self, claim: Claim, candidates: Sequence[tuple[EvidenceMatch, EvidenceNode]]) -> list:
		from browser_use.llm.messages import SystemMessage, UserMessage

		return [SystemMessage(content=_SEMANTIC_RERANKING_SYSTEM_PROMPT), UserMessage(content=self._prompt(claim, candidates))]

	def _prompt(self, claim: Claim, candidates: Sequence[tuple[EvidenceMatch, EvidenceNode]]) -> str:
		"""One claim plus all of its candidates, with every id spelled out so the model can copy it."""
		lines = ['Claim:', claim.text, '', 'Candidates:']
		for index, (_, node) in enumerate(candidates, start=1):
			lines.append(f'[Candidate {index}]')
			lines.append(f'evidence_id: {node.evidence_id}')
			lines.append(f'title: {self._clip(node.title)}')
			lines.append('content:')
			lines.append(self._clip(node.text))
			lines.append('')
		lines.append('Return one relevance score per candidate, copying every evidence_id verbatim.')
		return '\n'.join(lines)

	def _clip(self, text: str) -> str:
		"""Clip text to the prompt budget on a fixed character boundary, without rewording."""
		if len(text) <= self.max_evidence_chars:
			return text
		return text[: self.max_evidence_chars].rstrip() + _TRUNCATION_MARKER


# Re-resolve the forward reference kept inside the quoted annotation above.
ClaimReranking.model_rebuild()
