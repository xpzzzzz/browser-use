"""Deterministic lexical candidate alignment between claims and collected evidence.

This stage answers exactly one question: which evidence is worth handing to the verifier for a
given claim. A score is candidate *relevance*, never *support* -- a high score only means the two
texts share words. Whether evidence actually supports or contradicts a claim belongs to Phase 5.

Score definition, kept deliberately simple so benchmark runs can be reproduced and explained:

    claim_tokens    = set(tokenize(claim.text))
    evidence_tokens = set(tokenize(node.text)) | set(tokenize(node.title))
    title_tokens    = set(tokenize(node.title))

    claim_coverage  = |claim_tokens & evidence_tokens| / |claim_tokens|
    jaccard         = |claim_tokens & evidence_tokens| / |claim_tokens | evidence_tokens|
    title_coverage  = |claim_tokens & title_tokens| / |claim_tokens|

    score = clamp(0.65 * claim_coverage + 0.20 * jaccard + 0.15 * title_coverage, 0.0, 1.0)

Every fraction is defined as 0.0 when its denominator is empty, so the score is always in
[0, 1] and an empty claim or empty evidence never scores above a real overlap.
"""

import re
import unicodedata
from collections.abc import Sequence
from typing import NamedTuple

from pydantic import BaseModel, Field

from browser_use.evidence.claims import Claim, ClaimSet
from browser_use.evidence.models import EvidenceNode

# Component weights; they sum to 1.0 so the score is bounded by construction.
CLAIM_COVERAGE_WEIGHT = 0.65
JACCARD_WEIGHT = 0.20
TITLE_COVERAGE_WEIGHT = 0.15

# Word characters in any script, so no language-specific segmenter is needed.
_TOKEN_PATTERN = re.compile(r'\w+', re.UNICODE)
# Runs of CJK characters, which are not space separated and therefore need bigram indexing.
_CJK_PATTERN = re.compile(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]')


class EvidenceMatch(BaseModel):
	"""One evidence candidate for one claim, with its relevance score and rank."""

	evidence_id: str = Field(description='EvidenceNode.evidence_id of the candidate')
	score: float = Field(ge=0.0, le=1.0, description='Lexical relevance in [0, 1], see the module docstring')
	rank: int = Field(ge=1, description='1-based rank within the candidate list of this claim')


class ClaimAlignment(BaseModel):
	"""The candidate evidence for one claim, in descending relevance order.

	An empty ``matches`` list means "this claim exists but nothing in the evidence pool is even
	lexically related to it", which is a different situation from the claim not existing at all.
	"""

	claim_id: str = Field(description='Claim.claim_id this alignment belongs to, copied unchanged')
	matches: list[EvidenceMatch] = Field(default_factory=list, description='Candidates ordered by rank')


class AlignmentResult(BaseModel):
	"""Alignment outcome for a whole ClaimSet."""

	task_id: str = Field(description='Task the claims and evidence belong to')
	alignments: list[ClaimAlignment] = Field(default_factory=list, description='One entry per claim, in claim order')


def tokenize(text: str) -> list[str]:
	"""Split text into comparable tokens: NFKC-normalised, casefolded, punctuation dropped.

	Non-ASCII text is preserved. CJK runs additionally contribute their character bigrams, which
	is enough to recall Chinese evidence without pulling in a segmenter.
	"""
	if not text:
		return []

	tokens: list[str] = []
	for match in _TOKEN_PATTERN.findall(unicodedata.normalize('NFKC', text).casefold()):
		cjk_characters = _CJK_PATTERN.findall(match)
		if not cjk_characters:
			tokens.append(match)
			continue
		tokens.extend(cjk_characters)
		tokens.extend(first + second for first, second in zip(cjk_characters, cjk_characters[1:], strict=False))
	return tokens


def _coverage(tokens: frozenset[str], haystack: frozenset[str]) -> float:
	"""Share of ``tokens`` also found in ``haystack``, or 0.0 when there is nothing to cover."""
	if not tokens or not haystack:
		return 0.0
	return len(tokens & haystack) / len(tokens)


def _jaccard(tokens: frozenset[str], haystack: frozenset[str]) -> float:
	if not tokens or not haystack:
		return 0.0
	return len(tokens & haystack) / len(tokens | haystack)


class _EvidenceIndexEntry(NamedTuple):
	"""An evidence node plus the token sets derived from it once per ``align`` call."""

	node: EvidenceNode
	title_tokens: frozenset[str]
	content_tokens: frozenset[str]


class EvidenceAligner:
	"""Rank collected evidence per claim with a deterministic lexical baseline.

	The aligner is pure: it takes ``ClaimSet`` and ``EvidenceNode`` objects, reads no store, no
	screenshot, no metadata, no clock, and no LLM, and it never mutates its inputs. Callers decide
	where evidence came from, usually ``JsonlEvidenceStore.load_all()``.

	Screenshot, ``metadata`` and ``action_names`` are deliberately excluded from scoring: this
	baseline compares words on the page with words in the claim, and multimodal signals are a
	future extension.
	"""

	def __init__(self, top_k: int = 5) -> None:
		"""Args:
		top_k: maximum number of candidates kept per claim.
		"""
		if top_k < 1:
			raise ValueError(f'top_k must be at least 1, got {top_k}')
		self.top_k = top_k

	def align(self, *, claim_set: ClaimSet, evidence_nodes: Sequence[EvidenceNode]) -> AlignmentResult:
		"""Pick the Top-K lexical candidates for every claim.

		Empty claims produce an empty ``alignments`` list; with no evidence each claim still gets
		its own alignment with an empty ``matches`` list, so downstream phases can tell "no claims"
		apart from "claims with nothing to check".
		"""
		entries = self._build_index(evidence_nodes)
		return AlignmentResult(
			task_id=claim_set.task_id,
			alignments=[self._align_claim(claim, entries) for claim in claim_set.claims],
		)

	@classmethod
	def _build_index(cls, evidence_nodes: Sequence[EvidenceNode]) -> list[_EvidenceIndexEntry]:
		"""Tokenize every evidence node once, and reject duplicate ids before any scoring."""
		entries: list[_EvidenceIndexEntry] = []
		seen_ids: dict[str, int] = {}
		for position, node in enumerate(evidence_nodes):
			if node.evidence_id in seen_ids:
				raise ValueError(
					f'Duplicate evidence_id {node.evidence_id!r} at positions {seen_ids[node.evidence_id]} and {position}. '
					'evidence_id is the key the verifier uses to cite evidence, so it must be unique.'
				)
			seen_ids[node.evidence_id] = position
			title_tokens = frozenset(tokenize(node.title))
			content_tokens = frozenset(tokenize(node.text)) | title_tokens
			entries.append(_EvidenceIndexEntry(node=node, title_tokens=title_tokens, content_tokens=content_tokens))
		return entries

	def _align_claim(self, claim: Claim, entries: Sequence[_EvidenceIndexEntry]) -> ClaimAlignment:
		"""Score, filter, sort and truncate the candidates of one claim, then assign ranks."""
		claim_tokens = frozenset(tokenize(claim.text))
		scored = [(self._score(claim_tokens, entry), entry) for entry in entries]

		# Score 0 means no shared word at all, so it never becomes a candidate: no padding the Top-K
		# with pages that simply mention the same product name.
		relevant = [item for item in scored if item[0] > 0.0]
		relevant.sort(key=lambda item: (-item[0], item[1].node.step_number, item[1].node.evidence_id))

		matches = [
			EvidenceMatch(evidence_id=entry.node.evidence_id, score=score, rank=rank)
			for rank, (score, entry) in enumerate(relevant[: self.top_k], start=1)
		]
		return ClaimAlignment(claim_id=claim.claim_id, matches=matches)

	@staticmethod
	def _score(claim_tokens: frozenset[str], entry: _EvidenceIndexEntry) -> float:
		"""Weighted relevance of one evidence node for one claim; see the module docstring."""
		score = (
			CLAIM_COVERAGE_WEIGHT * _coverage(claim_tokens, entry.content_tokens)
			+ JACCARD_WEIGHT * _jaccard(claim_tokens, entry.content_tokens)
			+ TITLE_COVERAGE_WEIGHT * _coverage(claim_tokens, entry.title_tokens)
		)
		return max(0.0, min(1.0, score))
