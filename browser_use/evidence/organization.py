"""Deterministic structured organization of verified evidence into a graph.

Phases 3 to 5 answered what the answer claims, which evidence was worth reading, and how each piece
of evidence relates to each claim. This phase stops asking questions and only arranges what is
already known: it turns a ``ClaimSet``, a ``VerificationResult`` and the referenced ``EvidenceNode``
objects into one serialisable ``EvidenceGraph`` with claim and evidence nodes, the claim-to-evidence
relations Phase 5 produced, and three derived evidence-to-evidence relations -- shared source,
near-duplicate content, and explicit conflict inside one claim.

Two boundaries define the stage. It performs no verification of its own: no LLM, no network, no
re-judged relation, no re-derived status, because Phase 5 already decided what the evidence means
and re-asking would let this stage quietly disagree with it. And it performs no arbitration either:
a conflict edge records that two pieces of evidence cannot both support one claim's fact, not which
one is right, since source authority, recency and dedup policy belong to reporting.

Everything is pure Python over the given objects, so the graph is byte-for-byte reproducible: same
inputs in any order, same output. That is what makes the later report and benchmark stages able to
point at one structure and attribute a difference to a real change rather than to a reshuffle.
"""

import re
import unicodedata
from collections.abc import Sequence
from enum import Enum
from typing import NamedTuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

from browser_use.evidence.alignment import tokenize
from browser_use.evidence.claims import Claim, ClaimSet, NonBlankString
from browser_use.evidence.models import EvidenceNode
from browser_use.evidence.verification import (
	ClaimVerification,
	EvidenceAssessment,
	EvidenceRelation,
	VerificationResult,
	VerificationStatus,
)

# Relations that back a claim, and the one that denies it. Conflict is exactly the join of the two.
_SUPPORTING_RELATIONS = frozenset({EvidenceRelation.SUPPORTS, EvidenceRelation.PARTIAL_SUPPORT})
_CONTRADICTING_RELATIONS = frozenset({EvidenceRelation.CONTRADICTS})

_WHITESPACE_PATTERN = re.compile(r'\s+')


class EvidenceOrganizationError(RuntimeError):
	"""Raised when the graph cannot be built from the given stages.

	The graph is the artefact a report is written from, so a node or edge that points at an id which
	does not exist, or a claim whose verification is missing, would produce a citation trail that
	leads nowhere. Those are errors rather than something to repair, and nothing is dropped quietly.
	Error messages carry identifiers only, never evidence text or prompts.
	"""


class EvidenceEdgeType(str, Enum):
	"""How two pieces of evidence relate to each other, independent of any single claim.

	* ``SAME_SOURCE``: both were captured from the same host, which is how a report can tell that two
	  apparently agreeing citations are really one source speaking twice.
	* ``DUPLICATE``: near-identical content, possibly across different hosts, which is the same
	  redundancy problem seen from the other side.
	* ``CONFLICTS_WITH``: inside one specific claim, one supports and the other contradicts. This is
	  the only relation that is claim-relative, so it is the only one carrying a ``claim_id``.
	"""

	SAME_SOURCE = 'SAME_SOURCE'
	DUPLICATE = 'DUPLICATE'
	CONFLICTS_WITH = 'CONFLICTS_WITH'


class ClaimGraphNode(BaseModel):
	"""A claim as it appears in the graph, with the verdict Phase 5 already reached."""

	claim_id: str = Field(description='Claim.claim_id, copied unchanged')
	text: str = Field(description='Claim text, copied unchanged; ClaimSet stays the source of truth')
	order: int = Field(ge=1, description='Claim.order, which fixes the node order in the graph')
	verification_status: VerificationStatus = Field(description='Phase 5 status, never recomputed here')


class EvidenceGraphNode(BaseModel):
	"""Evidence that took part in verification, reduced to what a graph needs to identify it.

	The page text is deliberately not copied: ``EvidenceNode`` is the source of truth for content, a
	graph of full DOM text would dwarf the structure it is meant to describe, and a report can always
	resolve an ``evidence_id`` back to the original node. Screenshot paths, metadata and action names
	are left out for the same reason.
	"""

	evidence_id: str = Field(description='EvidenceNode.evidence_id, the key back to the real node')
	url: str = Field(description='Captured url, copied unchanged')
	title: str = Field(description='Page title at capture time, copied unchanged')
	step_number: int = Field(ge=1, description='1-based agent step that produced this evidence')
	source_host: str = Field(default='', description='Normalized host, see EvidenceOrganizer; empty if unknown')


class ClaimEvidenceEdge(BaseModel):
	"""One Phase 5 assessment, lifted into an edge without changing its meaning."""

	claim_id: str = Field(description='Claim end of the edge')
	evidence_id: str = Field(description='Evidence end of the edge')
	relation: EvidenceRelation = Field(description='EvidenceRelation as judged by Phase 5')
	explanation: NonBlankString = Field(description='The model rationale for this relation, copied verbatim')


class EvidenceEvidenceEdge(BaseModel):
	"""A derived relation between two evidence nodes of the same graph."""

	source_evidence_id: str = Field(description='Lower id of the canonical pair, so an edge is never stored twice')
	target_evidence_id: str = Field(description='Higher id of the canonical pair')
	relation: EvidenceEdgeType = Field(description='SAME_SOURCE, DUPLICATE or CONFLICTS_WITH')
	claim_id: str | None = Field(
		default=None,
		description='Claim the conflict was found under; only CONFLICTS_WITH has one, and it always does',
	)

	@model_validator(mode='after')
	def _check_claim_scope(self) -> 'EvidenceEvidenceEdge':
		"""Keep claim-relative and claim-independent relations from being mixed up.

		A ``SAME_SOURCE`` or ``DUPLICATE`` edge holds for the pair whatever is being asked about, so a
		claim id on it would suggest a scoping that does not exist. A conflict only ever exists
		*inside* a claim, so an unscoped one would be meaningless.
		"""
		if self.relation is EvidenceEdgeType.CONFLICTS_WITH:
			if not self.claim_id:
				raise ValueError(
					f'CONFLICTS_WITH between {self.source_evidence_id!r} and {self.target_evidence_id!r} needs the claim_id it was found under'
				)
		elif self.claim_id is not None:
			raise ValueError(
				f'{self.relation.value} between {self.source_evidence_id!r} and {self.target_evidence_id!r} must not carry a claim_id'
			)
		return self


class EvidenceGraphStats(BaseModel):
	"""Counts over a graph, always derived from the graph itself.

	These are the numbers a report leads with, so they are computed from the nodes and edges that are
	actually present rather than accumulated while building them: a count that disagrees with the
	lists beside it is worse than no count. No confidence, score or probability is kept here.
	"""

	claim_count: int = Field(default=0, ge=0, description='Claim nodes in the graph')
	evidence_count: int = Field(default=0, ge=0, description='Evidence nodes that took part in verification')
	supported_claim_count: int = Field(default=0, ge=0, description='Claims with status SUPPORTED')
	partial_claim_count: int = Field(default=0, ge=0, description='Claims with status PARTIAL')
	unsupported_claim_count: int = Field(default=0, ge=0, description='Claims with status UNSUPPORTED')
	contradicted_claim_count: int = Field(default=0, ge=0, description='Claims with status CONTRADICTED')
	conflicted_claim_count: int = Field(default=0, ge=0, description='Claims with status CONFLICTED')
	no_evidence_claim_count: int = Field(default=0, ge=0, description='Claims with status NO_EVIDENCE')
	support_edge_count: int = Field(default=0, ge=0, description='Claim-evidence edges with relation SUPPORTS')
	partial_support_edge_count: int = Field(default=0, ge=0, description='Claim-evidence edges with PARTIAL_SUPPORT')
	contradict_edge_count: int = Field(default=0, ge=0, description='Claim-evidence edges with CONTRADICTS')
	insufficient_edge_count: int = Field(default=0, ge=0, description='Claim-evidence edges with INSUFFICIENT')
	same_source_edge_count: int = Field(default=0, ge=0, description='Evidence pairs sharing a source host')
	duplicate_edge_count: int = Field(default=0, ge=0, description='Evidence pairs judged near-duplicates')
	conflict_edge_count: int = Field(default=0, ge=0, description='Claim-scoped support versus contradiction pairs')

	def as_dict(self) -> dict[str, int]:
		"""The stats as a plain mapping, which is how the graph recomputes and compares them."""
		return {name: getattr(self, name) for name in type(self).model_fields}


class EvidenceGraph(BaseModel):
	"""The verified evidence of one task as one deterministic, serialisable structure."""

	task_id: str = Field(description='Agent task the claims, verdicts and evidence belong to')
	claims: list[ClaimGraphNode] = Field(default_factory=list, description='Claim nodes, ordered by Claim.order')
	evidence: list[EvidenceGraphNode] = Field(
		default_factory=list, description='Evidence nodes referenced by at least one assessment, ordered by step then id'
	)
	claim_evidence_edges: list[ClaimEvidenceEdge] = Field(
		default_factory=list, description='Phase 5 assessments, in claim order and then Phase 5 assessment order'
	)
	evidence_evidence_edges: list[EvidenceEvidenceEdge] = Field(
		default_factory=list, description='Derived source, duplicate and conflict edges, in a fixed sorted order'
	)
	stats: EvidenceGraphStats = Field(
		default_factory=EvidenceGraphStats, description='Derived counts; never supplied by a caller'
	)

	@model_validator(mode='after')
	def _recompute_stats(self) -> 'EvidenceGraph':
		"""Derive ``stats`` from the content, so a stale or hand-written count cannot survive validation."""
		self.stats = _graph_stats(self.claims, self.evidence, self.claim_evidence_edges, self.evidence_evidence_edges)
		return self


def _graph_stats(
	claims: Sequence[ClaimGraphNode],
	evidence: Sequence[EvidenceGraphNode],
	claim_evidence_edges: Sequence[ClaimEvidenceEdge],
	evidence_evidence_edges: Sequence[EvidenceEvidenceEdge],
) -> EvidenceGraphStats:
	"""Count what is in the graph. ``evidence`` is counted as nodes, not as edge endpoints."""
	status_counts = {status: 0 for status in VerificationStatus}
	for node in claims:
		status_counts[node.verification_status] += 1

	relation_counts = {relation: 0 for relation in EvidenceRelation}
	for edge in claim_evidence_edges:
		relation_counts[edge.relation] += 1

	edge_type_counts = {edge_type: 0 for edge_type in EvidenceEdgeType}
	for edge in evidence_evidence_edges:
		edge_type_counts[edge.relation] += 1

	return EvidenceGraphStats(
		claim_count=len(claims),
		evidence_count=len(evidence),
		supported_claim_count=status_counts[VerificationStatus.SUPPORTED],
		partial_claim_count=status_counts[VerificationStatus.PARTIAL],
		unsupported_claim_count=status_counts[VerificationStatus.UNSUPPORTED],
		contradicted_claim_count=status_counts[VerificationStatus.CONTRADICTED],
		conflicted_claim_count=status_counts[VerificationStatus.CONFLICTED],
		no_evidence_claim_count=status_counts[VerificationStatus.NO_EVIDENCE],
		support_edge_count=relation_counts[EvidenceRelation.SUPPORTS],
		partial_support_edge_count=relation_counts[EvidenceRelation.PARTIAL_SUPPORT],
		contradict_edge_count=relation_counts[EvidenceRelation.CONTRADICTS],
		insufficient_edge_count=relation_counts[EvidenceRelation.INSUFFICIENT],
		same_source_edge_count=edge_type_counts[EvidenceEdgeType.SAME_SOURCE],
		duplicate_edge_count=edge_type_counts[EvidenceEdgeType.DUPLICATE],
		conflict_edge_count=edge_type_counts[EvidenceEdgeType.CONFLICTS_WITH],
	)


def _normalize_whitespace(text: str) -> str:
	"""Collapse every run of whitespace to one space, so layout differences stop mattering."""
	return _WHITESPACE_PATTERN.sub(' ', text).strip()


def _normalized_content(title: str, text: str) -> str:
	"""The comparable form of one evidence node: NFKC, casefolded, whitespace-collapsed.

	Title and text are normalized separately and then joined, so a page whose title happens to repeat
	its first line cannot look identical to a page that merely stored that line in the title.
	"""
	title = _normalize_whitespace(unicodedata.normalize('NFKC', title).casefold())
	text = _normalize_whitespace(unicodedata.normalize('NFKC', text).casefold())
	return f'{title}\n{text}'


def _source_host(url: str) -> str:
	"""Host of a url for grouping purposes: lowercased, no trailing dot, no leading ``www.``.

	Returns an empty string whenever a host cannot be read, because "unknown" must not be allowed to
	group with another "unknown" and look like a shared source. Only the ``www.`` label is removed;
	``docs.`` or ``en.`` distinguish real sites, so they stay.
	"""
	try:
		host = urlparse(url).hostname
	except ValueError:
		return ''

	host = (host or '').strip().rstrip('.').lower()
	return host[4:] if host.startswith('www.') else host


class _EvidenceProfile(NamedTuple):
	"""One used evidence node with the derived fields the edge rules need, computed once each."""

	node: EvidenceNode
	source_host: str
	content: str
	tokens: frozenset[str]


class EvidenceOrganizer:
	"""Build the ``EvidenceGraph`` for one task from the outputs of the earlier phases.

	The organizer is pure and synchronous: no model, no network, no store, and no mutation of its
	inputs. Its constructor takes only the two knobs of the duplicate rule, because those are the
	one genuinely debutable decision in this stage and a benchmark run needs to state them explicitly.

	Duplicate detection is deliberately deterministic rather than semantic:

	1. If the normalized ``title`` plus ``text`` of two nodes are equal and non-empty, they are
	   duplicates. Exact equality after NFKC, casefolding and whitespace collapsing is a strong
	   enough signal that no token floor is needed.
	2. Otherwise the two token sets of that same normalized content, tokenized with the Phase 4A
	   tokenizer so both stages mean the same thing by a token, must both hold at least
	   ``min_duplicate_tokens`` tokens and have a Jaccard similarity of at least
	   ``duplicate_threshold``. The token floor is what stops two one-word pages from matching on
	   the single word they share, and the high default threshold means "near-identical", not
	   "about the same topic" -- that judgement was Phase 4B's, and it stays there.

	Duplicates are recorded as an edge and never as a deletion: which copy a report shows is a
	presentation choice, and the two nodes can carry different assessments for different claims.

	Ordering is fixed regardless of how the inputs are arranged. Claims follow ``Claim.order``, then
	``claim_id``. Evidence follows ``step_number``, then ``evidence_id``. Claim-to-evidence edges walk
	the claims in that order and keep the Phase 5 assessment order within each claim, which was
	itself the candidate rank order. Evidence-to-evidence edges are sorted by ``(relation, claim_id or
	'', source_evidence_id, target_evidence_id)``, and every pair is stored once in canonical id order,
	so shuffling ``evidence_nodes`` cannot change ``graph.model_dump()``.
	"""

	def __init__(self, duplicate_threshold: float = 0.90, min_duplicate_tokens: int = 5) -> None:
		"""Args:
		duplicate_threshold: minimum token Jaccard similarity for the near-duplicate rule.
		min_duplicate_tokens: both nodes must have at least this many tokens to be compared by similarity.
		"""
		if not 0.0 < duplicate_threshold <= 1.0:
			raise ValueError(f'duplicate_threshold must be greater than 0.0 and at most 1.0, got {duplicate_threshold}')
		if min_duplicate_tokens < 1:
			raise ValueError(f'min_duplicate_tokens must be at least 1, got {min_duplicate_tokens}')

		self.duplicate_threshold = duplicate_threshold
		self.min_duplicate_tokens = min_duplicate_tokens

	def organize(
		self,
		*,
		claim_set: ClaimSet,
		verification_result: VerificationResult,
		evidence_nodes: Sequence[EvidenceNode],
	) -> EvidenceGraph:
		"""Organize one verified task into an ``EvidenceGraph``.

		Raises:
			EvidenceOrganizationError: when the inputs disagree with each other, so that a node or an
				edge could not be traced back to a real claim or a real piece of evidence.
		"""
		if verification_result.task_id != claim_set.task_id:
			raise EvidenceOrganizationError(
				f'Task mismatch: verification result is for {verification_result.task_id!r}, claim set is for {claim_set.task_id!r}'
			)

		claims_by_id = self._index_claims(claim_set)
		verifications_by_id = self._index_verifications(verification_result, claims_by_id)
		nodes_by_id = self._index_evidence(evidence_nodes)

		claim_nodes = [
			ClaimGraphNode(
				claim_id=claim.claim_id,
				text=claim.text,
				order=claim.order,
				verification_status=verifications_by_id[claim.claim_id].status,
			)
			for claim in sorted(claim_set.claims, key=lambda claim: (claim.order, claim.claim_id))
		]

		claim_evidence_edges: list[ClaimEvidenceEdge] = []
		used_ids: set[str] = set()
		for node in claim_nodes:
			# Phase 5 already ordered the assessments by candidate rank, and that order is the record.
			for assessment in self._assessments(verifications_by_id[node.claim_id], node):
				if assessment.evidence_id not in nodes_by_id:
					raise EvidenceOrganizationError(
						f'Verification for claim {node.claim_id!r} references unknown evidence_id {assessment.evidence_id!r}'
					)
				used_ids.add(assessment.evidence_id)
				claim_evidence_edges.append(
					ClaimEvidenceEdge(
						claim_id=node.claim_id,
						evidence_id=assessment.evidence_id,
						relation=assessment.relation,
						explanation=assessment.explanation,
					)
				)

		profiles = {
			evidence_id: self._profile(nodes_by_id[evidence_id])
			for evidence_id in sorted(used_ids, key=lambda evidence_id: (nodes_by_id[evidence_id].step_number, evidence_id))
		}
		evidence_nodes_graph = [
			EvidenceGraphNode(
				evidence_id=evidence_id,
				url=profile.node.url,
				title=profile.node.title,
				step_number=profile.node.step_number,
				source_host=profile.source_host,
			)
			for evidence_id, profile in profiles.items()
		]

		return EvidenceGraph(
			task_id=claim_set.task_id,
			claims=claim_nodes,
			evidence=evidence_nodes_graph,
			claim_evidence_edges=claim_evidence_edges,
			evidence_evidence_edges=self._evidence_edges(profiles, claim_evidence_edges),
		)

	@staticmethod
	def _index_claims(claim_set: ClaimSet) -> dict[str, Claim]:
		claims_by_id: dict[str, Claim] = {}
		for claim in claim_set.claims:
			if claim.claim_id in claims_by_id:
				raise EvidenceOrganizationError(f'Claim set contains claim_id {claim.claim_id!r} more than once')
			claims_by_id[claim.claim_id] = claim
		return claims_by_id

	@staticmethod
	def _index_verifications(
		verification_result: VerificationResult, claims_by_id: dict[str, Claim]
	) -> dict[str, ClaimVerification]:
		"""Require exactly one verification per claim, so no claim is quietly left out of the graph."""
		verifications_by_id: dict[str, ClaimVerification] = {}
		for verification in verification_result.verifications:
			if verification.claim_id in verifications_by_id:
				raise EvidenceOrganizationError(f'Verification result contains claim_id {verification.claim_id!r} more than once')
			if verification.claim_id not in claims_by_id:
				raise EvidenceOrganizationError(f'Verification references unknown claim_id {verification.claim_id!r}')
			verifications_by_id[verification.claim_id] = verification

		missing = [claim_id for claim_id in claims_by_id if claim_id not in verifications_by_id]
		if missing:
			raise EvidenceOrganizationError(f'{len(missing)} claim(s) have no verification, first one: {missing[0]!r}')
		return verifications_by_id

	@staticmethod
	def _index_evidence(evidence_nodes: Sequence[EvidenceNode]) -> dict[str, EvidenceNode]:
		nodes_by_id: dict[str, EvidenceNode] = {}
		for node in evidence_nodes:
			if node.evidence_id in nodes_by_id:
				raise EvidenceOrganizationError(f'Evidence list contains evidence_id {node.evidence_id!r} more than once')
			nodes_by_id[node.evidence_id] = node
		return nodes_by_id

	@staticmethod
	def _assessments(verification: ClaimVerification, node: ClaimGraphNode) -> Sequence[EvidenceAssessment]:
		"""Return one claim's assessments, rejecting a pair of verdicts about the same evidence."""
		seen: set[str] = set()
		for assessment in verification.assessments:
			if assessment.evidence_id in seen:
				raise EvidenceOrganizationError(
					f'Verification for claim {node.claim_id!r} assesses evidence_id {assessment.evidence_id!r} more than once'
				)
			seen.add(assessment.evidence_id)
		return verification.assessments

	@staticmethod
	def _profile(node: EvidenceNode) -> _EvidenceProfile:
		content = _normalized_content(node.title, node.text)
		return _EvidenceProfile(
			node=node,
			source_host=_source_host(node.url),
			content=content,
			tokens=frozenset(tokenize(content)),
		)

	def _evidence_edges(
		self, profiles: dict[str, _EvidenceProfile], claim_evidence_edges: Sequence[ClaimEvidenceEdge]
	) -> list[EvidenceEvidenceEdge]:
		"""Derive every evidence-to-evidence edge, one per canonical pair, then sort deterministically."""
		ordered_ids = sorted(profiles)
		edges: dict[tuple[str, str | None, str, str], EvidenceEvidenceEdge] = {}

		for position, source_id in enumerate(ordered_ids):
			for target_id in ordered_ids[position + 1 :]:
				source, target = profiles[source_id], profiles[target_id]
				if source.source_host and source.source_host == target.source_host:
					self._add_edge(edges, EvidenceEdgeType.SAME_SOURCE, source_id, target_id, claim_id=None)
				if self._is_duplicate(source, target):
					self._add_edge(edges, EvidenceEdgeType.DUPLICATE, source_id, target_id, claim_id=None)

		for source_id, target_id, claim_id in self._conflict_pairs(claim_evidence_edges):
			self._add_edge(edges, EvidenceEdgeType.CONFLICTS_WITH, source_id, target_id, claim_id=claim_id)

		return sorted(
			edges.values(),
			key=lambda edge: (edge.relation.value, edge.claim_id or '', edge.source_evidence_id, edge.target_evidence_id),
		)

	def _is_duplicate(self, source: _EvidenceProfile, target: _EvidenceProfile) -> bool:
		"""Deterministic near-duplicate test; see ``EvidenceOrganizer`` for the two rules."""
		if source.content.strip() and source.content == target.content:
			return True

		if len(source.tokens) < self.min_duplicate_tokens or len(target.tokens) < self.min_duplicate_tokens:
			return False

		union = source.tokens | target.tokens
		return len(source.tokens & target.tokens) / len(union) >= self.duplicate_threshold

	@staticmethod
	def _conflict_pairs(claim_evidence_edges: Sequence[ClaimEvidenceEdge]) -> list[tuple[str, str, str]]:
		"""Pair every supporting candidate of a claim with every contradicting candidate of that same claim.

		The claim id travels with the pair because that is the only thing making it a conflict: the
		two pages may be about different facts entirely, and outside this claim they may both be
		right. Pairs are symmetric, so a supporting-versus-contradicting direction carries no meaning
		and the canonical id order decides the endpoints instead.
		"""
		by_claim: dict[str, tuple[list[str], list[str]]] = {}
		for edge in claim_evidence_edges:
			supporting, contradicting = by_claim.setdefault(edge.claim_id, ([], []))
			if edge.relation in _SUPPORTING_RELATIONS:
				supporting.append(edge.evidence_id)
			elif edge.relation in _CONTRADICTING_RELATIONS:
				contradicting.append(edge.evidence_id)

		return [
			(source_id, target_id, claim_id)
			for claim_id, (supporting, contradicting) in by_claim.items()
			for source_id in supporting
			for target_id in contradicting
			if source_id != target_id
		]

	@staticmethod
	def _add_edge(
		edges: dict[tuple[str, str | None, str, str], EvidenceEvidenceEdge],
		relation: EvidenceEdgeType,
		first_id: str,
		second_id: str,
		*,
		claim_id: str | None,
	) -> None:
		source_id, target_id = (first_id, second_id) if first_id <= second_id else (second_id, first_id)
		edges.setdefault(
			(relation.value, claim_id, source_id, target_id),
			EvidenceEvidenceEdge(
				source_evidence_id=source_id,
				target_evidence_id=target_id,
				relation=relation,
				claim_id=claim_id,
			),
		)
