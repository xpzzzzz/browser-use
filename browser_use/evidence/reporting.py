"""Evidence-grounded user report built from an ``EvidenceGraph``, with no model in the loop.

Phases 3 to 6 did the thinking: they extracted the claims, retrieved candidate pages, verified each
claim against page content and organized the result into a graph. This stage only presents what those
stages concluded, so the report is a projection of the graph rather than a second opinion. Same
structured input, same report model, same Markdown, byte for byte.

Keeping the final user-visible layer model-free is the point. A model asked to summarize the
verification would reintroduce exactly the randomness, phrasing drift and hallucination risk the
pipeline exists to remove, and it would be free to change a conclusion it did not reach. Every
sentence here is either a Phase 5 assessment copied verbatim, the name of a graph structure, or fixed
text defined in this module.

Two rules shape the output. A claim is never dropped and never softened: ``NO_EVIDENCE``,
``UNSUPPORTED``, ``PARTIAL`` and ``CONTRADICTED`` each stay visibly distinct, because flattening them
into one pass or fail would destroy the information a reader came for. And every piece of untrusted
text -- task prompt, claim text, page title, assessment rationale, url -- is escaped before it reaches
the document, so a page titled ``<script>`` cannot make the report executable or rearrange its headings.
"""

import re
from collections.abc import Sequence

from pydantic import BaseModel, Field, ValidationError, model_validator

from browser_use.evidence.claims import Claim, ClaimSet, NonBlankString
from browser_use.evidence.organization import (
	ClaimGraphNode,
	EvidenceEdgeType,
	EvidenceGraph,
	EvidenceGraphNode,
)
from browser_use.evidence.verification import EvidenceRelation, VerificationStatus

# Citation labels are Python's own, one-based and in graph order, never a truncation of a uuid.
_CITATION_PREFIX = 'E'

# A run of whitespace becomes one space, which is what stops CR or LF from opening a new block.
_WHITESPACE_PATTERN = re.compile(r'\s+')
# Markup that would otherwise be interpreted inside untrusted text. One pass, so no escape is escaped
# twice: backslash, code spans, emphasis, link brackets, angle brackets for raw HTML.
_INLINE_MARKUP_PATTERN = re.compile(r'([\\`*_\[\]()<>])')
# Urls keep their parentheses, colon, hash and underscore so they stay readable and copyable. Brackets
# are still escaped because ``[text](target)`` is exactly how a markdown link gets built.
_URL_MARKUP_PATTERN = re.compile(r'([\\`*\[\]<>])')
# Block openers, which only matter when untrusted text starts a line. An ordered-list digit is left
# alone: at worst it renders a bullet, and escaping it would print a stray backslash.
_LINE_START_PATTERN = re.compile(r'^([-+>#])')

# Fixed document text, defined as constants so the report's own prose cannot drift between runs.
_REPORT_TITLE = '# WebEvidence Verification Report'
_NO_TASK_TEXT = 'No task prompt was recorded.'
_NO_CLAIMS_TEXT = 'The answer produced no atomic claims to verify.'
_NO_EVIDENCE_TEXT = 'No evidence candidates were available for verification.'
_NOT_CITED_TEXT = 'Not cited by any claim.'

# Relation groups in display order. A claim section shows only the groups it actually has.
_RELATION_GROUPS: tuple[tuple[EvidenceRelation, str], ...] = (
	(EvidenceRelation.SUPPORTS, 'Supporting evidence'),
	(EvidenceRelation.PARTIAL_SUPPORT, 'Partially supporting evidence'),
	(EvidenceRelation.CONTRADICTS, 'Contradicting evidence'),
	(EvidenceRelation.INSUFFICIENT, 'Evidence that does not speak to the claim'),
)

_INTERPRETATION_LINES = (
	'This report presents only the evidence that was available to this run.',
	'',
	'- SUPPORTED means the evidence available in this run supports the claim.',
	'- PARTIAL means some evidence supports part of the claim while a condition of it stays unproven.',
	'- UNSUPPORTED does not mean false. It means the evidence that was found neither supports nor contradicts the claim.',
	'- NO_EVIDENCE means no candidate evidence was available for verification, so this claim was never checked.',
	'- CONTRADICTED means the evidence provided directly conflicts with the claim.',
	'- CONFLICTED means the evidence set contains both supporting and contradicting evidence, and this report does not decide which source is right.',
	'',
	'There is no overall pass or fail verdict on purpose: one answer can mix supported, contradicted and unverified claims, and a single word would hide that.',
)


class EvidenceReportError(RuntimeError):
	"""Raised when a report cannot be built from the given graph.

	The report is the part a reader trusts, so a graph that disagrees with the claims it was built for
	-- a stale graph from an earlier run, an edge pointing at evidence the graph does not contain, stats
	that no longer match the content -- has to stop the build instead of being quietly repaired. A
	number fixed up at reporting time is a number that traces back to no verification at all. Messages
	carry identifiers and counts only, never claim or page text.
	"""


def _single_line(text: str) -> str:
	"""Collapse every whitespace run, including CR and LF, into a single space."""
	return _WHITESPACE_PATTERN.sub(' ', text or '').strip()


def escape_report_text(value: str) -> str:
	"""Make untrusted text inert within one markdown line.

	Newlines go first, since a claim or title containing them could otherwise open a fresh block.
	Markup characters are escaped in a single pass, so an escape is never escaped twice, and a leading
	block opener is escaped as well, which is what lets claim text sit at the start of a line without
	becoming a heading, a quote or a list item.
	"""
	text = _INLINE_MARKUP_PATTERN.sub(r'\\\1', _single_line(value))
	return _LINE_START_PATTERN.sub(r'\\\1', text)


def escape_report_url(value: str) -> str:
	"""Escape a url for plain-text display while keeping it readable and copyable.

	Brackets are the real risk, because ``[text](target)`` is exactly how a markdown link is built and
	a scraped url could supply all of it; backticks and angle brackets would open code spans or raw
	HTML. Parentheses, ``#``, ``&`` and ``:`` carry no such power on their own, so they stay unescaped
	and the url remains the string the reader can copy. The renderer never wraps a url in a link at all,
	so a ``javascript:`` scheme remains a piece of text rather than an action.
	"""
	return _URL_MARKUP_PATTERN.sub(r'\\\1', _single_line(value))


def build_citation_labels(evidence_nodes: Sequence[EvidenceGraphNode]) -> dict[str, str]:
	"""Map ``evidence_id`` to ``E1``, ``E2``, ... following graph order exactly.

	Position-based labels are unique by construction, one-based, and stable for a given graph.
	Truncating a uuid instead would risk collisions between ids sharing a prefix and would hand the
	reader something that looks like an identifier but is only part of one; the full id stays on every
	citation so any label can be resolved back to the evidence store.
	"""
	return {node.evidence_id: f'{_CITATION_PREFIX}{index}' for index, node in enumerate(evidence_nodes, start=1)}


class ReportEvidenceSource(BaseModel):
	"""One cited source: the short label a reader sees, plus everything needed to trace it back."""

	citation_label: NonBlankString = Field(description='Position-based label such as E1, generated by Python')
	evidence_id: str = Field(description='Full EvidenceNode.evidence_id, the key that outlives the report')
	url: str = Field(description='Original url string, displayed only as escaped plain text')
	title: str = Field(description='Page title, copied unchanged and escaped when rendered')
	source_host: str = Field(default='', description='Normalized host from Phase 6, empty when unknown')
	step_number: int = Field(ge=1, description='1-based browser step that captured this evidence')


class ReportClaimEvidence(BaseModel):
	"""One citation as used by one claim, with the graph relations that qualify it.

	Source metadata lives on ``ReportEvidenceSource`` instead of being repeated here: a single page can
	support three claims, and three copies of its url and title would be three chances to disagree with
	each other.
	"""

	evidence_id: str = Field(description='EvidenceNode.evidence_id of the cited evidence')
	relation: EvidenceRelation = Field(description='Phase 5 relation, unchanged')
	explanation: NonBlankString = Field(description='Phase 5 rationale, unchanged')
	same_source_evidence_ids: list[str] = Field(default_factory=list, description='Other cited evidence from the same host')
	duplicate_evidence_ids: list[str] = Field(default_factory=list, description='Other cited evidence judged near-duplicate')
	conflicting_evidence_ids: list[str] = Field(
		default_factory=list, description='Evidence this one conflicts with, for this claim and this claim only'
	)


class ClaimReportSection(BaseModel):
	"""One claim, its verdict, and every citation that qualified it."""

	claim_id: str = Field(description='Claim.claim_id, copied unchanged')
	order: int = Field(ge=1, description='Claim.order, which fixes the section order')
	claim_text: NonBlankString = Field(description='The claim as the answer stated it')
	status: VerificationStatus = Field(description='Phase 5 status, never recomputed here')
	evidence: list[ReportClaimEvidence] = Field(
		default_factory=list, description='Citations in Phase 6 edge order; empty only for a NO_EVIDENCE claim'
	)


class ReportSummary(BaseModel):
	"""The headline numbers, every one of them a count or a ratio over the report's own content."""

	claim_count: int = Field(default=0, ge=0, description='Claims in the report')
	evidence_count: int = Field(default=0, ge=0, description='Distinct cited evidence sources')
	unique_source_count: int = Field(
		default=0, ge=0, description='Distinct known source hosts; evidence with no host is not counted as a source'
	)
	supported_claim_count: int = Field(default=0, ge=0, description='Claims with status SUPPORTED')
	partial_claim_count: int = Field(default=0, ge=0, description='Claims with status PARTIAL')
	unsupported_claim_count: int = Field(default=0, ge=0, description='Claims with status UNSUPPORTED')
	contradicted_claim_count: int = Field(default=0, ge=0, description='Claims with status CONTRADICTED')
	conflicted_claim_count: int = Field(default=0, ge=0, description='Claims with status CONFLICTED')
	no_evidence_claim_count: int = Field(default=0, ge=0, description='Claims with status NO_EVIDENCE')
	evidence_covered_claim_count: int = Field(
		default=0, ge=0, description='Claims that had at least one candidate to be checked against'
	)
	evidence_coverage_rate: float = Field(
		default=0.0, ge=0.0, le=1.0, description='Covered claims over all claims, or 0.0 when there are no claims'
	)

	@model_validator(mode='after')
	def _recompute_coverage(self) -> 'ReportSummary':
		"""Derive the coverage figures from the counts, so a stale pair cannot be published."""
		self.evidence_covered_claim_count = max(0, self.claim_count - self.no_evidence_claim_count)
		rate = 0.0 if self.claim_count == 0 else self.evidence_covered_claim_count / self.claim_count
		self.evidence_coverage_rate = min(1.0, max(0.0, rate))
		return self


class EvidenceGroundedReport(BaseModel):
	"""The structured report: one task, its summary, its citations and one section per claim.

	There is no timestamp and no overall verdict. A ``created_at`` would make two builds of identical
	input differ, which is the property every later stage of this pipeline depends on, and a single
	pass or fail would throw away the distribution this report exists to show.
	"""

	task_id: str = Field(description='Task the claims and evidence belong to')
	task: str = Field(default='', description='The original task prompt, from ClaimSet.task')
	summary: ReportSummary = Field(default_factory=ReportSummary, description='Derived headline numbers')
	sources: list[ReportEvidenceSource] = Field(default_factory=list, description='Cited sources, in graph evidence order')
	claims: list[ClaimReportSection] = Field(default_factory=list, description='One section per claim, in Claim.order')


class EvidenceReportBuilder:
	"""Project a Phase 6 ``EvidenceGraph`` into an ``EvidenceGroundedReport``.

	The graph is the only source of facts. There is no store read, no page fetch and no model call, and
	``ClaimSet`` arrives only to supply the task prompt and to cross-check that the graph still
	describes these same claims with the same text and the same order -- the check that turns a stale
	graph into an error instead of a confidently wrong report.

	Annotations come from graph edges rather than from fresh inference. ``SAME_SOURCE`` and
	``DUPLICATE`` hold for a pair regardless of what is being asked, so both neighbours are recorded
	wherever that evidence is cited. ``CONFLICTS_WITH`` is scoped to one claim, so it is recorded for
	that claim alone: two pages that disagree about one claim do not thereby contradict each other about
	every other claim that happens to cite them.
	"""

	def build(self, *, claim_set: ClaimSet, evidence_graph: EvidenceGraph) -> EvidenceGroundedReport:
		"""Build the report for one task.

		Raises:
			EvidenceReportError: when the graph and the claims disagree, when an edge names a node that
				is not in the graph, or when the graph stats no longer match the graph content.
		"""
		if evidence_graph.task_id != claim_set.task_id:
			raise EvidenceReportError(
				f'Task mismatch: evidence graph is for {evidence_graph.task_id!r}, claim set is for {claim_set.task_id!r}'
			)

		claims_by_id = self._index_claims(claim_set)
		graph_claims = self._index_graph_claims(evidence_graph, claims_by_id)
		evidence_by_id = self._index_evidence(evidence_graph)
		self._check_edges(evidence_graph, evidence_by_id, graph_claims)
		self._check_stats(evidence_graph)

		sources = [
			ReportEvidenceSource(
				citation_label=label,
				evidence_id=node.evidence_id,
				url=node.url,
				title=node.title,
				source_host=node.source_host,
				step_number=node.step_number,
			)
			for label, node in zip(build_citation_labels(evidence_graph.evidence).values(), evidence_graph.evidence, strict=True)
		]

		shared = self._neighbours(evidence_graph, EvidenceEdgeType.SAME_SOURCE)
		duplicates = self._neighbours(evidence_graph, EvidenceEdgeType.DUPLICATE)
		conflicts = self._claim_conflicts(evidence_graph)

		sections = [
			ClaimReportSection(
				claim_id=node.claim_id,
				order=node.order,
				claim_text=node.text,
				status=node.verification_status,
				evidence=self._section_evidence(node.claim_id, evidence_graph, shared, duplicates, conflicts),
			)
			for node in sorted(graph_claims.values(), key=lambda node: (node.order, node.claim_id))
		]

		return EvidenceGroundedReport(
			task_id=claim_set.task_id,
			task=claim_set.task,
			summary=self._summary(evidence_graph, sections),
			sources=sources,
			claims=sections,
		)

	@staticmethod
	def _index_claims(claim_set: ClaimSet) -> dict[str, Claim]:
		claims_by_id: dict[str, Claim] = {}
		for claim in claim_set.claims:
			if claim.claim_id in claims_by_id:
				raise EvidenceReportError(f'Claim set contains claim_id {claim.claim_id!r} more than once')
			claims_by_id[claim.claim_id] = claim
		return claims_by_id

	@staticmethod
	def _index_graph_claims(evidence_graph: EvidenceGraph, claims_by_id: dict[str, Claim]) -> dict[str, ClaimGraphNode]:
		"""Require the graph to describe exactly these claims, with the same text and the same order."""
		graph_claims: dict[str, ClaimGraphNode] = {}
		for node in evidence_graph.claims:
			if node.claim_id in graph_claims:
				raise EvidenceReportError(f'Evidence graph contains claim_id {node.claim_id!r} more than once')
			graph_claims[node.claim_id] = node

		missing = sorted(set(claims_by_id) - set(graph_claims))
		if missing:
			raise EvidenceReportError(f'{len(missing)} claim(s) have no graph node, first one: {missing[0]!r}')
		extra = sorted(set(graph_claims) - set(claims_by_id))
		if extra:
			raise EvidenceReportError(f'Evidence graph has {len(extra)} claim(s) the claim set does not, first one: {extra[0]!r}')

		for claim_id in sorted(graph_claims):
			node, claim = graph_claims[claim_id], claims_by_id[claim_id]
			if node.text != claim.text:
				raise EvidenceReportError(f'Claim {claim_id!r} text differs between the claim set and the evidence graph')
			if node.order != claim.order:
				raise EvidenceReportError(
					f'Claim {claim_id!r} is order {claim.order} in the claim set but order {node.order} in the graph'
				)
		return graph_claims

	@staticmethod
	def _index_evidence(evidence_graph: EvidenceGraph) -> dict[str, EvidenceGraphNode]:
		evidence_by_id: dict[str, EvidenceGraphNode] = {}
		for node in evidence_graph.evidence:
			if node.evidence_id in evidence_by_id:
				raise EvidenceReportError(f'Evidence graph contains evidence_id {node.evidence_id!r} more than once')
			evidence_by_id[node.evidence_id] = node
		return evidence_by_id

	@staticmethod
	def _check_edges(
		evidence_graph: EvidenceGraph, evidence_by_id: dict[str, EvidenceGraphNode], graph_claims: dict[str, ClaimGraphNode]
	) -> None:
		"""Every edge endpoint has to name a node of this graph, or the report cites what it cannot show."""
		for edge in evidence_graph.claim_evidence_edges:
			if edge.evidence_id not in evidence_by_id:
				raise EvidenceReportError(
					f'Claim-evidence edge of claim {edge.claim_id!r} references evidence_id {edge.evidence_id!r} which is not in the graph'
				)
			if edge.claim_id not in graph_claims:
				raise EvidenceReportError(f'Claim-evidence edge references claim_id {edge.claim_id!r} which is not in the graph')

		for edge in evidence_graph.evidence_evidence_edges:
			for evidence_id in (edge.source_evidence_id, edge.target_evidence_id):
				if evidence_id not in evidence_by_id:
					raise EvidenceReportError(
						f'{edge.relation.value} edge references evidence_id {evidence_id!r} which is not in the graph'
					)
			if edge.relation is EvidenceEdgeType.CONFLICTS_WITH and edge.claim_id not in graph_claims:
				raise EvidenceReportError(f'CONFLICTS_WITH edge references claim_id {edge.claim_id!r} which is not in the graph')

	@staticmethod
	def _check_stats(evidence_graph: EvidenceGraph) -> None:
		"""Rebuild the graph, which recomputes stats from content, and require them to agree."""
		try:
			rebuilt = EvidenceGraph(
				task_id=evidence_graph.task_id,
				claims=evidence_graph.claims,
				evidence=evidence_graph.evidence,
				claim_evidence_edges=evidence_graph.claim_evidence_edges,
				evidence_evidence_edges=evidence_graph.evidence_evidence_edges,
			)
		except ValidationError as e:
			raise EvidenceReportError(f'Evidence graph is structurally invalid: {type(e).__name__}') from e

		if rebuilt.stats != evidence_graph.stats:
			stale = [name for name, value in evidence_graph.stats.as_dict().items() if getattr(rebuilt.stats, name) != value]
			raise EvidenceReportError(f'Evidence graph stats disagree with its content for: {", ".join(sorted(stale))}')

	@staticmethod
	def _neighbours(evidence_graph: EvidenceGraph, relation: EvidenceEdgeType) -> dict[str, list[str]]:
		"""Flatten undirected evidence-to-evidence edges into one ordered neighbour list per evidence id.

		Both endpoints are recorded because an edge states one pair, not two claims about direction. The
		result is ordered by position in ``EvidenceGraph.evidence``, which is also citation label order,
		so the same pair of pages reads the same way wherever it appears in the report.
		"""
		position = {node.evidence_id: index for index, node in enumerate(evidence_graph.evidence)}
		neighbours: dict[str, set[str]] = {evidence_id: set() for evidence_id in position}
		for edge in evidence_graph.evidence_evidence_edges:
			if edge.relation is relation:
				neighbours[edge.source_evidence_id].add(edge.target_evidence_id)
				neighbours[edge.target_evidence_id].add(edge.source_evidence_id)

		return {evidence_id: sorted(ids, key=lambda other_id: position[other_id]) for evidence_id, ids in neighbours.items()}

	@staticmethod
	def _claim_conflicts(evidence_graph: EvidenceGraph) -> dict[tuple[str, str], list[str]]:
		"""Conflict neighbours per claim, since a conflict is only ever true inside one claim."""
		position = {node.evidence_id: index for index, node in enumerate(evidence_graph.evidence)}
		pairs: dict[tuple[str, str], set[str]] = {}
		for edge in evidence_graph.evidence_evidence_edges:
			if edge.relation is not EvidenceEdgeType.CONFLICTS_WITH or edge.claim_id is None:
				continue
			pairs.setdefault((edge.claim_id, edge.source_evidence_id), set()).add(edge.target_evidence_id)
			pairs.setdefault((edge.claim_id, edge.target_evidence_id), set()).add(edge.source_evidence_id)

		return {key: sorted(ids, key=lambda other_id: position[other_id]) for key, ids in pairs.items()}

	@staticmethod
	def _section_evidence(
		claim_id: str,
		evidence_graph: EvidenceGraph,
		shared: dict[str, list[str]],
		duplicates: dict[str, list[str]],
		conflicts: dict[tuple[str, str], list[str]],
	) -> list[ReportClaimEvidence]:
		"""One claim's citations, in Phase 6 edge order, carrying their annotations.

		Edge order is kept rather than re-sorted because it already encodes Phase 4B candidate rank,
		which is the only defensible ordering left at this point. Relation grouping is a rendering
		decision and belongs to the renderer, so the model stays a faithful projection of the graph.
		"""
		return [
			ReportClaimEvidence(
				evidence_id=edge.evidence_id,
				relation=edge.relation,
				explanation=edge.explanation,
				same_source_evidence_ids=list(shared.get(edge.evidence_id, ())),
				duplicate_evidence_ids=list(duplicates.get(edge.evidence_id, ())),
				conflicting_evidence_ids=list(conflicts.get((claim_id, edge.evidence_id), ())),
			)
			for edge in evidence_graph.claim_evidence_edges
			if edge.claim_id == claim_id
		]

	@staticmethod
	def _summary(evidence_graph: EvidenceGraph, sections: Sequence[ClaimReportSection]) -> ReportSummary:
		"""Count over the report's own sections, which the stats check already tied to the graph."""
		status_counts = {status: 0 for status in VerificationStatus}
		for section in sections:
			status_counts[section.status] += 1

		return ReportSummary(
			claim_count=len(sections),
			evidence_count=len(evidence_graph.evidence),
			unique_source_count=len({node.source_host for node in evidence_graph.evidence if node.source_host}),
			supported_claim_count=status_counts[VerificationStatus.SUPPORTED],
			partial_claim_count=status_counts[VerificationStatus.PARTIAL],
			unsupported_claim_count=status_counts[VerificationStatus.UNSUPPORTED],
			contradicted_claim_count=status_counts[VerificationStatus.CONTRADICTED],
			conflicted_claim_count=status_counts[VerificationStatus.CONFLICTED],
			no_evidence_claim_count=status_counts[VerificationStatus.NO_EVIDENCE],
		)


class MarkdownReportRenderer:
	"""Render an ``EvidenceGroundedReport`` as deterministic, injection-safe Markdown.

	Layout is fixed here and content comes only from the report. Every untrusted string passes through
	:func:`escape_report_text` or :func:`escape_report_url`, evidence is named by its short citation
	label rather than its uuid, and urls are shown as plain text instead of links. Two identical
	reports therefore produce identical bytes, which is what lets a report be diffed, cached, or used as
	benchmark ground truth with no model anywhere in the comparison.
	"""

	def render(self, report: EvidenceGroundedReport) -> str:
		"""Produce the document for one report."""
		sources = {source.evidence_id: source for source in report.sources}
		lines: list[str] = [
			_REPORT_TITLE,
			'',
			'## Task',
			'',
			self._task_line(report.task),
			'',
			*self._summary_lines(report.summary),
			*self._claim_lines(report, sources),
			*self._source_lines(report, sources),
			'## Interpretation',
			'',
			*_INTERPRETATION_LINES,
			'',
		]
		return '\n'.join(lines)

	@staticmethod
	def _task_line(task: str) -> str:
		return escape_report_text(task) or _NO_TASK_TEXT

	@staticmethod
	def _summary_lines(summary: ReportSummary) -> list[str]:
		return [
			'## Verification Summary',
			'',
			f'Claims: {summary.claim_count}',
			f'Evidence sources: {summary.evidence_count}',
			f'Unique sources: {summary.unique_source_count}',
			f'Evidence coverage: {summary.evidence_coverage_rate:.1%} ({summary.evidence_covered_claim_count} of {summary.claim_count} claims)',
			'',
			f'{VerificationStatus.SUPPORTED.value}: {summary.supported_claim_count}',
			f'{VerificationStatus.PARTIAL.value}: {summary.partial_claim_count}',
			f'{VerificationStatus.UNSUPPORTED.value}: {summary.unsupported_claim_count}',
			f'{VerificationStatus.CONTRADICTED.value}: {summary.contradicted_claim_count}',
			f'{VerificationStatus.CONFLICTED.value}: {summary.conflicted_claim_count}',
			f'{VerificationStatus.NO_EVIDENCE.value}: {summary.no_evidence_claim_count}',
			'',
		]

	@staticmethod
	def _claim_lines(report: EvidenceGroundedReport, sources: dict[str, ReportEvidenceSource]) -> list[str]:
		lines = ['## Claim Verification', '']
		if not report.claims:
			return [*lines, _NO_CLAIMS_TEXT, '']

		for section in report.claims:
			lines.append(f'### Claim {section.order}: {section.status.value}')
			lines.append('')
			lines.append(escape_report_text(section.claim_text))
			lines.append('')
			if not section.evidence:
				lines.extend([_NO_EVIDENCE_TEXT, ''])
				continue
			lines.extend(MarkdownReportRenderer._grouped_evidence_lines(section, sources))
		return lines

	@staticmethod
	def _grouped_evidence_lines(section: ClaimReportSection, sources: dict[str, ReportEvidenceSource]) -> list[str]:
		"""Group a claim's citations by relation, keeping report order inside each group.

		Grouping is a display decision only. For a CONFLICTED claim it is what makes the structure
		obvious: the supporting citations and the contradicting ones appear under different headings and
		each names the other, so the disagreement is legible without anyone being told which side wins.
		"""
		lines: list[str] = []
		for relation, heading in _RELATION_GROUPS:
			cited = [item for item in section.evidence if item.relation is relation]
			if not cited:
				continue
			lines.append(f'{heading} ({len(cited)}):')
			lines.append('')
			for item in cited:
				lines.extend(MarkdownReportRenderer._citation_lines(item, sources))
		return lines

	@staticmethod
	def _citation_lines(item: ReportClaimEvidence, sources: dict[str, ReportEvidenceSource]) -> list[str]:
		source = sources.get(item.evidence_id)
		label = source.citation_label if source is not None else _CITATION_PREFIX + '?'
		lines = [f'- [{label}] {escape_report_text(item.relation.value)}']
		if source is not None:
			lines.append(f'  - Source: {escape_report_text(source.source_host) or "(unknown host)"}')
			lines.append(f'  - Title: {escape_report_text(source.title) or "(untitled)"}')
			lines.append(f'  - URL: {escape_report_url(source.url) or "(none)"}')
		lines.append(f'  - Assessment: {escape_report_text(item.explanation)}')
		lines.extend(MarkdownReportRenderer._annotation_lines('Same source as', item.same_source_evidence_ids, sources))
		lines.extend(MarkdownReportRenderer._annotation_lines('Duplicate of', item.duplicate_evidence_ids, sources))
		lines.extend(MarkdownReportRenderer._annotation_lines('Conflicts with', item.conflicting_evidence_ids, sources))
		lines.append('')
		return lines

	@staticmethod
	def _annotation_lines(
		heading: str,
		evidence_ids: Sequence[str],
		sources: dict[str, ReportEvidenceSource],
	) -> list[str]:
		"""Show relations as citation labels, never as a wall of uuids."""
		if not evidence_ids:
			return []
		citations = ', '.join(
			f'[{sources[evidence_id].citation_label}]' for evidence_id in evidence_ids if evidence_id in sources
		)
		return [f'  - {heading}: {citations}'] if citations else []

	@staticmethod
	def _source_lines(report: EvidenceGroundedReport, sources: dict[str, ReportEvidenceSource]) -> list[str]:
		lines = ['## Evidence Sources', '']
		if not report.sources:
			return [*lines, _NO_EVIDENCE_TEXT, '']

		cited_by = {source.evidence_id: [] for source in report.sources}
		for section in report.claims:
			for item in section.evidence:
				if item.evidence_id in cited_by and section.order not in cited_by[item.evidence_id]:
					cited_by[item.evidence_id].append(section.order)

		for source in report.sources:
			numbers = ', '.join(str(order) for order in cited_by.get(source.evidence_id, ()))
			lines.append(f'- [{source.citation_label}] {escape_report_text(source.title) or "(untitled)"}')
			lines.append(f'  - Source: {escape_report_text(source.source_host) or "(unknown host)"}')
			lines.append(f'  - URL: {escape_report_url(source.url) or "(none)"}')
			lines.append(f'  - Captured at browser step: {source.step_number}')
			lines.append(f'  - Cited by claim: {numbers}' if numbers else f'  - {_NOT_CITED_TEXT}')
		lines.append('')
		return lines
