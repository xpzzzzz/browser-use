"""Unit tests for the evidence-grounded user report and its Markdown renderer.

Phase 7 is pure Python. Graphs come from the real Phase 6 organizer so these tests exercise the same
structures the pipeline actually produces, and nothing here may reach a model, a browser or a network.
Letter tags in comments map onto the Phase 7 specification checklist.
"""

import random
import re
from copy import deepcopy
from pathlib import Path

import pytest

from browser_use.evidence import (
	Claim,
	ClaimEvidenceEdge,
	ClaimGraphNode,
	ClaimReportSection,
	ClaimSet,
	ClaimVerification,
	EvidenceAssessment,
	EvidenceEdgeType,
	EvidenceEvidenceEdge,
	EvidenceGraph,
	EvidenceGraphNode,
	EvidenceGroundedReport,
	EvidenceNode,
	EvidenceOrganizer,
	EvidenceRelation,
	EvidenceReportBuilder,
	EvidenceReportError,
	MarkdownReportRenderer,
	ReportClaimEvidence,
	ReportEvidenceSource,
	ReportSummary,
	VerificationResult,
	VerificationStatus,
)
from browser_use.evidence.organization import EvidenceGraphStats
from browser_use.evidence.reporting import build_citation_labels, escape_report_text, escape_report_url

STARS_CLAIM = 'Browser Use has more than 100,000 GitHub stars.'
LANGUAGE_CLAIM = 'Browser Use is primarily written in Python.'
CREATED_CLAIM = 'Project X was created in 2019.'
HIGH_STAR_TEXT = 'Browser Use has 111,799 GitHub stars.'
LOW_STAR_TEXT = 'Browser Use has 30,000 GitHub stars.'
LANGUAGE_TEXT = 'Browser Use is primarily written in Python.'

GITHUB_URL = 'https://www.github.com/browser-use/browser-use'
MIRROR_URL = 'https://mirror.example.com/stars'
DOCS_URL = 'https://docs.python.org/3/library/app.html'


def _node(
	evidence_id: str, text: str, *, url: str = 'https://example.com/p', title: str = '', step_number: int = 1
) -> EvidenceNode:
	return EvidenceNode(evidence_id=evidence_id, task_id='task-1', step_number=step_number, url=url, title=title, text=text)


def _claim_set(*claims: tuple[str, int, str], task: str = 'How popular is Browser Use?') -> ClaimSet:
	return ClaimSet(
		task_id='task-1',
		task=task,
		answer=' '.join(text for _, _, text in claims),
		claims=[Claim(claim_id=claim_id, order=order, text=text) for claim_id, order, text in claims],
	)


def _assessment(evidence_id: str, relation: EvidenceRelation, explanation: str | None = None) -> EvidenceAssessment:
	return EvidenceAssessment(
		evidence_id=evidence_id,
		relation=relation,
		explanation=explanation or f'{evidence_id} {relation.value.lower()}.',
	)


def _verification(claim_id: str, status: VerificationStatus, *assessments: EvidenceAssessment) -> ClaimVerification:
	return ClaimVerification(claim_id=claim_id, status=status, assessments=list(assessments))


def _graph(claim_set: ClaimSet, verifications: list[ClaimVerification], nodes: list[EvidenceNode]) -> EvidenceGraph:
	"""Run the real Phase 6 organizer, so every report input is a graph the pipeline could produce."""
	result = VerificationResult(task_id=claim_set.task_id, verifications=verifications)
	return EvidenceOrganizer().organize(claim_set=claim_set, verification_result=result, evidence_nodes=nodes)


def _report(claim_set: ClaimSet, verifications: list[ClaimVerification], nodes: list[EvidenceNode]) -> EvidenceGroundedReport:
	return EvidenceReportBuilder().build(claim_set=claim_set, evidence_graph=_graph(claim_set, verifications, nodes))


def _section(report: EvidenceGroundedReport, claim_id: str) -> ClaimReportSection:
	return next(section for section in report.claims if section.claim_id == claim_id)


def _conflicted_fixture():
	"""Claim 1 supported by one page, contradicted by another, and echoed by a third claim."""
	claim_set = _claim_set(('claim-1', 1, STARS_CLAIM), ('claim-2', 2, LANGUAGE_CLAIM))
	nodes = [
		_node('evidence-high', HIGH_STAR_TEXT, url=GITHUB_URL, title='GitHub', step_number=1),
		_node('evidence-low', LOW_STAR_TEXT, url='https://blog.example.com/old', title='Old post', step_number=2),
		_node('evidence-language', LANGUAGE_TEXT, url=DOCS_URL, title='Docs', step_number=3),
	]
	verifications = [
		_verification(
			'claim-1',
			VerificationStatus.CONFLICTED,
			_assessment('evidence-high', EvidenceRelation.SUPPORTS, explanation='111,799 clears the threshold.'),
			_assessment('evidence-low', EvidenceRelation.CONTRADICTS, explanation='30,000 is below the threshold.'),
		),
		_verification(
			'claim-2',
			VerificationStatus.SUPPORTED,
			_assessment('evidence-language', EvidenceRelation.SUPPORTS, explanation='the docs name the language.'),
		),
	]
	return claim_set, verifications, nodes


class TestClaimSections:
	@pytest.mark.parametrize(
		'status',
		[
			VerificationStatus.SUPPORTED,  # A
			VerificationStatus.PARTIAL,  # B
			VerificationStatus.UNSUPPORTED,  # C
			VerificationStatus.CONTRADICTED,  # D
			VerificationStatus.CONFLICTED,  # E
			VerificationStatus.NO_EVIDENCE,  # F
		],
	)
	def test_every_status_produces_exactly_one_section(self, status):
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [_node('evidence-a', HIGH_STAR_TEXT, step_number=1)]
		assessment = (
			_assessment('evidence-a', EvidenceRelation.SUPPORTS) if status is not VerificationStatus.NO_EVIDENCE else None
		)
		verifications = [
			_verification('claim-1', status, *([assessment] if assessment else [])),
		]

		report = _report(claim_set, verifications, nodes)

		assert len(report.claims) == 1
		assert report.claims[0].status is status
		assert report.claims[0].claim_text == STARS_CLAIM

	def test_no_evidence_claim_is_kept_with_no_citations(self):  # G, item 5
		claim_set, verifications, nodes = _conflicted_fixture()
		verifications.append(_verification('claim-3', VerificationStatus.NO_EVIDENCE))
		claim_set.claims.append(Claim(claim_id='claim-3', order=3, text=CREATED_CLAIM))

		report = _report(claim_set, verifications, nodes)

		section = _section(report, 'claim-3')
		assert section.status is VerificationStatus.NO_EVIDENCE
		assert section.evidence == []
		assert len(report.claims) == 3

	def test_unsupported_claim_keeps_its_insufficient_citations(self):  # H, item 28
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [_node('evidence-language', LANGUAGE_TEXT, url=DOCS_URL, title='Docs', step_number=1)]
		verifications = [
			_verification(
				'claim-1',
				VerificationStatus.UNSUPPORTED,
				_assessment('evidence-language', EvidenceRelation.INSUFFICIENT, explanation='the docs give no star count.'),
			)
		]

		report = _report(claim_set, verifications, nodes)

		cited = report.claims[0].evidence
		assert [(item.evidence_id, item.relation) for item in cited] == [('evidence-language', EvidenceRelation.INSUFFICIENT)]
		assert cited[0].explanation == 'the docs give no star count.'

	def test_partial_claim_keeps_the_partial_relation(self):  # item 31
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [_node('evidence-a', HIGH_STAR_TEXT, url=GITHUB_URL, step_number=1)]
		verifications = [
			_verification('claim-1', VerificationStatus.PARTIAL, _assessment('evidence-a', EvidenceRelation.PARTIAL_SUPPORT))
		]

		report = _report(claim_set, verifications, nodes)

		assert report.claims[0].evidence[0].relation is EvidenceRelation.PARTIAL_SUPPORT
		assert report.claims[0].status is VerificationStatus.PARTIAL


class TestSourcesAndLabels:
	def test_labels_are_e1_e2_e3_in_graph_order(self):  # I
		claim_set, verifications, nodes = _conflicted_fixture()

		sources = _report(claim_set, verifications, nodes).sources

		assert [source.citation_label for source in sources] == ['E1', 'E2', 'E3']
		assert [source.evidence_id for source in sources] == ['evidence-high', 'evidence-low', 'evidence-language']

	def test_labels_are_deterministic_across_builds_and_shuffles(self):  # J
		claim_set, verifications, nodes = _conflicted_fixture()
		builder = EvidenceReportBuilder()
		graph = _graph(claim_set, verifications, nodes)

		first = builder.build(claim_set=claim_set, evidence_graph=graph).sources
		shuffled_nodes = deepcopy(nodes)
		random.Random(3).shuffle(shuffled_nodes)
		second = builder.build(claim_set=claim_set, evidence_graph=_graph(claim_set, verifications, shuffled_nodes)).sources

		assert [(source.citation_label, source.evidence_id) for source in first] == [
			(source.citation_label, source.evidence_id) for source in second
		]

	def test_source_metadata_is_carried_once_per_evidence(self):  # K
		claim_set, verifications, nodes = _conflicted_fixture()

		sources = _report(claim_set, verifications, nodes).sources

		assert sources[0].url == GITHUB_URL
		assert sources[0].title == 'GitHub'
		assert sources[0].source_host == 'github.com'
		assert sources[0].step_number == 1
		assert not {field for field in ReportClaimEvidence.model_fields} & {'url', 'title', 'source_host', 'step_number'}

	def test_labels_map_positions_not_uuid_prefixes(self):  # item 11
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		shared_prefix = '01988576-4c3d-7a10-8b25-'
		nodes = [
			_node(f'{shared_prefix}0001', HIGH_STAR_TEXT, url=GITHUB_URL, step_number=1),
			_node(f'{shared_prefix}0002', LANGUAGE_TEXT, url=DOCS_URL, step_number=2),
		]
		verifications = [
			_verification(
				'claim-1',
				VerificationStatus.SUPPORTED,
				_assessment(f'{shared_prefix}0001', EvidenceRelation.SUPPORTS),
				_assessment(f'{shared_prefix}0002', EvidenceRelation.INSUFFICIENT),
			)
		]

		report = _report(claim_set, verifications, nodes)

		assert [source.citation_label for source in report.sources] == ['E1', 'E2']
		assert [item.evidence_id for item in report.claims[0].evidence] == [f'{shared_prefix}0001', f'{shared_prefix}0002']

	def test_label_builder_is_one_based_and_exhaustive(self):
		nodes = [_node(f'evidence-{index}', 'text', step_number=index) for index in range(1, 6)]

		labels = build_citation_labels(nodes)

		assert list(labels.values()) == ['E1', 'E2', 'E3', 'E4', 'E5']
		assert list(labels) == [node.evidence_id for node in nodes]


class TestAnnotations:
	def test_same_source_annotation_records_both_directions(self):  # L
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [
			_node(
				'evidence-a', 'Release notes list 111,799 stars for the repository.', url='https://github.com/a', step_number=1
			),
			_node('evidence-b', 'Star history chart shows a rise to 111,799.', url='https://www.github.com/b', step_number=2),
		]
		verifications = [
			_verification(
				'claim-1',
				VerificationStatus.SUPPORTED,
				_assessment('evidence-a', EvidenceRelation.SUPPORTS),
				_assessment('evidence-b', EvidenceRelation.SUPPORTS),
			)
		]

		cited = _report(claim_set, verifications, nodes).claims[0].evidence

		assert [item.same_source_evidence_ids for item in cited] == [['evidence-b'], ['evidence-a']]
		assert [item.duplicate_evidence_ids for item in cited] == [[], []]

	def test_duplicate_annotation_records_both_directions(self):  # M
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [
			_node('evidence-a', HIGH_STAR_TEXT, url=GITHUB_URL, title='GitHub', step_number=1),
			_node('evidence-b', HIGH_STAR_TEXT, url=MIRROR_URL, title='GitHub', step_number=2),
		]
		verifications = [
			_verification(
				'claim-1',
				VerificationStatus.SUPPORTED,
				_assessment('evidence-a', EvidenceRelation.SUPPORTS),
				_assessment('evidence-b', EvidenceRelation.SUPPORTS),
			)
		]

		report = _report(claim_set, verifications, nodes)

		cited = report.claims[0].evidence
		assert [item.duplicate_evidence_ids for item in cited] == [['evidence-b'], ['evidence-a']]
		# Both copies stay in the report; annotation is not deletion (item 13).
		assert [source.evidence_id for source in report.sources] == ['evidence-a', 'evidence-b']

	def test_conflict_annotation_applies_only_to_its_own_claim(self):  # N, O, item 14
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM), ('claim-2', 2, LANGUAGE_CLAIM))
		nodes = [
			_node('evidence-high', HIGH_STAR_TEXT, url=GITHUB_URL, step_number=1),
			_node('evidence-low', LOW_STAR_TEXT, url='https://blog.example.com/old', step_number=2),
		]
		verifications = [
			_verification(
				'claim-1',
				VerificationStatus.CONFLICTED,
				_assessment('evidence-high', EvidenceRelation.SUPPORTS),
				_assessment('evidence-low', EvidenceRelation.CONTRADICTS),
			),
			_verification(
				'claim-2',
				VerificationStatus.SUPPORTED,
				_assessment('evidence-high', EvidenceRelation.SUPPORTS),
				_assessment('evidence-low', EvidenceRelation.INSUFFICIENT),
			),
		]

		report = _report(claim_set, verifications, nodes)

		assert [item.conflicting_evidence_ids for item in _section(report, 'claim-1').evidence] == [
			['evidence-low'],
			['evidence-high'],
		]
		assert [item.conflicting_evidence_ids for item in _section(report, 'claim-2').evidence] == [[], []]

	def test_annotations_are_ordered_by_citation_label(self):
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [
			_node('evidence-z', 'First page about the star count.', url='https://github.com/z', step_number=1),
			_node('evidence-m', 'Second page about the star count.', url='https://github.com/m', step_number=2),
			_node('evidence-a', 'Third page about the star count.', url='https://github.com/a', step_number=3),
		]
		verifications = [
			_verification(
				'claim-1',
				VerificationStatus.SUPPORTED,
				*[_assessment(node.evidence_id, EvidenceRelation.SUPPORTS) for node in nodes],
			)
		]

		cited = _report(claim_set, verifications, nodes).claims[0].evidence

		# Graph order is E1 evidence-z, E2 evidence-m, E3 evidence-a, and annotations follow it.
		assert [item.same_source_evidence_ids for item in cited] == [
			['evidence-m', 'evidence-a'],
			['evidence-z', 'evidence-a'],
			['evidence-z', 'evidence-m'],
		]


class TestOrdering:
	def test_sections_follow_claim_order(self):  # Q
		claim_set = ClaimSet(
			task_id='task-1',
			task='order',
			answer='x',
			claims=[
				Claim(claim_id='claim-b', order=7, text=LANGUAGE_CLAIM),
				Claim(claim_id='claim-a', order=2, text=STARS_CLAIM),
			],
		)
		nodes = [_node('evidence-a', HIGH_STAR_TEXT, url=GITHUB_URL, step_number=1)]
		verifications = [
			_verification('claim-b', VerificationStatus.SUPPORTED, _assessment('evidence-a', EvidenceRelation.SUPPORTS)),
			_verification('claim-a', VerificationStatus.SUPPORTED, _assessment('evidence-a', EvidenceRelation.SUPPORTS)),
		]

		report = _report(claim_set, verifications, nodes)

		assert [(section.claim_id, section.order) for section in report.claims] == [('claim-a', 2), ('claim-b', 7)]

	def test_sources_follow_graph_evidence_order(self):  # R
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [
			_node('evidence-late', 'Late page about the stars.', url=DOCS_URL, step_number=9),
			_node('evidence-early', 'Early page about the stars.', url=GITHUB_URL, step_number=2),
		]
		verifications = [
			_verification(
				'claim-1',
				VerificationStatus.SUPPORTED,
				_assessment('evidence-late', EvidenceRelation.SUPPORTS),
				_assessment('evidence-early', EvidenceRelation.SUPPORTS),
			)
		]

		sources = _report(claim_set, verifications, nodes).sources

		assert [source.evidence_id for source in sources] == ['evidence-early', 'evidence-late']
		assert [source.citation_label for source in sources] == ['E1', 'E2']

	def test_citations_keep_phase_6_edge_order(self):  # S
		claim_set, verifications, nodes = _conflicted_fixture()

		report = _report(claim_set, verifications, nodes)
		graph = _graph(claim_set, verifications, nodes)

		edge_order = [edge.evidence_id for edge in graph.claim_evidence_edges if edge.claim_id == 'claim-1']
		assert [item.evidence_id for item in _section(report, 'claim-1').evidence] == edge_order


class TestSummary:
	def test_status_counts_match_the_sections(self):  # T
		claim_set = _claim_set(
			('claim-1', 1, 'A'),
			('claim-2', 2, 'B'),
			('claim-3', 3, 'C'),
			('claim-4', 4, 'D'),
			('claim-5', 5, 'E'),
			('claim-6', 6, 'F'),
		)
		nodes = [_node('evidence-a', HIGH_STAR_TEXT, url=GITHUB_URL, step_number=1)]
		supports = _assessment('evidence-a', EvidenceRelation.SUPPORTS)
		verifications = [
			_verification('claim-1', VerificationStatus.SUPPORTED, supports),
			_verification('claim-2', VerificationStatus.PARTIAL, supports),
			_verification('claim-3', VerificationStatus.UNSUPPORTED, _assessment('evidence-a', EvidenceRelation.INSUFFICIENT)),
			_verification('claim-4', VerificationStatus.CONTRADICTED, _assessment('evidence-a', EvidenceRelation.CONTRADICTS)),
			_verification('claim-5', VerificationStatus.CONFLICTED, supports),
			_verification('claim-6', VerificationStatus.NO_EVIDENCE),
		]

		summary = _report(claim_set, verifications, nodes).summary

		assert summary.claim_count == 6
		assert summary.evidence_count == 1
		assert (
			summary.supported_claim_count,
			summary.partial_claim_count,
			summary.unsupported_claim_count,
			summary.contradicted_claim_count,
			summary.conflicted_claim_count,
			summary.no_evidence_claim_count,
		) == (1, 1, 1, 1, 1, 1)

	def test_unique_sources_count_distinct_hosts(self):  # U, V
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [
			_node('evidence-a', 'First page about the star count.', url='https://www.github.com/a', step_number=1),
			_node('evidence-b', 'Second page about the star count.', url='https://github.com/b', step_number=2),
			_node('evidence-c', 'Third page about the star count.', url=DOCS_URL, step_number=3),
			_node('evidence-d', 'Fourth page about the star count.', url='about:blank', step_number=4),
		]
		verifications = [
			_verification(
				'claim-1',
				VerificationStatus.SUPPORTED,
				*[_assessment(node.evidence_id, EvidenceRelation.SUPPORTS) for node in nodes],
			)
		]

		report = _report(claim_set, verifications, nodes)

		assert report.summary.evidence_count == 4
		assert report.summary.unique_source_count == 2
		assert [source.source_host for source in report.sources] == ['github.com', 'github.com', 'docs.python.org', '']

	def test_coverage_counts_and_rate(self):  # W, X
		claim_set = _claim_set(
			('claim-1', 1, 'A'), ('claim-2', 2, 'B'), ('claim-3', 3, 'C'), ('claim-4', 4, 'D'), ('claim-5', 5, 'E')
		)
		nodes = [_node('evidence-a', HIGH_STAR_TEXT, url=GITHUB_URL, step_number=1)]
		supports = _assessment('evidence-a', EvidenceRelation.SUPPORTS)
		verifications = [_verification(f'claim-{index}', VerificationStatus.SUPPORTED, supports) for index in range(1, 5)] + [
			_verification('claim-5', VerificationStatus.NO_EVIDENCE)
		]

		summary = _report(claim_set, verifications, nodes).summary

		assert summary.claim_count == 5
		assert summary.no_evidence_claim_count == 1
		assert summary.evidence_covered_claim_count == 4
		assert summary.evidence_coverage_rate == pytest.approx(0.8)

	def test_coverage_of_nothing_is_zero_not_a_crash(self):  # Y
		claim_set = _claim_set()

		report = _report(claim_set, [], [])

		assert report.summary.claim_count == 0
		assert report.summary.evidence_covered_claim_count == 0
		assert report.summary.evidence_coverage_rate == 0.0
		assert report.claims == []
		assert report.sources == []

	def test_coverage_figures_are_derived_not_supplied(self):
		summary = ReportSummary(
			claim_count=4, no_evidence_claim_count=1, evidence_covered_claim_count=99, evidence_coverage_rate=0.99
		)

		assert summary.evidence_covered_claim_count == 3
		assert summary.evidence_coverage_rate == pytest.approx(0.75)


class TestIntegrity:
	"""Spec 10: a stale or broken graph stops the build instead of becoming a confident report."""

	def test_task_id_mismatch_is_rejected(self):  # Z
		claim_set, verifications, nodes = _conflicted_fixture()
		graph = _graph(claim_set, verifications, nodes).model_copy(update={'task_id': 'task-other'})

		with pytest.raises(EvidenceReportError, match='Task mismatch'):
			EvidenceReportBuilder().build(claim_set=claim_set, evidence_graph=graph)

	def test_claim_id_sets_must_match_in_both_directions(self):  # AA
		claim_set, verifications, nodes = _conflicted_fixture()
		graph = _graph(claim_set, verifications, nodes)

		# A claim set that has dropped one of the verified claims is a stale pairing, in either direction.
		too_few = _claim_set(('claim-1', 1, STARS_CLAIM))
		with pytest.raises(EvidenceReportError, match='the claim set does not'):
			EvidenceReportBuilder().build(claim_set=too_few, evidence_graph=graph)

		too_many = _claim_set(('claim-1', 1, STARS_CLAIM), ('claim-2', 2, LANGUAGE_CLAIM), ('claim-3', 3, CREATED_CLAIM))
		with pytest.raises(EvidenceReportError, match='have no graph node'):
			EvidenceReportBuilder().build(claim_set=too_many, evidence_graph=graph)

	def test_duplicate_claim_id_in_the_claim_set_is_rejected(self):  # AB
		claim_set, verifications, nodes = _conflicted_fixture()
		claim_set.claims[1].claim_id = 'claim-1'
		graph = _graph(_claim_set(('claim-1', 1, STARS_CLAIM), ('claim-2', 2, LANGUAGE_CLAIM)), verifications, nodes)

		with pytest.raises(EvidenceReportError, match='Claim set contains claim_id'):
			EvidenceReportBuilder().build(claim_set=claim_set, evidence_graph=graph)

	def test_duplicate_claim_id_in_the_graph_is_rejected(self):  # AB
		claim_set, verifications, nodes = _conflicted_fixture()
		graph = _graph(claim_set, verifications, nodes)
		stale = graph.model_copy(update={'claims': [*graph.claims, graph.claims[0]]})

		with pytest.raises(EvidenceReportError, match='Evidence graph contains claim_id'):
			EvidenceReportBuilder().build(claim_set=claim_set, evidence_graph=stale)

	def test_claim_text_mismatch_is_rejected(self):  # AC
		claim_set, verifications, nodes = _conflicted_fixture()
		graph = _graph(claim_set, verifications, nodes)
		claim_set.claims[0].text = 'Browser Use has fewer than 100,000 GitHub stars.'

		with pytest.raises(EvidenceReportError, match='text differs'):
			EvidenceReportBuilder().build(claim_set=claim_set, evidence_graph=graph)

	def test_claim_order_mismatch_is_rejected(self):  # AD
		claim_set, verifications, nodes = _conflicted_fixture()
		graph = _graph(claim_set, verifications, nodes)
		claim_set.claims[0].order = 99

		with pytest.raises(EvidenceReportError, match='is order 99 in the claim set'):
			EvidenceReportBuilder().build(claim_set=claim_set, evidence_graph=graph)

	def test_edge_to_unknown_evidence_is_rejected(self):  # AE, AF
		claim_set, verifications, nodes = _conflicted_fixture()
		graph = _graph(claim_set, verifications, nodes)

		phantom_claim_edge = graph.model_copy(
			update={
				'claim_evidence_edges': [
					*graph.claim_evidence_edges,
					ClaimEvidenceEdge(
						claim_id='claim-1',
						evidence_id='evidence-phantom',
						relation=EvidenceRelation.SUPPORTS,
						explanation='invented',
					),
				]
			}
		)
		with pytest.raises(EvidenceReportError, match='evidence_id .* which is not in the graph'):
			EvidenceReportBuilder().build(claim_set=claim_set, evidence_graph=phantom_claim_edge)

		phantom_pair_edge = graph.model_copy(
			update={
				'evidence_evidence_edges': [
					*graph.evidence_evidence_edges,
					EvidenceEvidenceEdge(
						source_evidence_id='evidence-a',
						target_evidence_id='evidence-phantom',
						relation=EvidenceEdgeType.SAME_SOURCE,
					),
				]
			}
		)
		with pytest.raises(EvidenceReportError, match='SAME_SOURCE edge references evidence_id'):
			EvidenceReportBuilder().build(claim_set=claim_set, evidence_graph=phantom_pair_edge)

	def test_conflict_edge_to_unknown_claim_is_rejected(self):  # AG
		claim_set, verifications, nodes = _conflicted_fixture()
		graph = _graph(claim_set, verifications, nodes)
		conflicts = [edge for edge in graph.evidence_evidence_edges if edge.relation is EvidenceEdgeType.CONFLICTS_WITH]
		orphaned = conflicts[0].model_copy(update={'claim_id': 'claim-ghost'})

		stale = graph.model_copy(update={'evidence_evidence_edges': [orphaned, *graph.evidence_evidence_edges[1:]]})
		with pytest.raises(EvidenceReportError, match='CONFLICTS_WITH edge references claim_id'):
			EvidenceReportBuilder().build(claim_set=claim_set, evidence_graph=stale)

	def test_stale_stats_are_rejected_rather_than_fixed(self):  # AH
		claim_set, verifications, nodes = _conflicted_fixture()
		graph = _graph(claim_set, verifications, nodes)
		stale = graph.model_copy(update={'stats': EvidenceGraphStats(**{**graph.stats.as_dict(), 'claim_count': 99})})

		with pytest.raises(EvidenceReportError, match='stats disagree with its content for: claim_count'):
			EvidenceReportBuilder().build(claim_set=claim_set, evidence_graph=stale)

	def test_errors_carry_ids_not_claim_or_page_text(self):
		claim_set, verifications, nodes = _conflicted_fixture()
		nodes[0].text = 'PRIVATE_EVIDENCE_BODY'
		graph = _graph(claim_set, verifications, nodes)
		claim_set.claims[0].text = 'PRIVATE_CLAIM_BODY'

		with pytest.raises(EvidenceReportError) as excinfo:
			EvidenceReportBuilder().build(claim_set=claim_set, evidence_graph=graph)

		message = str(excinfo.value)
		assert 'PRIVATE_CLAIM_BODY' not in message
		assert 'PRIVATE_EVIDENCE_BODY' not in message
		assert 'claim-1' in message


class TestMarkdownContent:
	@pytest.fixture
	def document(self):
		claim_set, verifications, nodes = _conflicted_fixture()
		return MarkdownReportRenderer().render(_report(claim_set, verifications, nodes))

	def test_document_opens_with_the_title_and_task(self, document):  # AM
		assert document.startswith('# WebEvidence Verification Report\n')
		assert '## Task' in document
		assert 'How popular is Browser Use?' in document

	def test_each_claim_shows_its_status(self, document):  # AN
		assert '### Claim 1: CONFLICTED' in document
		assert '### Claim 2: SUPPORTED' in document

	def test_citations_use_short_labels(self, document):  # AO, P
		assert '- [E1] SUPPORTS' in document
		assert '- [E2] CONTRADICTS' in document
		assert '[E1]' in document and '[E2]' in document

	def test_source_host_and_url_are_shown(self, document):  # AP, AQ
		assert '  - Source: github.com' in document
		assert '  - URL: https://www.github.com/browser-use/browser-use' in document
		assert '  - Captured at browser step: 1' in document

	def test_assessment_rationale_is_shown(self, document):  # AR
		assert '  - Assessment: 111,799 clears the threshold.' in document
		assert '  - Assessment: 30,000 is below the threshold.' in document

	def test_full_ids_are_not_the_reader_facing_label(self):  # AS
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		node_id = '01988576-4c3d-7a10-8b25-1a2b3c4d5e6f'
		nodes = [_node(node_id, HIGH_STAR_TEXT, url=GITHUB_URL, title='GitHub', step_number=1)]
		verifications = [
			_verification(
				'claim-1',
				VerificationStatus.SUPPORTED,
				_assessment(node_id, EvidenceRelation.SUPPORTS, explanation='the repository shows the count.'),
			)
		]

		report = _report(claim_set, verifications, nodes)
		document = MarkdownReportRenderer().render(report)

		assert '[E1]' in document
		assert node_id not in document
		assert report.sources[0].evidence_id == node_id

	def test_static_disclaimer_is_present_and_explains_every_status(self, document):  # AT, AU
		assert '## Interpretation' in document
		assert '- UNSUPPORTED does not mean false.' in document
		assert '- NO_EVIDENCE means no candidate evidence was available for verification' in document
		assert '- CONTRADICTED means the evidence provided directly conflicts with the claim.' in document
		assert '- CONFLICTED means the evidence set contains both supporting and contradicting evidence' in document
		assert 'this report does not decide which source is right' in document

	def test_supporting_and_contradicting_evidence_both_appear_for_a_conflict(self, document):  # AV, item 30
		assert 'Supporting evidence (1):' in document
		assert 'Contradicting evidence (1):' in document
		supporting = document.index('Supporting evidence (1):')
		contradicting = document.index('Contradicting evidence (1):')
		assert supporting < contradicting
		assert 'Conflicts with: [E2]' in document[supporting:contradicting]
		assert 'Conflicts with: [E1]' in document[contradicting:]

	def test_no_evidence_claim_is_rendered_not_silently_dropped(self):  # AW, item 27
		claim_set = _claim_set(('claim-1', 1, CREATED_CLAIM))

		document = MarkdownReportRenderer().render(
			_report(claim_set, [_verification('claim-1', VerificationStatus.NO_EVIDENCE)], [])
		)

		assert '### Claim 1: NO_EVIDENCE' in document
		assert CREATED_CLAIM in document
		assert 'No evidence candidates were available for verification.' in document
		for forbidden in ('FALSE', 'CONTRADICTED:', 'UNSUPPORTED:'):
			# The status word may appear in the summary or the disclaimer, never as this claim's verdict.
			assert forbidden not in document.split('## Interpretation')[0].replace('CONTRADICTED: 0', '').replace(
				'UNSUPPORTED: 0', ''
			)

	def test_unsupported_claim_shows_what_was_checked(self, document):  # item 28
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [_node('evidence-a', LANGUAGE_TEXT, url=DOCS_URL, title='Docs', step_number=1)]
		verifications = [
			_verification(
				'claim-1',
				VerificationStatus.UNSUPPORTED,
				_assessment('evidence-a', EvidenceRelation.INSUFFICIENT, explanation='the docs list no star count.'),
			)
		]

		rendered = MarkdownReportRenderer().render(_report(claim_set, verifications, nodes))

		assert 'Evidence that does not speak to the claim (1):' in rendered
		assert '  - Assessment: the docs list no star count.' in rendered
		assert '- [E1] INSUFFICIENT' in rendered

	def test_empty_report_still_renders_every_section(self):
		document = MarkdownReportRenderer().render(EvidenceGroundedReport(task_id='task-1', task=''))

		for heading in (
			'## Task',
			'## Verification Summary',
			'## Claim Verification',
			'## Evidence Sources',
			'## Interpretation',
		):
			assert heading in document
		assert 'No task prompt was recorded.' in document
		assert 'The answer produced no atomic claims to verify.' in document
		assert 'Claims: 0' in document
		assert 'Evidence coverage: 0.0% (0 of 0 claims)' in document


class TestWholeAnswerVerdict:
	def test_report_models_have_exactly_the_specified_fields(self):  # items 3-7
		assert set(ReportEvidenceSource.model_fields) == {
			'citation_label',
			'evidence_id',
			'url',
			'title',
			'source_host',
			'step_number',
		}
		assert set(ReportClaimEvidence.model_fields) == {
			'evidence_id',
			'relation',
			'explanation',
			'same_source_evidence_ids',
			'duplicate_evidence_ids',
			'conflicting_evidence_ids',
		}
		assert set(ClaimReportSection.model_fields) == {'claim_id', 'order', 'claim_text', 'status', 'evidence'}
		assert set(ReportSummary.model_fields) == {
			'claim_count',
			'evidence_count',
			'unique_source_count',
			'supported_claim_count',
			'partial_claim_count',
			'unsupported_claim_count',
			'contradicted_claim_count',
			'conflicted_claim_count',
			'no_evidence_claim_count',
			'evidence_covered_claim_count',
			'evidence_coverage_rate',
		}
		assert set(EvidenceGroundedReport.model_fields) == {'task_id', 'task', 'summary', 'sources', 'claims'}

	def test_no_model_carries_a_timestamp_or_a_confidence(self):  # items 7, 32
		forbidden = {'created_at', 'generated_at', 'updated_at', 'timestamp', 'confidence', 'probability', 'score'}
		for model in (ReportEvidenceSource, ReportClaimEvidence, ClaimReportSection, ReportSummary, EvidenceGroundedReport):
			assert not set(model.model_fields) & forbidden, model.__name__

	def test_there_is_no_pass_or_fail_field_anywhere(self):  # AX, item 18
		assert not any('overall' in name or 'verdict' in name for name in EvidenceGroundedReport.model_fields)
		assert not any('overall' in name or 'verdict' in name for name in ReportSummary.model_fields)

	def test_the_rendered_document_never_collapses_the_distribution(self):
		claim_set, verifications, nodes = _conflicted_fixture()
		document = MarkdownReportRenderer().render(_report(claim_set, verifications, nodes))

		for forbidden in ('PASS', 'FAIL', 'overall_status', 'answer_is_true', 'VERIFIED', 'NOT VERIFIED'):
			assert forbidden not in document


class TestInjectionSafety:
	"""Spec 24 and 34: scraped and generated text must never become report structure."""

	TASK_ATTACK = '# ATTACK'
	CLAIM_ATTACK = '## Fake Section'
	TITLE_ATTACK = "<script>alert('x')</script>"
	EXPLANATION_ATTACK = '[click](javascript:alert(1))'

	@pytest.fixture
	def hostile(self):
		claim_set = _claim_set(('claim-1', 1, self.CLAIM_ATTACK), task=self.TASK_ATTACK)
		nodes = [_node('evidence-a', HIGH_STAR_TEXT, url='https://github.com/x', title=self.TITLE_ATTACK, step_number=1)]
		verifications = [
			_verification(
				'claim-1',
				VerificationStatus.SUPPORTED,
				_assessment('evidence-a', EvidenceRelation.SUPPORTS, self.EXPLANATION_ATTACK),
			)
		]
		return claim_set, verifications, nodes

	def test_escaping_neutralizes_markup(self):
		assert escape_report_text('# ATTACK') == '\\# ATTACK'
		assert escape_report_text('## Fake Section').startswith('\\##')
		assert escape_report_text("<script>alert('x')</script>") == "\\<script\\>alert\\('x'\\)\\</script\\>"
		assert escape_report_text('[click](javascript:alert(1))') == '\\[click\\]\\(javascript:alert\\(1\\)\\)'
		assert escape_report_text('line one\nline two') == 'line one line two'
		assert escape_report_text('crlf\r\ninjection') == 'crlf injection'
		assert escape_report_text('- quiet list') == '\\- quiet list'

	def test_the_report_keeps_the_raw_text_for_structured_consumers(self, hostile):
		claim_set, verifications, nodes = hostile

		report = _report(claim_set, verifications, nodes)

		assert report.task == self.TASK_ATTACK
		assert report.claims[0].claim_text == self.CLAIM_ATTACK
		assert report.sources[0].title == self.TITLE_ATTACK
		assert report.claims[0].evidence[0].explanation == self.EXPLANATION_ATTACK

	def test_escaping_is_applied_to_every_untrusted_field(self, hostile):  # 34
		"""The renderer must not trust any string it prints, including its own fixtures' defaults."""
		claim_set, verifications, nodes = hostile
		nodes[0].text = 'a\nb'

		def _headings(document: str) -> list[str]:
			return [line for line in document.splitlines() if line.startswith('#')]

		graph = _graph(_claim_set(('claim-1', 1, STARS_CLAIM), task='plain'), verifications, nodes)
		baseline = _headings(
			MarkdownReportRenderer().render(_report(_claim_set(('claim-1', 1, STARS_CLAIM), task='plain'), verifications, nodes))
		)
		hostile_headings = _headings(MarkdownReportRenderer().render(_report(claim_set, verifications, nodes)))

		assert baseline == hostile_headings  # the same structure, whatever the text says

	def test_untrusted_text_cannot_add_headings(self, hostile):  # 34
		claim_set, verifications, nodes = hostile

		document = MarkdownReportRenderer().render(_report(claim_set, verifications, nodes))

		headings = [line for line in document.splitlines() if line.startswith('#')]
		assert headings == [
			'# WebEvidence Verification Report',
			'## Task',
			'## Verification Summary',
			'## Claim Verification',
			'### Claim 1: SUPPORTED',
			'## Evidence Sources',
			'## Interpretation',
		]

	def test_no_raw_html_survives_rendering(self, hostile):  # 34
		claim_set, verifications, nodes = hostile

		document = MarkdownReportRenderer().render(_report(claim_set, verifications, nodes))

		# An angle bracket is only dangerous when markdown can see it unescaped.
		assert re.search(r'(?<!\\)<', document) is None
		assert re.search(r'(?<!\\)>', document) is None
		assert '\\<script\\>' in document

	def test_the_renderer_never_builds_a_link_of_its_own(self, hostile):  # 34, 35
		claim_set, verifications, nodes = hostile

		document = MarkdownReportRenderer().render(_report(claim_set, verifications, nodes))

		# No link in the document at all: a label is always followed by a space, never by a bracket.
		assert '](' not in document
		assert '[http' not in document
		assert '[javascript' not in document

	def test_annotations_are_rendered_as_labels_not_ids(self):  # item 23
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [
			_node('evidence-a', HIGH_STAR_TEXT, url='https://github.com/a', title='GitHub', step_number=1),
			_node('evidence-b', HIGH_STAR_TEXT, url='https://github.com/b', title='GitHub', step_number=2),
			_node('evidence-c', LOW_STAR_TEXT, url='https://other.example.com/c', title='Blog', step_number=3),
		]
		verifications = [
			_verification(
				'claim-1',
				VerificationStatus.CONFLICTED,
				_assessment('evidence-a', EvidenceRelation.SUPPORTS, explanation='the repository header counts 111,799.'),
				_assessment('evidence-b', EvidenceRelation.SUPPORTS, explanation='a second page of the same repository agrees.'),
				_assessment('evidence-c', EvidenceRelation.CONTRADICTS, explanation='an older post reports 30,000.'),
			)
		]

		document = MarkdownReportRenderer().render(_report(claim_set, verifications, nodes))

		assert '  - Same source as: [E2]' in document
		assert '  - Duplicate of: [E1]' in document
		assert '  - Conflicts with: [E3]' in document
		assert '  - Conflicts with: [E1], [E2]' in document
		# Labels, not identifiers: nothing in the document spells out an evidence id.
		for evidence_id in ('evidence-a', 'evidence-b', 'evidence-c'):
			assert evidence_id not in document

	def test_shared_host_alone_never_creates_an_annotation(self):
		"""Phase 6 owns source grouping, so the report reads edges and infers nothing (item 12)."""
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		# Two pages from one host with no SAME_SOURCE edge: a graph the organizer would not produce,
		# but a structurally valid one, and the report must follow the edge, not the host.
		graph = EvidenceGraph(
			task_id='task-1',
			claims=[
				ClaimGraphNode(claim_id='claim-1', text=STARS_CLAIM, order=1, verification_status=VerificationStatus.SUPPORTED)
			],
			evidence=[
				EvidenceGraphNode(
					evidence_id='evidence-a', url='https://github.com/a', title='GitHub', step_number=1, source_host='github.com'
				),
				EvidenceGraphNode(
					evidence_id='evidence-b', url='https://github.com/b', title='GitHub', step_number=2, source_host='github.com'
				),
			],
			claim_evidence_edges=[
				ClaimEvidenceEdge(
					claim_id='claim-1',
					evidence_id='evidence-a',
					relation=EvidenceRelation.SUPPORTS,
					explanation='counts the stars.',
				),
				ClaimEvidenceEdge(
					claim_id='claim-1',
					evidence_id='evidence-b',
					relation=EvidenceRelation.SUPPORTS,
					explanation='counts them again.',
				),
			],
			evidence_evidence_edges=[],
		)

		report = EvidenceReportBuilder().build(claim_set=claim_set, evidence_graph=graph)

		assert all(not item.same_source_evidence_ids for section in report.claims for item in section.evidence)
		document = MarkdownReportRenderer().render(report)
		for heading in ('Same source as', 'Duplicate of', 'Conflicts with'):
			assert heading not in document


class TestRelationGroupingChoice:
	def test_the_model_keeps_edge_order_while_the_document_groups_by_relation(self):
		"""Documented choice for item 22: grouping is presentation, the structured report stays faithful."""
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [
			_node('evidence-deni', LOW_STAR_TEXT, url='https://blog.example.com/a', title='Blog', step_number=1),
			_node('evidence-proof', HIGH_STAR_TEXT, url='https://github.com/b', title='GitHub', step_number=2),
		]
		verifications = [
			_verification(
				'claim-1',
				VerificationStatus.CONFLICTED,
				# Phase 4B ranked the denying page first, and that rank order is what Phase 5 recorded.
				_assessment('evidence-deni', EvidenceRelation.CONTRADICTS, explanation='30,000 falls short.'),
				_assessment('evidence-proof', EvidenceRelation.SUPPORTS, explanation='111,799 clears it.'),
			)
		]

		report = _report(claim_set, verifications, nodes)
		document = MarkdownReportRenderer().render(report)

		assert [item.relation for item in report.claims[0].evidence] == [
			EvidenceRelation.CONTRADICTS,
			EvidenceRelation.SUPPORTS,
		]
		assert document.index('Supporting evidence (1):') < document.index('Contradicting evidence (1):')
		assert document.index('- [E2] SUPPORTS') < document.index('- [E1] CONTRADICTS')

	def test_only_the_groups_a_claim_has_are_printed(self):
		"""An empty group costs the reader nothing, so a clean claim stays short."""
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [_node('evidence-a', HIGH_STAR_TEXT, url=GITHUB_URL, title='GitHub', step_number=1)]
		verifications = [
			_verification('claim-1', VerificationStatus.SUPPORTED, _assessment('evidence-a', EvidenceRelation.SUPPORTS))
		]

		section = MarkdownReportRenderer().render(_report(claim_set, verifications, nodes)).split('## Evidence Sources')[0]

		assert 'Supporting evidence (1):' in section
		for absent in ('Partially supporting evidence', 'Contradicting evidence', 'Evidence that does not speak'):
			assert absent not in section


class TestEvidenceTextStaysOutOfTheReport:
	def test_page_text_never_enters_the_report_or_the_document(self):  # item 26
		# Texts that no claim repeats, so a hit could only come from copying the page body.
		page_text = 'In the header of the repository page a counter sits beside the watch button.'
		other_text = 'A blog post mentions the project once and then talks about keyboards.'
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [
			_node('evidence-a', page_text, url=GITHUB_URL, title='GitHub', step_number=1),
			_node('evidence-b', other_text, url=MIRROR_URL, title='Blog', step_number=2),
		]
		verifications = [
			_verification(
				'claim-1',
				VerificationStatus.SUPPORTED,
				_assessment('evidence-a', EvidenceRelation.SUPPORTS, explanation='the counter reads 111,799.'),
				_assessment('evidence-b', EvidenceRelation.INSUFFICIENT, explanation='the post never counts stars.'),
			)
		]

		report = _report(claim_set, verifications, nodes)
		document = MarkdownReportRenderer().render(report)

		for text in (page_text, other_text):
			assert text not in report.model_dump_json()
			assert text not in document
		# What the reader gets instead: enough to identify and trust the source, not the page body.
		assert 'GitHub' in document and 'https://www.github.com/browser-use/browser-use' in document


class TestUncitedSource:
	def test_a_source_that_no_claim_cites_says_so(self):  # item 23 fallback text
		"""Phase 6 can only produce cited evidence, so this path needs a hand-built graph."""
		claim_set = _claim_set(('claim-1', 1, CREATED_CLAIM))
		graph = EvidenceGraph(
			task_id='task-1',
			claims=[
				ClaimGraphNode(
					claim_id='claim-1', text=CREATED_CLAIM, order=1, verification_status=VerificationStatus.NO_EVIDENCE
				)
			],
			evidence=[
				EvidenceGraphNode(
					evidence_id='evidence-a', url=GITHUB_URL, title='GitHub', step_number=3, source_host='github.com'
				)
			],
		)

		report = EvidenceReportBuilder().build(claim_set=claim_set, evidence_graph=graph)
		document = MarkdownReportRenderer().render(report)

		assert report.summary.evidence_count == 1
		assert report.summary.evidence_covered_claim_count == 0
		assert report.claims[0].evidence == []
		assert '  - Not cited by any claim.' in document
		assert '  - Captured at browser step: 3' in document

	def test_a_javascript_url_is_plain_escaped_text(self):  # 35
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [_node('evidence-a', HIGH_STAR_TEXT, url='javascript:alert(1)', title='Hostile', step_number=1)]
		verifications = [
			_verification('claim-1', VerificationStatus.SUPPORTED, _assessment('evidence-a', EvidenceRelation.SUPPORTS))
		]

		report = _report(claim_set, verifications, nodes)
		document = MarkdownReportRenderer().render(report)

		assert report.sources[0].url == 'javascript:alert(1)'
		assert '  - URL: javascript:alert(1)' in document
		assert '[javascript' not in document
		assert '](javascript' not in document

	def test_url_escaping_keeps_ordinary_urls_usable(self):
		url = 'https://en.wikipedia.org/wiki/Browser_(use)?ref=a&b=2#frag'
		assert escape_report_url(url) == url
		assert escape_report_url('https://x.example/[y](z)') == 'https://x.example/\\[y\\](z)'
		assert escape_report_url('https://x.example/a\nb') == 'https://x.example/a b'
		assert escape_report_url('https://x.example/<img onerror=1>') == 'https://x.example/\\<img onerror=1\\>'


class TestUserScenarios:
	def test_a_conflicted_claim_shows_both_sides_and_picks_neither(self):  # 36
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [
			_node('evidence-e1', 'Browser Use has 111,799 stars', url='https://github.com/e1', title='GitHub', step_number=1),
			_node('evidence-e2', 'Browser Use has 30,000 stars', url='https://example.com/e2', title='Blog', step_number=2),
		]
		verifications = [
			_verification(
				'claim-1',
				VerificationStatus.CONFLICTED,
				_assessment('evidence-e1', EvidenceRelation.SUPPORTS, explanation='111,799 clears the threshold.'),
				_assessment('evidence-e2', EvidenceRelation.CONTRADICTS, explanation='30,000 falls short.'),
			)
		]

		document = MarkdownReportRenderer().render(_report(claim_set, verifications, nodes))
		section = document[document.index('## Claim Verification') : document.index('## Evidence Sources')]

		assert 'CONFLICTED' in section
		assert '- [E1] SUPPORTS' in section
		assert '- [E2] CONTRADICTS' in section
		assert 'github.com' in section and 'example.com' in section
		assert '111,799 clears the threshold.' in section
		assert '30,000 falls short.' in section
		assert 'Conflicts with: [E2]' in section and 'Conflicts with: [E1]' in section
		for arbiter in ('winner', 'correct source', 'more credible', 'resolved'):
			assert arbiter not in section.lower()

	def test_a_no_evidence_claim_is_reported_as_unchecked_not_false(self):  # 37
		claim_set = _claim_set(('claim-1', 1, CREATED_CLAIM))

		document = MarkdownReportRenderer().render(
			_report(claim_set, [_verification('claim-1', VerificationStatus.NO_EVIDENCE)], [])
		)
		section = document[document.index('## Claim Verification') : document.index('## Evidence Sources')]

		assert CREATED_CLAIM in section
		assert 'No evidence candidates were available for verification.' in section
		assert 'FALSE' not in section
		assert 'CONTRADICT' not in section.upper()
		assert 'UNSUPPORTED' not in section


class TestDeterminismAndPurity:
	def test_report_round_trips_through_json(self):  # AI
		claim_set, verifications, nodes = _conflicted_fixture()
		report = _report(claim_set, verifications, nodes)

		parsed = EvidenceGroundedReport.model_validate_json(report.model_dump_json())

		assert parsed == report
		assert parsed.claims[0].status is VerificationStatus.CONFLICTED
		assert parsed.claims[0].evidence[0].relation is EvidenceRelation.SUPPORTS

	def test_inputs_are_not_mutated(self):  # AJ
		claim_set, verifications, nodes = _conflicted_fixture()
		graph = _graph(claim_set, verifications, nodes)
		claim_set_before, graph_before = deepcopy(claim_set), deepcopy(graph)

		EvidenceReportBuilder().build(claim_set=claim_set, evidence_graph=graph)

		assert claim_set == claim_set_before
		assert graph == graph_before

	def test_rebuilding_the_same_input_is_identical(self):  # AK
		claim_set, verifications, nodes = _conflicted_fixture()
		graph = _graph(claim_set, verifications, nodes)
		builder = EvidenceReportBuilder()

		first = builder.build(claim_set=claim_set, evidence_graph=graph)
		second = builder.build(claim_set=claim_set, evidence_graph=graph)

		assert first.model_dump() == second.model_dump()
		assert first.model_dump_json() == second.model_dump_json()

	def test_rendering_the_same_report_is_identical(self):  # AL
		claim_set, verifications, nodes = _conflicted_fixture()
		report = _report(claim_set, verifications, nodes)
		renderer = MarkdownReportRenderer()

		assert renderer.render(report) == renderer.render(report)
		assert renderer.render(report) == renderer.render(deepcopy(report))

	def test_nothing_in_the_output_depends_on_time_or_randomness(self):
		"""Spec 32: no clock, no id generation, no sampling anywhere in this stage."""
		import ast

		import browser_use.evidence.reporting as reporting_module

		tree = ast.parse(Path(reporting_module.__file__).read_text(encoding='utf-8'))
		imported: set[str] = set()
		for node in ast.walk(tree):
			if isinstance(node, ast.Import):
				imported.update(alias.name.partition('.')[0] for alias in node.names)
			elif isinstance(node, ast.ImportFrom) and node.module:
				imported.add(node.module)

		assert imported == {
			're',
			'collections.abc',
			'pydantic',
			'browser_use.evidence.claims',
			'browser_use.evidence.organization',
			'browser_use.evidence.verification',
		}, sorted(imported)
		assert not imported & {'datetime', 'time', 'uuid', 'random', 'secrets'}
