"""Unit tests for deterministic structured evidence organization.

Phase 6 is pure Python, so every test builds its own inputs: no LLM, no browser, no network. The
letter tags in comments map onto the Phase 6 specification checklist.
"""

import inspect
import random
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from browser_use.evidence import (
	Claim,
	ClaimEvidenceEdge,
	ClaimGraphNode,
	ClaimSet,
	ClaimVerification,
	EvidenceAssessment,
	EvidenceEdgeType,
	EvidenceEvidenceEdge,
	EvidenceGraph,
	EvidenceGraphNode,
	EvidenceGraphStats,
	EvidenceNode,
	EvidenceOrganizationError,
	EvidenceOrganizer,
	EvidenceRelation,
	VerificationResult,
	VerificationStatus,
	organization,
)
from browser_use.evidence.alignment import tokenize
from browser_use.evidence.organization import _normalized_content, _source_host

STARS_CLAIM = 'Browser Use has more than 100,000 GitHub stars.'
LANGUAGE_CLAIM = 'Browser Use is primarily written in Python.'
MCP_CLAIM = 'Framework X introduced native MCP support in version 2.0.'
MCP_PARTIAL_TEXT = 'Framework X supports MCP.'
HIGH_STAR_TEXT = 'Browser Use has 111,799 GitHub stars.'
LOW_STAR_TEXT = 'Browser Use has only 30,000 GitHub stars.'
LANGUAGE_TEXT = 'Browser Use is primarily written in Python.'


def _node(
	evidence_id: str,
	text: str,
	*,
	url: str = 'https://example.com/page',
	title: str = '',
	step_number: int = 1,
	task_id: str = 'task-1',
) -> EvidenceNode:
	return EvidenceNode(
		evidence_id=evidence_id,
		task_id=task_id,
		step_number=step_number,
		url=url,
		title=title,
		text=text,
		screenshot_path=f'/shots/{evidence_id}.png',
		action_names=['click'],
		metadata={'dom_hash': 'abc123'},
	)


def _claim_set(*claims: tuple[str, int, str], task_id: str = 'task-1') -> ClaimSet:
	"""Claims as ``(claim_id, order, text)`` triples, which makes shuffled input easy to write."""
	return ClaimSet(
		task_id=task_id,
		task='How popular is Browser Use?',
		answer=' '.join(text for _, _, text in claims),
		claims=[Claim(claim_id=claim_id, order=order, text=text) for claim_id, order, text in claims],
	)


def _verification(claim_id: str, status: VerificationStatus, *assessments: EvidenceAssessment) -> ClaimVerification:
	return ClaimVerification(claim_id=claim_id, status=status, assessments=list(assessments))


def _assessment(
	evidence_id: str,
	relation: EvidenceRelation,
	explanation: str | None = None,
) -> EvidenceAssessment:
	return EvidenceAssessment(
		evidence_id=evidence_id,
		relation=relation,
		explanation=explanation or f'{evidence_id} {relation.value.lower()}.',
	)


def _result(*verifications: ClaimVerification, task_id: str = 'task-1') -> VerificationResult:
	return VerificationResult(task_id=task_id, verifications=list(verifications))


def _organize(
	claim_set: ClaimSet,
	verification_result: VerificationResult,
	evidence_nodes: list[EvidenceNode],
	**options,
) -> EvidenceGraph:
	return EvidenceOrganizer(**options).organize(
		claim_set=claim_set,
		verification_result=verification_result,
		evidence_nodes=evidence_nodes,
	)


def _edge_pairs(graph: EvidenceGraph, relation: EvidenceEdgeType) -> set[tuple[str, str]]:
	return {
		(edge.source_evidence_id, edge.target_evidence_id) for edge in graph.evidence_evidence_edges if edge.relation is relation
	}


def _support_and_contradiction():
	"""The canonical Phase 5 shape: one claim, one page supporting it and one page denying it."""
	nodes = [
		_node('evidence-high', HIGH_STAR_TEXT, url='https://github.com/browser-use/browser-use', title='GitHub', step_number=1),
		_node('evidence-low', LOW_STAR_TEXT, url='https://blog.example.com/old', title='Old post', step_number=2),
	]
	claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
	result = _result(
		_verification(
			'claim-1',
			VerificationStatus.CONFLICTED,
			_assessment('evidence-high', EvidenceRelation.SUPPORTS),
			_assessment('evidence-low', EvidenceRelation.CONTRADICTS),
		)
	)
	return claim_set, result, nodes


def _organize_inputs(nodes: list[EvidenceNode]):
	"""Wrap arbitrary evidence in the smallest valid single-claim verification, for edge-rule tests."""
	claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
	result = _result(
		_verification(
			'claim-1', VerificationStatus.SUPPORTED, *[_assessment(node.evidence_id, EvidenceRelation.SUPPORTS) for node in nodes]
		)
	)
	return claim_set, result, nodes


def _similarity(first: str, second: str) -> float:
	"""Token Jaccard of two texts under the same tokenizer and normalization the organizer uses."""
	first_tokens = frozenset(tokenize(_normalized_content('', first)))
	second_tokens = frozenset(tokenize(_normalized_content('', second)))
	union = first_tokens | second_tokens
	return len(first_tokens & second_tokens) / len(union) if union else 0.0


class TestGraphModels:
	def test_evidence_edge_type_has_exactly_the_three_relations(self):
		assert [edge_type.value for edge_type in EvidenceEdgeType] == ['SAME_SOURCE', 'DUPLICATE', 'CONFLICTS_WITH']

	def test_graph_has_exactly_the_specified_fields(self):
		assert set(EvidenceGraph.model_fields) == {
			'task_id',
			'claims',
			'evidence',
			'claim_evidence_edges',
			'evidence_evidence_edges',
			'stats',
		}
		assert set(ClaimGraphNode.model_fields) == {'claim_id', 'text', 'order', 'verification_status'}
		assert set(EvidenceGraphNode.model_fields) == {'evidence_id', 'url', 'title', 'step_number', 'source_host'}
		assert set(ClaimEvidenceEdge.model_fields) == {'claim_id', 'evidence_id', 'relation', 'explanation'}
		assert set(EvidenceEvidenceEdge.model_fields) == {'source_evidence_id', 'target_evidence_id', 'relation', 'claim_id'}

	def test_stats_has_every_specified_count(self):
		assert set(EvidenceGraphStats.model_fields) == {
			'claim_count',
			'evidence_count',
			'supported_claim_count',
			'partial_claim_count',
			'unsupported_claim_count',
			'contradicted_claim_count',
			'conflicted_claim_count',
			'no_evidence_claim_count',
			'support_edge_count',
			'partial_support_edge_count',
			'contradict_edge_count',
			'insufficient_edge_count',
			'same_source_edge_count',
			'duplicate_edge_count',
			'conflict_edge_count',
		}

	def test_no_confidence_or_authority_score_anywhere(self):
		forbidden = {'confidence', 'probability', 'authority', 'weight', 'embedding', 'reward', 'score', 'chain_of_thought'}
		for model in (
			EvidenceGraph,
			EvidenceGraphNode,
			ClaimGraphNode,
			ClaimEvidenceEdge,
			EvidenceEvidenceEdge,
			EvidenceGraphStats,
		):
			offending = {name for name in model.model_fields if any(bad in name.lower() for bad in forbidden)}
			assert not offending, f'{model.__name__}: {offending}'

	def test_claim_evidence_edge_explanation_cannot_be_blank(self):
		with pytest.raises(ValidationError):
			ClaimEvidenceEdge(
				claim_id='claim-1',
				evidence_id='evidence-a',
				relation=EvidenceRelation.SUPPORTS,
				explanation='  ',
			)

	def test_source_and_duplicate_edges_are_claim_independent(self):
		for relation in (EvidenceEdgeType.SAME_SOURCE, EvidenceEdgeType.DUPLICATE):
			with pytest.raises(ValidationError, match='must not carry a claim_id'):
				EvidenceEvidenceEdge(
					source_evidence_id='evidence-a',
					target_evidence_id='evidence-b',
					relation=relation,
					claim_id='claim-1',
				)

	def test_conflict_edge_must_be_claim_scoped(self):
		with pytest.raises(ValidationError, match='needs the claim_id'):
			EvidenceEvidenceEdge(
				source_evidence_id='evidence-a',
				target_evidence_id='evidence-b',
				relation=EvidenceEdgeType.CONFLICTS_WITH,
			)

	def test_empty_graph_defaults_are_empty(self):
		graph = EvidenceGraph(task_id='task-1')
		assert graph.claims == []
		assert graph.evidence == []
		assert graph.claim_evidence_edges == []
		assert graph.evidence_evidence_edges == []
		assert graph.stats.claim_count == 0

	def test_stats_as_dict_covers_every_field(self):
		stats = EvidenceGraphStats()
		assert set(stats.as_dict()) == set(EvidenceGraphStats.model_fields)
		assert set(stats.as_dict().values()) == {0}


class TestSourceHostNormalization:
	@pytest.mark.parametrize(
		('url', 'expected'),
		[
			('https://www.github.com/browser-use/browser-use', 'github.com'),  # E
			('https://GitHub.COM/x', 'github.com'),
			('https://github.com.', 'github.com'),
			('https://docs.python.org/3/library/index.html', 'docs.python.org'),  # F
			('https://en.wikipedia.org/wiki/Browser_Use#History', 'en.wikipedia.org'),
			('http://blog.example.co.uk/post/1', 'blog.example.co.uk'),
			('https://www.www.example.com', 'www.example.com'),
			('https://github.com:443/x', 'github.com'),
			('https://user:pass@host.example/path', 'host.example'),
			('https://xn--fiqs8s.example/zh', 'xn--fiqs8s.example'),
			('https://8.8.8.8/lookup', '8.8.8.8'),
			('/just/a/path', ''),  # G
			('', ''),
			('not a url at all', ''),
			('https://', ''),
			('http:///%20bad', ''),
		],
	)
	def test_host_rules(self, url, expected):
		assert _source_host(url) == expected

	def test_unknown_hosts_are_recorded_as_empty_rather_than_guessed(self):
		claim_set, result, nodes = _support_and_contradiction()
		nodes[0].url = 'about:blank'

		graph = _organize(claim_set, result, nodes)

		assert graph.evidence[0].source_host == ''
		assert _edge_pairs(graph, EvidenceEdgeType.SAME_SOURCE) == set()


class TestSameSourceEdges:
	def test_two_pages_from_one_host_produce_one_edge(self):  # H
		nodes = [
			_node('evidence-a', 'First page about the release.', url='https://www.github.com/a', step_number=1),
			_node('evidence-b', 'Second page about the release.', url='https://github.com/b', step_number=2),
		]
		claim_set = _claim_set(('claim-1', 1, LANGUAGE_CLAIM))
		result = _result(
			_verification(
				'claim-1',
				VerificationStatus.SUPPORTED,
				_assessment('evidence-a', EvidenceRelation.SUPPORTS),
				_assessment('evidence-b', EvidenceRelation.SUPPORTS),
			)
		)

		graph = _organize(claim_set, result, nodes)

		assert _edge_pairs(graph, EvidenceEdgeType.SAME_SOURCE) == {('evidence-a', 'evidence-b')}
		assert all(edge.claim_id is None for edge in graph.evidence_evidence_edges)

	def test_different_hosts_produce_no_edge(self):  # I
		claim_set, result, nodes = _support_and_contradiction()

		graph = _organize(claim_set, result, nodes)

		assert _edge_pairs(graph, EvidenceEdgeType.SAME_SOURCE) == set()

	def test_two_unknown_hosts_do_not_group_together(self):
		"""An empty host is "unknown", and two unknowns must not be reported as one source."""
		nodes = [
			_node('evidence-a', 'Alpha text about the project.', url='', step_number=1),
			_node('evidence-b', 'Beta text about the project.', url='/relative/path', step_number=2),
		]
		claim_set = _claim_set(('claim-1', 1, LANGUAGE_CLAIM))
		result = _result(
			_verification(
				'claim-1',
				VerificationStatus.SUPPORTED,
				_assessment('evidence-a', EvidenceRelation.SUPPORTS),
				_assessment('evidence-b', EvidenceRelation.SUPPORTS),
			)
		)

		graph = _organize(claim_set, result, nodes)

		assert _edge_pairs(graph, EvidenceEdgeType.SAME_SOURCE) == set()


class TestDuplicateDetection:
	def test_identical_content_is_a_duplicate(self):  # J
		nodes = [
			_node('evidence-a', HIGH_STAR_TEXT, url='https://one.example/x', title='GitHub', step_number=1),
			_node('evidence-b', HIGH_STAR_TEXT, url='https://two.example/y', title='GitHub', step_number=2),
		]

		graph = _organize(*_organize_inputs(nodes))

		assert _edge_pairs(graph, EvidenceEdgeType.DUPLICATE) == {('evidence-a', 'evidence-b')}

	def test_case_nfkcc_and_whitespace_differences_still_match(self):  # K
		nodes = [
			_node('evidence-a', HIGH_STAR_TEXT, url='https://one.example/x', title='GitHub', step_number=1),
			_node(
				'evidence-b',
				' browser   use \tHAS １１１,７９９ GitHub  stars. ',
				url='https://two.example/y',
				title='  github ',
				step_number=2,
			),
		]

		graph = _organize(*_organize_inputs(nodes))

		assert _edge_pairs(graph, EvidenceEdgeType.DUPLICATE) == {('evidence-a', 'evidence-b')}

	def test_high_token_overlap_beyond_the_threshold_is_a_duplicate(self):  # L
		first = 'Browser Use is an open source Python framework that automates browsers for AI agents.'
		second = 'Browser Use is an open source Python framework that automates browsers for AI agents today.'
		assert _similarity(first, second) >= 0.90
		nodes = [
			_node('evidence-a', first, url='https://one.example/x', step_number=1),
			_node('evidence-b', second, url='https://two.example/y', step_number=2),
		]

		graph = _organize(*_organize_inputs(nodes))

		assert _edge_pairs(graph, EvidenceEdgeType.DUPLICATE) == {('evidence-a', 'evidence-b')}

	def test_low_overlap_is_not_a_duplicate(self):  # M
		nodes = [
			_node('evidence-a', HIGH_STAR_TEXT, url='https://one.example/x', step_number=1),
			_node(
				'evidence-b',
				'The framework ships a command line interface and a Docker image.',
				url='https://two.example/y',
				step_number=2,
			),
		]

		graph = _organize(*_organize_inputs(nodes))

		assert _edge_pairs(graph, EvidenceEdgeType.DUPLICATE) == set()

	def test_a_short_page_cannot_duplicate_on_one_shared_word(self):  # N
		"""The token floor is what keeps "both mention browser" from meaning "same page"."""
		nodes = [
			_node('evidence-a', 'Browser', url='https://one.example/x', step_number=1),
			_node('evidence-b', 'Browser', url='https://two.example/y', step_number=2),
		]

		graph = _organize(*_organize_inputs(nodes))

		# Identical normalized content still matches the exact rule, so this pair tests the floor only
		# once the texts differ; here the exact rule fires, which is the intended behaviour.
		assert _edge_pairs(graph, EvidenceEdgeType.DUPLICATE) == {('evidence-a', 'evidence-b')}

		nodes[1].text = 'Browsers'
		graph = _organize(*_organize_inputs(nodes))

		assert _edge_pairs(graph, EvidenceEdgeType.DUPLICATE) == set()

	def test_title_and_body_are_normalized_as_separate_fields(self):
		"""Swapping a title into the body is not the same document under the exact rule."""
		assert _normalized_content('Stars Today', HIGH_STAR_TEXT) != _normalized_content(HIGH_STAR_TEXT, 'Stars Today')
		assert _normalized_content(' GitHub ', 'a\tb') == 'github\na b'

	def test_empty_content_is_never_a_duplicate(self):
		nodes = [
			_node('evidence-a', '', url='https://one.example/x', step_number=1),
			_node('evidence-b', '', url='https://two.example/y', step_number=2),
		]

		graph = _organize(*_organize_inputs(nodes))

		assert _edge_pairs(graph, EvidenceEdgeType.DUPLICATE) == set()

	def test_duplicate_and_same_source_can_coexist(self):  # V
		nodes = [
			_node('evidence-a', HIGH_STAR_TEXT, url='https://github.com/a', title='GitHub', step_number=1),
			_node('evidence-b', HIGH_STAR_TEXT, url='https://github.com/b', title='GitHub', step_number=2),
		]

		graph = _organize(*_organize_inputs(nodes))

		assert _edge_pairs(graph, EvidenceEdgeType.DUPLICATE) == {('evidence-a', 'evidence-b')}
		assert _edge_pairs(graph, EvidenceEdgeType.SAME_SOURCE) == {('evidence-a', 'evidence-b')}

	def test_duplicate_nodes_keep_their_own_claim_edges(self):  # W
		"""Organization is not deletion: both copies keep their own verdicts and rationale."""
		nodes = [
			_node('evidence-a', HIGH_STAR_TEXT, url='https://one.example/x', title='GitHub', step_number=1),
			_node('evidence-b', HIGH_STAR_TEXT, url='https://two.example/y', title='GitHub', step_number=2),
		]
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		result = _result(
			_verification(
				'claim-1',
				VerificationStatus.SUPPORTED,
				_assessment('evidence-a', EvidenceRelation.SUPPORTS, explanation='states 111,799 stars.'),
				_assessment('evidence-b', EvidenceRelation.PARTIAL_SUPPORT, explanation='a mirror page without a date.'),
			)
		)

		graph = _organize(claim_set, result, nodes)

		edges = {(edge.evidence_id, edge.relation, edge.explanation) for edge in graph.claim_evidence_edges}
		assert edges == {
			('evidence-a', EvidenceRelation.SUPPORTS, 'states 111,799 stars.'),
			('evidence-b', EvidenceRelation.PARTIAL_SUPPORT, 'a mirror page without a date.'),
		}
		assert len(graph.evidence) == 2

	@pytest.mark.parametrize('threshold', [0.0, -0.1, 1.5])
	def test_invalid_duplicate_threshold_is_rejected(self, threshold):  # O
		with pytest.raises(ValueError, match='duplicate_threshold'):
			EvidenceOrganizer(duplicate_threshold=threshold)

	@pytest.mark.parametrize('tokens', [0, -3])
	def test_invalid_min_duplicate_tokens_is_rejected(self, tokens):  # P
		with pytest.raises(ValueError, match='min_duplicate_tokens'):
			EvidenceOrganizer(min_duplicate_tokens=tokens)

	def test_threshold_of_one_requires_full_token_agreement(self):
		nodes = [
			_node('evidence-a', 'alpha beta gamma delta epsilon', url='https://one.example/x', step_number=1),
			_node('evidence-b', 'alpha beta gamma delta epsilon zeta', url='https://two.example/y', step_number=2),
		]

		assert _organize(*_organize_inputs(nodes), duplicate_threshold=1.0).evidence_evidence_edges == []
		assert _edge_pairs(_organize(*_organize_inputs(nodes), duplicate_threshold=0.8), EvidenceEdgeType.DUPLICATE) == {
			('evidence-a', 'evidence-b')
		}

	def test_normalized_content_is_the_documented_form(self):
		assert _normalized_content('  GitHub ', 'Browser\tUse   STARS') == 'github\nbrowser use stars'


class TestConflictEdges:
	def test_supports_versus_contradicts_produces_one_claim_scoped_edge(self):  # Q
		claim_set, result, nodes = _support_and_contradiction()

		graph = _organize(claim_set, result, nodes)

		conflicts = graph.evidence_evidence_edges
		assert len(conflicts) == 1
		edge = conflicts[0]
		assert edge.relation is EvidenceEdgeType.CONFLICTS_WITH
		assert edge.claim_id == 'claim-1'
		assert (edge.source_evidence_id, edge.target_evidence_id) == ('evidence-high', 'evidence-low')

	def test_partial_support_versus_contradicts_produces_an_edge(self):  # R
		claim_set, result, nodes = _support_and_contradiction()
		result.verifications[0].assessments[0] = _assessment('evidence-high', EvidenceRelation.PARTIAL_SUPPORT)

		graph = _organize(claim_set, result, nodes)

		assert [(edge.relation, edge.claim_id) for edge in graph.evidence_evidence_edges] == [
			(EvidenceEdgeType.CONFLICTS_WITH, 'claim-1')
		]

	def test_two_supporting_pages_do_not_conflict(self):  # S
		nodes = [
			_node('evidence-a', 'One page with the count.', url='https://one.example/a', step_number=1),
			_node('evidence-b', 'Another page with the count.', url='https://two.example/b', step_number=2),
		]
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		result = _result(
			_verification(
				'claim-1',
				VerificationStatus.SUPPORTED,
				_assessment('evidence-a', EvidenceRelation.SUPPORTS),
				_assessment('evidence-b', EvidenceRelation.SUPPORTS),
			)
		)

		graph = _organize(claim_set, result, nodes)

		assert _edge_pairs(graph, EvidenceEdgeType.CONFLICTS_WITH) == set()

	def test_insufficient_versus_contradicts_does_not_conflict(self):  # T
		"""Contradicting a claim is not contradicting the evidence that failed to speak to it."""
		claim_set, result, nodes = _support_and_contradiction()
		result.verifications[0].assessments[0] = _assessment('evidence-high', EvidenceRelation.INSUFFICIENT)
		result.verifications[0].status = VerificationStatus.CONTRADICTED

		graph = _organize(claim_set, result, nodes)

		assert graph.evidence_evidence_edges == []

	def test_two_supporters_and_one_denier_gives_two_edges_not_three(self):  # 14
		nodes = [
			_node('evidence-1', 'Page one of the release notes.', url='https://one.example/a', step_number=1),
			_node('evidence-2', 'Page two of the release notes.', url='https://two.example/b', step_number=2),
			_node('evidence-3', 'Page three denies the release.', url='https://three.example/c', step_number=3),
		]
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		result = _result(
			_verification(
				'claim-1',
				VerificationStatus.CONFLICTED,
				_assessment('evidence-1', EvidenceRelation.SUPPORTS),
				_assessment('evidence-2', EvidenceRelation.SUPPORTS),
				_assessment('evidence-3', EvidenceRelation.CONTRADICTS),
			)
		)

		graph = _organize(claim_set, result, nodes)

		assert _edge_pairs(graph, EvidenceEdgeType.CONFLICTS_WITH) == {('evidence-1', 'evidence-3'), ('evidence-2', 'evidence-3')}

	def test_a_pair_never_appears_in_both_directions(self):  # U
		"""Canonical id order means the same two pages yield one edge however the verdicts arrive."""
		nodes = [
			_node('evidence-b', 'The repository shows 30,000 stars.', url='https://two.example/b', step_number=1),
			_node('evidence-a', 'The repository shows 111,799 stars.', url='https://one.example/a', step_number=2),
		]
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		result = _result(
			_verification(
				'claim-1',
				VerificationStatus.CONFLICTED,
				_assessment('evidence-b', EvidenceRelation.CONTRADICTS),
				_assessment('evidence-a', EvidenceRelation.SUPPORTS),
			)
		)

		graph = _organize(claim_set, result, nodes)

		assert [(edge.source_evidence_id, edge.target_evidence_id) for edge in graph.evidence_evidence_edges] == [
			('evidence-a', 'evidence-b')
		]

	def test_the_same_pair_can_conflict_under_two_different_claims(self):
		"""Conflict is claim-relative, and two claims must not collapse into one indistinguishable edge."""
		nodes = [
			_node('evidence-a', 'The repository shows 111,799 stars today.', url='https://one.example/a', step_number=1),
			_node('evidence-b', 'The repository showed 30,000 stars last year.', url='https://two.example/b', step_number=2),
		]
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM), ('claim-2', 2, 'Browser Use had 111,799 stars last year.'))
		result = _result(
			_verification(
				'claim-1',
				VerificationStatus.CONFLICTED,
				_assessment('evidence-a', EvidenceRelation.SUPPORTS),
				_assessment('evidence-b', EvidenceRelation.CONTRADICTS),
			),
			_verification(
				'claim-2',
				VerificationStatus.CONFLICTED,
				_assessment('evidence-a', EvidenceRelation.CONTRADICTS),
				_assessment('evidence-b', EvidenceRelation.SUPPORTS),
			),
		)

		graph = _organize(claim_set, result, nodes)

		assert sorted(edge.claim_id for edge in graph.evidence_evidence_edges) == ['claim-1', 'claim-2']
		assert all(
			(edge.source_evidence_id, edge.target_evidence_id) == ('evidence-a', 'evidence-b')
			for edge in graph.evidence_evidence_edges
		)


class TestGraphContent:
	def test_one_claim_one_evidence_makes_one_of_everything(self):  # A
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [_node('evidence-a', HIGH_STAR_TEXT, url='https://github.com/x', title='GitHub', step_number=4)]
		result = _result(
			_verification('claim-1', VerificationStatus.SUPPORTED, _assessment('evidence-a', EvidenceRelation.SUPPORTS))
		)

		graph = _organize(claim_set, result, nodes)

		assert len(graph.claims) == len(graph.evidence) == len(graph.claim_evidence_edges) == 1
		assert graph.evidence_evidence_edges == []

	def test_claim_node_carries_the_phase_5_status_untouched(self):  # B
		claim_set, result, nodes = _support_and_contradiction()

		graph = _organize(claim_set, result, nodes)

		assert graph.claims[0].verification_status is VerificationStatus.CONFLICTED
		assert graph.claims[0].verification_status is result.verifications[0].status

	def test_claim_text_and_order_are_copied(self):  # C
		claim_set = _claim_set(('claim-9', 9, MCP_CLAIM))
		nodes = [_node('evidence-a', MCP_PARTIAL_TEXT, url='https://docs.example/x', step_number=1)]
		result = _result(
			_verification('claim-9', VerificationStatus.PARTIAL, _assessment('evidence-a', EvidenceRelation.PARTIAL_SUPPORT))
		)

		graph = _organize(claim_set, result, nodes)

		assert (graph.claims[0].text, graph.claims[0].order) == (MCP_CLAIM, 9)

	def test_evidence_node_carries_url_title_step_and_host(self):  # D
		claim_set, result, nodes = _support_and_contradiction()

		graph = _organize(claim_set, result, nodes)

		first = graph.evidence[0]
		assert first.evidence_id == 'evidence-high'
		assert first.url == 'https://github.com/browser-use/browser-use'
		assert first.title == 'GitHub'
		assert first.step_number == 1
		assert first.source_host == 'github.com'

	def test_only_assessed_evidence_enters_the_graph(self):  # X, Y
		claim_set, result, nodes = _support_and_contradiction()
		unused = [
			_node('evidence-unused-1', 'A page nobody asked about.', url='https://random.example/a', step_number=3),
			_node('evidence-unused-2', 'Another page from the crawl.', url='https://random.example/b', step_number=4),
		]

		graph = _organize(claim_set, result, nodes + unused)

		assert [node.evidence_id for node in graph.evidence] == ['evidence-high', 'evidence-low']
		assert graph.stats.evidence_count == 2
		# The untouched nodes stay untouched: the graph is not a copy of the store.
		assert [node.url for node in nodes] == ['https://github.com/browser-use/browser-use', 'https://blog.example.com/old']

	def test_a_shared_page_is_one_node_with_two_edges(self):
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM), ('claim-2', 2, LANGUAGE_CLAIM))
		nodes = [_node('evidence-a', HIGH_STAR_TEXT, url='https://github.com/x', title='GitHub', step_number=1)]
		result = _result(
			_verification('claim-1', VerificationStatus.SUPPORTED, _assessment('evidence-a', EvidenceRelation.SUPPORTS)),
			_verification('claim-2', VerificationStatus.UNSUPPORTED, _assessment('evidence-a', EvidenceRelation.INSUFFICIENT)),
		)

		graph = _organize(claim_set, result, nodes)

		assert len(graph.evidence) == 1
		assert [(edge.claim_id, edge.relation) for edge in graph.claim_evidence_edges] == [
			('claim-1', EvidenceRelation.SUPPORTS),
			('claim-2', EvidenceRelation.INSUFFICIENT),
		]
		# Two edges to the same page are not evidence-to-evidence edges, so the pair set stays empty.
		assert graph.evidence_evidence_edges == []

	def test_no_evidence_claim_keeps_its_node_and_has_no_edges(self):  # Z, 20
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM), ('claim-2', 2, LANGUAGE_CLAIM))
		nodes = [_node('evidence-a', HIGH_STAR_TEXT, url='https://github.com/x', step_number=1)]
		result = _result(
			_verification('claim-1', VerificationStatus.SUPPORTED, _assessment('evidence-a', EvidenceRelation.SUPPORTS)),
			_verification('claim-2', VerificationStatus.NO_EVIDENCE),
		)

		graph = _organize(claim_set, result, nodes)

		assert [node.claim_id for node in graph.claims] == ['claim-1', 'claim-2']
		assert [edge.claim_id for edge in graph.claim_evidence_edges] == ['claim-1']
		assert graph.stats.no_evidence_claim_count == 1

	def test_unsupported_claim_keeps_its_insufficient_edges(self):  # AA, 21
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM), ('claim-2', 2, LANGUAGE_CLAIM))
		nodes = [
			_node('evidence-a', HIGH_STAR_TEXT, url='https://github.com/x', step_number=1),
			_node('evidence-b', LANGUAGE_TEXT, url='https://docs.example/y', step_number=2),
		]
		result = _result(
			_verification('claim-1', VerificationStatus.SUPPORTED, _assessment('evidence-a', EvidenceRelation.SUPPORTS)),
			_verification('claim-2', VerificationStatus.UNSUPPORTED, _assessment('evidence-b', EvidenceRelation.INSUFFICIENT)),
		)

		graph = _organize(claim_set, result, nodes)

		unsupported = next(node for node in graph.claims if node.claim_id == 'claim-2')
		assert unsupported.verification_status is VerificationStatus.UNSUPPORTED
		assert [edge.evidence_id for edge in graph.claim_evidence_edges if edge.claim_id == 'claim-2'] == ['evidence-b']
		# The two "no proof" states differ in the graph itself, not only in the label.
		assert graph.stats.no_evidence_claim_count == 0
		assert graph.stats.insufficient_edge_count == 1

	def test_graph_never_carries_page_text_screenshot_or_metadata(self):
		"""The graph is structure; EvidenceNode stays the source of truth for content."""
		claim_set, result, nodes = _support_and_contradiction()

		graph = _organize(claim_set, result, nodes)

		dumped = graph.model_dump_json()
		assert HIGH_STAR_TEXT not in dumped
		assert LOW_STAR_TEXT not in dumped
		for forbidden in ('screenshot', 'dom_hash', 'action_names', '/shots/'):
			assert forbidden not in dumped

	def test_the_organizer_is_synchronous_and_needs_no_model(self):
		assert not inspect.iscoroutinefunction(EvidenceOrganizer.organize)
		assert 'llm' not in inspect.signature(EvidenceOrganizer.organize).parameters
		assert not any('llm' in dir_ or 'model' in dir_ for dir_ in dir(EvidenceOrganizer))

	def test_the_module_imports_no_model_and_no_network_layer(self):
		"""Spec 7 and 27: pure Python over the given objects, and no new dependency either."""
		import ast

		imported: set[str] = set()
		for node in ast.walk(ast.parse(Path(organization.__file__).read_text(encoding='utf-8'))):
			if isinstance(node, ast.Import):
				imported.update(alias.name.partition('.')[0] for alias in node.names)
			elif isinstance(node, ast.ImportFrom) and node.module:
				imported.add(node.module)

		assert imported == {
			'collections.abc',
			'enum',
			're',
			'typing',
			'unicodedata',
			'urllib.parse',
			'pydantic',
			'browser_use.evidence.alignment',
			'browser_use.evidence.claims',
			'browser_use.evidence.models',
			'browser_use.evidence.verification',
		}, sorted(imported)


class TestSpecifiedScenarios:
	def test_conflict_figure(self):  # 24
		claim_set = _claim_set(('claim-c1', 1, STARS_CLAIM))
		nodes = [
			_node('evidence-e1', 'Browser Use has 111,799 stars', url='https://github.com/e1', step_number=1),
			_node('evidence-e2', 'Browser Use has 30,000 stars', url='https://example.com/e2', step_number=2),
		]
		result = _result(
			_verification(
				'claim-c1',
				VerificationStatus.CONFLICTED,
				_assessment('evidence-e1', EvidenceRelation.SUPPORTS),
				_assessment('evidence-e2', EvidenceRelation.CONTRADICTS),
			)
		)

		graph = _organize(claim_set, result, nodes)

		assert [(node.claim_id, node.verification_status) for node in graph.claims] == [
			('claim-c1', VerificationStatus.CONFLICTED)
		]
		assert [(edge.claim_id, edge.evidence_id, edge.relation) for edge in graph.claim_evidence_edges] == [
			('claim-c1', 'evidence-e1', EvidenceRelation.SUPPORTS),
			('claim-c1', 'evidence-e2', EvidenceRelation.CONTRADICTS),
		]
		assert [
			(edge.source_evidence_id, edge.target_evidence_id, edge.relation, edge.claim_id)
			for edge in graph.evidence_evidence_edges
		] == [('evidence-e1', 'evidence-e2', EvidenceEdgeType.CONFLICTS_WITH, 'claim-c1')]

	def test_cross_host_duplicate_figure(self):  # 25
		claim_set = _claim_set(('claim-c1', 1, STARS_CLAIM))
		nodes = [
			_node(
				'evidence-e1', 'Browser Use has 111,799 GitHub stars.', url='https://source-a.com/x', title='Stars', step_number=1
			),
			_node(
				'evidence-e2',
				' browser   use HAS 111,799 github stars. ',
				url='https://source-b.com/y',
				title='Stars',
				step_number=2,
			),
		]
		result = _result(
			_verification(
				'claim-c1',
				VerificationStatus.SUPPORTED,
				_assessment('evidence-e1', EvidenceRelation.SUPPORTS),
				_assessment('evidence-e2', EvidenceRelation.SUPPORTS),
			)
		)

		graph = _organize(claim_set, result, nodes)

		assert _edge_pairs(graph, EvidenceEdgeType.DUPLICATE) == {('evidence-e1', 'evidence-e2')}
		assert _edge_pairs(graph, EvidenceEdgeType.SAME_SOURCE) == set()
		assert graph.stats.duplicate_edge_count == 1
		assert graph.stats.same_source_edge_count == 0


class TestInputIntegrity:
	"""Spec 8: the graph is a citation trail, so a dangling id has to stop the build."""

	def test_task_id_mismatch_is_rejected(self):  # AB
		claim_set, result, nodes = _support_and_contradiction()
		result = _result(*result.verifications, task_id='task-other')

		with pytest.raises(EvidenceOrganizationError, match='Task mismatch'):
			_organize(claim_set, result, nodes)

	def test_duplicate_claim_ids_are_rejected(self):  # AC
		_, result, nodes = _support_and_contradiction()
		claim_set = ClaimSet(
			task_id='task-1',
			task='duplicated',
			answer=STARS_CLAIM,
			claims=[
				Claim(claim_id='claim-1', order=1, text=STARS_CLAIM),
				Claim(claim_id='claim-1', order=2, text=LANGUAGE_CLAIM),
			],
		)

		with pytest.raises(EvidenceOrganizationError, match='Claim set contains claim_id'):
			_organize(claim_set, result, nodes)

	def test_verification_for_unknown_claim_is_rejected(self):  # AD
		claim_set, _, nodes = _support_and_contradiction()
		result = _result(
			_verification('claim-ghost', VerificationStatus.SUPPORTED, _assessment('evidence-high', EvidenceRelation.SUPPORTS))
		)

		with pytest.raises(EvidenceOrganizationError, match='unknown claim_id'):
			_organize(claim_set, result, nodes)

	def test_missing_verification_is_rejected(self):  # AE
		claim_set, result, nodes = _support_and_contradiction()
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM), ('claim-2', 2, LANGUAGE_CLAIM))

		with pytest.raises(EvidenceOrganizationError, match='have no verification'):
			_organize(claim_set, result, nodes)

	def test_duplicate_verification_for_one_claim_is_rejected(self):  # AE
		claim_set, result, nodes = _support_and_contradiction()
		result = _result(result.verifications[0], result.verifications[0])

		with pytest.raises(EvidenceOrganizationError, match='more than once'):
			_organize(claim_set, result, nodes)

	def test_duplicate_evidence_ids_are_rejected(self):  # AF
		claim_set, result, nodes = _support_and_contradiction()
		duplicated = deepcopy(nodes)
		duplicated[1].evidence_id = duplicated[0].evidence_id

		with pytest.raises(EvidenceOrganizationError, match='Evidence list contains evidence_id'):
			_organize(claim_set, result, duplicated)

	def test_assessment_for_unknown_evidence_is_rejected(self):  # AG
		claim_set, result, nodes = _support_and_contradiction()
		result.verifications[0].assessments[1] = _assessment('evidence-phantom', EvidenceRelation.CONTRADICTS)

		with pytest.raises(EvidenceOrganizationError, match='unknown evidence_id'):
			_organize(claim_set, result, nodes)

	def test_one_claim_cannot_be_assessed_twice_for_the_same_evidence(self):
		"""Two verdicts about one page would make the edge set ambiguous."""
		claim_set, result, nodes = _support_and_contradiction()
		result.verifications[0].assessments.append(_assessment('evidence-high', EvidenceRelation.CONTRADICTS))

		with pytest.raises(EvidenceOrganizationError, match='assesses evidence_id'):
			_organize(claim_set, result, nodes)

	def test_integrity_errors_never_quote_evidence_or_claim_text(self):
		"""Spec 22: identifiers are fine to log, page content and prompts are not."""
		claim_set, result, nodes = _support_and_contradiction()
		nodes[0].text = 'PRIVATE_EVIDENCE_BODY'
		claim_set.claims[0].text = 'PRIVATE_CLAIM_BODY'
		result.verifications[0].assessments[0] = _assessment('evidence-phantom', EvidenceRelation.SUPPORTS)

		with pytest.raises(EvidenceOrganizationError) as excinfo:
			_organize(claim_set, result, nodes)

		message = str(excinfo.value)
		assert 'PRIVATE_EVIDENCE_BODY' not in message
		assert 'PRIVATE_CLAIM_BODY' not in message
		assert 'evidence-phantom' in message


class TestOrderingAndDeterminism:
	def test_claims_are_ordered_by_claim_order(self):  # AH
		claim_set = ClaimSet(
			task_id='task-1',
			task='shuffled',
			answer=f'{LANGUAGE_CLAIM} {STARS_CLAIM}',
			claims=[
				Claim(claim_id='claim-later', order=5, text=LANGUAGE_CLAIM),
				Claim(claim_id='claim-earlier', order=2, text=STARS_CLAIM),
			],
		)
		nodes = [_node('evidence-a', HIGH_STAR_TEXT, url='https://github.com/x', step_number=1)]
		result = _result(
			_verification(
				'claim-later', VerificationStatus.UNSUPPORTED, _assessment('evidence-a', EvidenceRelation.INSUFFICIENT)
			),
			_verification('claim-earlier', VerificationStatus.SUPPORTED, _assessment('evidence-a', EvidenceRelation.SUPPORTS)),
		)

		graph = _organize(claim_set, result, nodes)

		assert [(node.claim_id, node.order) for node in graph.claims] == [('claim-earlier', 2), ('claim-later', 5)]
		assert [edge.claim_id for edge in graph.claim_evidence_edges] == ['claim-earlier', 'claim-later']

	def test_evidence_is_ordered_by_step_then_id(self):
		nodes = [
			_node('evidence-z', 'Text about stars one.', url='https://one.example/z', step_number=7),
			_node('evidence-a', 'Text about stars two.', url='https://two.example/a', step_number=3),
			_node('evidence-b', 'Text about stars three.', url='https://three.example/b', step_number=3),
		]

		graph = _organize(*_organize_inputs(nodes))

		assert [node.evidence_id for node in graph.evidence] == ['evidence-a', 'evidence-b', 'evidence-z']

	def test_shuffled_evidence_input_yields_an_identical_graph(self):  # AI
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM), ('claim-2', 2, LANGUAGE_CLAIM))
		nodes = [
			_node('evidence-a', HIGH_STAR_TEXT, url='https://github.com/a', title='GitHub', step_number=1),
			_node('evidence-b', LOW_STAR_TEXT, url='https://blog.example.com/b', title='Blog', step_number=2),
			_node(
				'evidence-c',
				'Browser Use has 111,799 GitHub stars.',
				url='https://mirror.example.com/c',
				title='GitHub',
				step_number=3,
			),
			_node('evidence-d', LANGUAGE_TEXT, url='https://docs.example.com/d', title='Docs', step_number=4),
		]
		result = _result(
			_verification(
				'claim-1',
				VerificationStatus.CONFLICTED,
				_assessment('evidence-a', EvidenceRelation.SUPPORTS),
				_assessment('evidence-b', EvidenceRelation.CONTRADICTS),
				_assessment('evidence-c', EvidenceRelation.SUPPORTS),
			),
			_verification(
				'claim-2',
				VerificationStatus.SUPPORTED,
				_assessment('evidence-d', EvidenceRelation.SUPPORTS),
				_assessment('evidence-c', EvidenceRelation.INSUFFICIENT),
			),
		)

		baseline = _organize(claim_set, result, nodes).model_dump()
		for seed in range(5):
			shuffled = deepcopy(nodes)
			random.Random(seed).shuffle(shuffled)
			assert _organize(claim_set, result, shuffled).model_dump() == baseline, seed

	def test_evidence_edges_follow_the_documented_sort(self):  # AJ
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [
			_node('evidence-a', HIGH_STAR_TEXT, url='https://shared.example/a', title='Stars', step_number=1),
			_node('evidence-b', LOW_STAR_TEXT, url='https://shared.example/b', title='Stars', step_number=2),
			_node('evidence-c', HIGH_STAR_TEXT, url='https://other.example/c', title='Stars', step_number=3),
		]
		result = _result(
			_verification(
				'claim-1',
				VerificationStatus.CONFLICTED,
				_assessment('evidence-a', EvidenceRelation.SUPPORTS),
				_assessment('evidence-b', EvidenceRelation.CONTRADICTS),
				_assessment('evidence-c', EvidenceRelation.SUPPORTS),
			)
		)

		graph = _organize(claim_set, result, nodes)

		assert [
			(edge.relation.value, edge.claim_id or '', edge.source_evidence_id, edge.target_evidence_id)
			for edge in graph.evidence_evidence_edges
		] == [
			# relation first, then claim scope, then the canonical pair; c is on another host, so it
			# conflicts and duplicates but shares no source with b.
			('CONFLICTS_WITH', 'claim-1', 'evidence-a', 'evidence-b'),
			('CONFLICTS_WITH', 'claim-1', 'evidence-b', 'evidence-c'),
			('DUPLICATE', '', 'evidence-a', 'evidence-c'),
			('SAME_SOURCE', '', 'evidence-a', 'evidence-b'),
		]


class TestStats:
	def test_claim_counts_cover_every_status(self):  # AK
		nodes = [_node('evidence-a', HIGH_STAR_TEXT, url='https://one.example/a', step_number=1)]
		claim_set = _claim_set(
			('claim-1', 1, 'A'),
			('claim-2', 2, 'B'),
			('claim-3', 3, 'C'),
			('claim-4', 4, 'D'),
			('claim-5', 5, 'E'),
			('claim-6', 6, 'F'),
		)
		result = _result(
			_verification('claim-1', VerificationStatus.SUPPORTED, _assessment('evidence-a', EvidenceRelation.SUPPORTS)),
			_verification('claim-2', VerificationStatus.PARTIAL, _assessment('evidence-a', EvidenceRelation.PARTIAL_SUPPORT)),
			_verification('claim-3', VerificationStatus.UNSUPPORTED, _assessment('evidence-a', EvidenceRelation.INSUFFICIENT)),
			_verification('claim-4', VerificationStatus.CONTRADICTED, _assessment('evidence-a', EvidenceRelation.CONTRADICTS)),
			_verification('claim-5', VerificationStatus.CONFLICTED, _assessment('evidence-a', EvidenceRelation.SUPPORTS)),
			_verification('claim-6', VerificationStatus.NO_EVIDENCE),
		)

		stats = _organize(claim_set, result, nodes).stats

		assert (
			stats.claim_count,
			stats.supported_claim_count,
			stats.partial_claim_count,
			stats.unsupported_claim_count,
			stats.contradicted_claim_count,
			stats.conflicted_claim_count,
			stats.no_evidence_claim_count,
		) == (6, 1, 1, 1, 1, 1, 1)
		assert stats.supported_claim_count + stats.partial_claim_count + stats.unsupported_claim_count == 3

	def test_relation_counts_match_the_edges(self):  # AL
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM))
		nodes = [
			_node('evidence-a', HIGH_STAR_TEXT, url='https://one.example/a', step_number=1),
			_node('evidence-b', LOW_STAR_TEXT, url='https://two.example/b', step_number=2),
			_node('evidence-c', LANGUAGE_TEXT, url='https://three.example/c', step_number=3),
			_node('evidence-d', MCP_PARTIAL_TEXT, url='https://four.example/d', step_number=4),
		]
		result = _result(
			_verification(
				'claim-1',
				VerificationStatus.CONFLICTED,
				_assessment('evidence-a', EvidenceRelation.SUPPORTS),
				_assessment('evidence-b', EvidenceRelation.CONTRADICTS),
				_assessment('evidence-c', EvidenceRelation.INSUFFICIENT),
				_assessment('evidence-d', EvidenceRelation.PARTIAL_SUPPORT),
			)
		)

		stats = _organize(claim_set, result, nodes).stats

		assert (
			stats.support_edge_count,
			stats.partial_support_edge_count,
			stats.contradict_edge_count,
			stats.insufficient_edge_count,
		) == (1, 1, 1, 1)
		assert (
			stats.support_edge_count
			+ stats.partial_support_edge_count
			+ stats.contradict_edge_count
			+ stats.insufficient_edge_count
		) == len(_organize(claim_set, result, nodes).claim_evidence_edges)

	def test_evidence_relation_counts_match_the_edges(self):  # AM
		claim_set, result, nodes = _support_and_contradiction()

		stats = _organize(claim_set, result, nodes).stats

		assert (stats.same_source_edge_count, stats.duplicate_edge_count, stats.conflict_edge_count) == (0, 0, 1)

	def test_stats_are_recomputed_rather_than_trusted(self):  # AN
		claim_set, result, nodes = _support_and_contradiction()
		graph = _organize(claim_set, result, nodes)

		lying = EvidenceGraph(
			task_id=graph.task_id,
			claims=graph.claims,
			evidence=graph.evidence,
			claim_evidence_edges=graph.claim_evidence_edges,
			evidence_evidence_edges=graph.evidence_evidence_edges,
			stats=EvidenceGraphStats(claim_count=99, evidence_count=99, conflict_edge_count=99),
		)

		assert lying.stats == graph.stats
		assert lying.stats.claim_count == 1
		assert lying.stats.conflict_edge_count == 1

	def test_stats_describe_the_lists_that_ship_with_them(self):
		"""No count may refer to something the caller cannot find in the graph."""
		claim_set = _claim_set(('claim-1', 1, STARS_CLAIM), ('claim-2', 2, LANGUAGE_CLAIM))
		nodes = [
			_node('evidence-a', HIGH_STAR_TEXT, url='https://shared.example/a', title='GitHub', step_number=1),
			_node('evidence-b', HIGH_STAR_TEXT, url='https://shared.example/b', title='GitHub', step_number=2),
		]
		result = _result(
			_verification(
				'claim-1',
				VerificationStatus.SUPPORTED,
				*[_assessment(node.evidence_id, EvidenceRelation.SUPPORTS) for node in nodes],
			),
			_verification('claim-2', VerificationStatus.NO_EVIDENCE),
		)

		graph = _organize(claim_set, result, nodes)
		stats = graph.stats

		assert stats.claim_count == len(graph.claims)
		assert stats.evidence_count == len(graph.evidence)
		assert stats.support_edge_count == len(graph.claim_evidence_edges)
		assert stats.same_source_edge_count + stats.duplicate_edge_count + stats.conflict_edge_count == len(
			graph.evidence_evidence_edges
		)
		assert stats.no_evidence_claim_count == 1


class TestPurityAndSerialisation:
	def test_inputs_are_not_mutated(self):  # AP
		claim_set, result, nodes = _support_and_contradiction()
		before = (deepcopy(claim_set), deepcopy(result), deepcopy(nodes))

		_organize(claim_set, result, nodes)

		assert (claim_set, result, nodes) == before

	def test_graph_round_trips_through_json(self):  # AO
		claim_set, result, nodes = _support_and_contradiction()
		graph = _organize(claim_set, result, nodes)

		parsed = EvidenceGraph.model_validate_json(graph.model_dump_json())

		assert parsed == graph
		assert parsed.stats == graph.stats
		assert parsed.evidence_evidence_edges[0].relation is EvidenceEdgeType.CONFLICTS_WITH
		assert parsed.claims[0].verification_status is VerificationStatus.CONFLICTED

	def test_empty_task_produces_an_empty_graph(self):
		claim_set = _claim_set()

		graph = _organize(claim_set, _result(), [])

		assert graph == EvidenceGraph(task_id='task-1')
		assert graph.stats.as_dict() == {name: 0 for name in EvidenceGraphStats.model_fields}
