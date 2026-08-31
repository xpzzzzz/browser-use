"""Unit tests for deterministic claim/evidence candidate alignment."""

import inspect

import pytest
from pydantic import ValidationError

from browser_use.evidence import (
	AlignmentResult,
	Claim,
	ClaimAlignment,
	ClaimSet,
	EvidenceAligner,
	EvidenceMatch,
	EvidenceNode,
)
from browser_use.evidence.alignment import (
	CLAIM_COVERAGE_WEIGHT,
	JACCARD_WEIGHT,
	TITLE_COVERAGE_WEIGHT,
	tokenize,
)


def _node(step_number: int, text: str, *, title: str = '', evidence_id: str | None = None) -> EvidenceNode:
	return EvidenceNode(
		evidence_id=evidence_id or f'evidence-{step_number}',
		task_id='task-1',
		step_number=step_number,
		url=f'https://example.com/{step_number}',
		title=title,
		text=text,
	)


def _claim_set(*texts: str) -> ClaimSet:
	return ClaimSet(
		task_id='task-1',
		task='How does Browser Use do?',
		answer=' '.join(texts),
		claims=[Claim(claim_id=f'claim-{index}', order=index, text=text) for index, text in enumerate(texts, start=1)],
	)


class TestEvidenceMatch:
	def test_score_must_stay_in_unit_range(self):
		for invalid in (-0.01, 1.01):
			with pytest.raises(ValidationError):
				EvidenceMatch(evidence_id='e1', score=invalid, rank=1)

	def test_rank_must_be_one_based(self):
		with pytest.raises(ValidationError):
			EvidenceMatch(evidence_id='e1', score=0.5, rank=0)


class TestTokenizer:
	def test_empty_text_yields_no_tokens(self):
		assert tokenize('') == []
		assert tokenize('   ') == []

	def test_punctuation_is_dropped_and_case_is_folded(self):
		assert tokenize('Browser-Use, "STARS"!') == ['browser', 'use', 'stars']

	def test_numbers_are_kept(self):
		assert tokenize('v0.1.12 and 30k') == ['v0', '1', '12', 'and', '30k']

	def test_non_ascii_text_survives(self):
		assert tokenize('Étoile café') == ['étoile', 'café']

	def test_nfkc_normalisation_makes_full_width_text_match(self):
		assert tokenize('Ｐｙｔｈｏｎ') == tokenize('Python')

	def test_cjk_text_produces_characters_and_bigrams(self):
		tokens = tokenize('中文分词')

		assert '中' in tokens
		assert '中文' in tokens
		assert tokens.count('中文') == 1


class TestScoreFormula:
	def test_weights_sum_to_one(self):
		assert CLAIM_COVERAGE_WEIGHT + JACCARD_WEIGHT + TITLE_COVERAGE_WEIGHT == pytest.approx(1.0)

	def test_identical_tokens_score_coverage_plus_jaccard(self):
		result = EvidenceAligner().align(
			claim_set=_claim_set('alpha beta'),
			evidence_nodes=[_node(1, 'alpha beta')],
		)

		# claim_coverage = 1.0, jaccard = 1.0, title_coverage = 0.0
		assert result.alignments[0].matches[0].score == pytest.approx(0.65 + 0.20)

	def test_title_overlap_adds_its_weight(self):
		result = EvidenceAligner().align(
			claim_set=_claim_set('alpha beta'),
			evidence_nodes=[
				_node(1, 'alpha beta', title='alpha', evidence_id='with-title'),
				_node(2, 'alpha beta', title='unrelated', evidence_id='bad-title'),
			],
		)

		scores = {match.evidence_id: match.score for match in result.alignments[0].matches}
		assert scores['with-title'] == pytest.approx(0.65 + 0.20 + 0.5 * 0.15)
		assert scores['with-title'] > scores['bad-title']


class TestEvidenceAligner:
	def test_align_is_synchronous(self):
		assert inspect.iscoroutinefunction(EvidenceAligner.align) is False

	def test_default_top_k_is_five(self):
		assert EvidenceAligner().top_k == 5

	def test_clearly_relevant_evidence_ranks_first(self):
		claim_set = _claim_set('Browser Use has 30k stars.')
		nodes = [
			_node(1, 'A cooking recipe for pasta.'),
			_node(2, 'Browser Use has 30k stars on GitHub.'),
			_node(3, 'Weather report for Berlin.'),
		]

		result = EvidenceAligner().align(claim_set=claim_set, evidence_nodes=nodes)

		assert result.alignments[0].matches[0].evidence_id == 'evidence-2'
		assert result.alignments[0].matches[0].rank == 1

	def test_top_k_limits_candidate_count(self):
		claim_set = _claim_set('alpha beta gamma')
		nodes = [_node(step, 'alpha beta gamma delta', evidence_id=f'evidence-{step}') for step in range(1, 8)]

		result = EvidenceAligner(top_k=3).align(claim_set=claim_set, evidence_nodes=nodes)

		assert len(result.alignments[0].matches) == 3

	def test_ranks_are_contiguous_from_one(self):
		claim_set = _claim_set('alpha beta gamma')
		nodes = [_node(step, f'alpha beta gamma extra {step}') for step in range(1, 5)]

		result = EvidenceAligner(top_k=4).align(claim_set=claim_set, evidence_nodes=nodes)

		assert [match.rank for match in result.alignments[0].matches] == [1, 2, 3, 4]

	def test_all_scores_stay_inside_unit_range(self):
		claim_set = _claim_set('alpha beta gamma', 'alpha', 'totally different tokens')
		nodes = [_node(1, 'alpha beta gamma'), _node(2, 'alpha'), _node(3, 'gamma epsilon'), _node(4, '')]

		result = EvidenceAligner().align(claim_set=claim_set, evidence_nodes=nodes)

		all_scores = [match.score for alignment in result.alignments for match in alignment.matches]
		assert all_scores
		assert all(0.0 <= score <= 1.0 for score in all_scores)

	def test_higher_claim_token_coverage_scores_higher(self):
		claim_set = _claim_set('alpha beta gamma delta')
		nodes = [_node(1, 'alpha beta', evidence_id='half'), _node(2, 'alpha beta gamma delta', evidence_id='full')]

		result = EvidenceAligner().align(claim_set=claim_set, evidence_nodes=nodes)

		scores = {match.evidence_id: match.score for match in result.alignments[0].matches}
		assert scores['full'] > scores['half']

	def test_no_token_overlap_yields_empty_matches(self):
		result = EvidenceAligner().align(
			claim_set=_claim_set('quantum flux capacitor'),
			evidence_nodes=[_node(1, 'A recipe for tomato soup.')],
		)

		assert result.alignments[0].matches == []

	def test_irrelevant_evidence_is_not_used_to_pad_top_k(self):
		claim_set = _claim_set('alpha beta')
		nodes = [_node(step, 'alpha beta') for step in (1, 2)] + [_node(step, 'unrelated content here') for step in (3, 4, 5)]

		result = EvidenceAligner(top_k=5).align(claim_set=claim_set, evidence_nodes=nodes)

		assert [match.evidence_id for match in result.alignments[0].matches] == ['evidence-1', 'evidence-2']

	def test_empty_evidence_keeps_one_alignment_per_claim(self):
		claim_set = _claim_set('first claim', 'second claim')

		result = EvidenceAligner().align(claim_set=claim_set, evidence_nodes=[])

		assert len(result.alignments) == 2
		assert [alignment.matches for alignment in result.alignments] == [[], []]
		assert [alignment.claim_id for alignment in result.alignments] == ['claim-1', 'claim-2']

	def test_empty_claim_set_yields_no_alignments(self):
		result = EvidenceAligner().align(claim_set=_claim_set(), evidence_nodes=[_node(1, 'alpha')])

		assert result.alignments == []

	def test_empty_text_with_informative_title_is_still_recalled(self):
		result = EvidenceAligner().align(
			claim_set=_claim_set('browser use stars'),
			evidence_nodes=[_node(1, '', title='Browser Use GitHub stars')],
		)

		assert [match.evidence_id for match in result.alignments[0].matches] == ['evidence-1']

	def test_node_without_text_or_title_is_never_recalled(self):
		result = EvidenceAligner().align(
			claim_set=_claim_set('browser use stars'),
			evidence_nodes=[_node(1, '', title='', evidence_id='blank')],
		)

		assert result.alignments[0].matches == []

	def test_equal_scores_use_the_documented_tie_break(self):
		claim_set = _claim_set('alpha beta')
		nodes = [
			_node(2, 'alpha beta', evidence_id='step-two'),
			_node(1, 'alpha beta', evidence_id='step-one-b'),
			_node(1, 'alpha beta', evidence_id='step-one-a'),
		]

		result = EvidenceAligner().align(claim_set=claim_set, evidence_nodes=nodes)

		scores = {match.evidence_id: match.score for match in result.alignments[0].matches}
		assert len(set(scores.values())) == 1
		assert [match.evidence_id for match in result.alignments[0].matches] == ['step-one-a', 'step-one-b', 'step-two']

	def test_duplicate_evidence_id_is_rejected(self):
		aligner = EvidenceAligner()
		nodes = [_node(1, 'alpha', evidence_id='same'), _node(2, 'beta', evidence_id='same')]

		with pytest.raises(ValueError, match='Duplicate evidence_id'):
			aligner.align(claim_set=_claim_set('alpha'), evidence_nodes=nodes)

	def test_top_k_below_one_is_rejected(self):
		for invalid in (0, -3):
			with pytest.raises(ValueError, match='top_k'):
				EvidenceAligner(top_k=invalid)

	def test_claim_and_evidence_ids_are_copied_verbatim(self):
		claim_set = _claim_set('alpha beta', 'gamma delta')
		nodes = [_node(1, 'alpha beta', evidence_id='weird-id-019f'), _node(2, 'gamma delta', evidence_id='weird-id-019g')]

		result = EvidenceAligner().align(claim_set=claim_set, evidence_nodes=nodes)

		assert [alignment.claim_id for alignment in result.alignments] == [claim.claim_id for claim in claim_set.claims]
		matched_ids = {match.evidence_id for alignment in result.alignments for match in alignment.matches}
		assert matched_ids == {'weird-id-019f', 'weird-id-019g'}

	def test_alignment_result_carries_the_task_id(self):
		assert EvidenceAligner().align(claim_set=_claim_set('alpha'), evidence_nodes=[]).task_id == 'task-1'

	def test_repeated_calls_produce_identical_results(self):
		claim_set = _claim_set('alpha beta gamma', 'browser use stars count')
		nodes = [_node(1, 'alpha beta'), _node(2, 'browser use has 30k stars'), _node(3, 'gamma'), _node(4, 'noise token here')]
		aligner = EvidenceAligner(top_k=2)

		first = aligner.align(claim_set=claim_set, evidence_nodes=nodes).model_dump()
		for _ in range(3):
			assert aligner.align(claim_set=claim_set, evidence_nodes=nodes).model_dump() == first

	def test_inputs_are_not_mutated(self):
		claim_set = _claim_set('alpha beta')
		nodes = [_node(1, 'alpha beta', title='alpha', evidence_id='keep-me')]
		nodes_copy = [node.model_copy(deep=True) for node in nodes]

		EvidenceAligner().align(claim_set=claim_set, evidence_nodes=nodes)

		assert nodes == nodes_copy
		assert [claim.text for claim in claim_set.claims] == ['alpha beta']

	def test_scoring_ignores_screenshot_metadata_and_actions(self):
		plain = _node(1, 'alpha beta', title='alpha', evidence_id='plain')
		laden = _node(1, 'alpha beta', title='alpha', evidence_id='laden')
		laden = laden.model_copy(
			update={
				'screenshot_path': '/tmp/step_0001.png',
				'action_names': ['click', 'scroll'],
				'metadata': {'blob': 'x' * 5000},
			}
		)

		first = EvidenceAligner().align(claim_set=_claim_set('alpha beta'), evidence_nodes=[plain])
		second = EvidenceAligner().align(claim_set=_claim_set('alpha beta'), evidence_nodes=[laden])

		assert first.alignments[0].matches[0].score == second.alignments[0].matches[0].score

	def test_result_is_a_pydantic_model_that_round_trips_through_json(self):
		result = EvidenceAligner().align(
			claim_set=_claim_set('alpha beta'), evidence_nodes=[_node(1, 'alpha beta', title='alpha')]
		)

		assert isinstance(result, AlignmentResult)
		assert AlignmentResult.model_validate_json(result.model_dump_json()) == result
		assert isinstance(result.alignments[0], ClaimAlignment)
		assert isinstance(result.alignments[0].matches[0], EvidenceMatch)

	def test_alignments_follow_claim_order(self):
		claim_set = _claim_set('alpha one', 'beta two', 'gamma three')
		nodes = [_node(1, 'alpha one'), _node(2, 'gamma three')]

		result = EvidenceAligner().align(claim_set=claim_set, evidence_nodes=nodes)

		assert [alignment.claim_id for alignment in result.alignments] == ['claim-1', 'claim-2', 'claim-3']


class TestScenarioExamples:
	def test_python_claim_prefers_the_page_that_states_it(self):
		claim_set = _claim_set('Browser Use is primarily written in Python.')
		nodes = [
			_node(
				1,
				'Browser Use is an open-source project primarily written in Python.',
				title='browser-use/browser-use',
				evidence_id='evidence-a',
			),
			_node(2, 'Tomorrow will be sunny with a high temperature.', title='Weather Forecast', evidence_id='evidence-b'),
			_node(
				3,
				'This page discusses browser automation frameworks.',
				title='Python Browser Automation',
				evidence_id='evidence-c',
			),
		]

		result = EvidenceAligner().align(claim_set=claim_set, evidence_nodes=nodes)

		matches = result.alignments[0].matches
		assert matches[0].evidence_id == 'evidence-a'
		assert matches[0].rank == 1
		assert 'evidence-b' not in [match.evidence_id for match in matches]

	def test_numeric_claim_prefers_numeric_evidence_without_number_reasoning(self):
		claim_set = _claim_set('Browser Use has more than 100,000 GitHub stars.')
		nodes = [
			_node(1, 'Browser Use is a Python browser automation project.', evidence_id='evidence-b'),
			_node(2, 'Browser Use has 111,799 stars on GitHub.', evidence_id='evidence-a'),
		]

		result = EvidenceAligner().align(claim_set=claim_set, evidence_nodes=nodes)

		scores = {match.evidence_id: match.score for match in result.alignments[0].matches}
		assert scores['evidence-a'] > scores['evidence-b']
		assert result.alignments[0].matches[0].evidence_id == 'evidence-a'

	def test_cjk_claim_recalls_cjk_evidence(self):
		claim_set = _claim_set('Browser Use 项目有三十万星标')
		nodes = [
			_node(1, '今天的天气预报是晴天。', evidence_id='weather'),
			_node(2, 'Browser Use 项目星标数达到三十万。', evidence_id='stars'),
		]

		result = EvidenceAligner().align(claim_set=claim_set, evidence_nodes=nodes)

		assert result.alignments[0].matches[0].evidence_id == 'stars'
