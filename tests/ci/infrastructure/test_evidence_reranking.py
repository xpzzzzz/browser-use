"""Unit tests for LLM semantic evidence reranking.

Only the test marked ``integration`` talks to a real model, and it skips unless an API key is set.
"""

import os
import re
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from browser_use.evidence import (
	AlignmentResult,
	Claim,
	ClaimAlignment,
	ClaimReranking,
	ClaimSet,
	EvidenceAligner,
	EvidenceMatch,
	EvidenceNode,
	EvidenceRerankingError,
	RawSemanticEvidenceScore,
	RawSemanticReranking,
	RerankedEvidenceMatch,
	RerankingResult,
	SemanticEvidenceReranker,
)
from browser_use.evidence.reranking import _SEMANTIC_RERANKING_SYSTEM_PROMPT, _TRUNCATION_MARKER
from browser_use.llm.views import ChatInvokeCompletion

_CANDIDATE_ID_PATTERN = re.compile(r'^evidence_id: (.+)$', re.MULTILINE)

STARS_CLAIM = 'Browser Use has 100,000 GitHub stars.'
HIGH_STAR_TEXT = 'Browser Use has 111,799 GitHub stars.'
LOW_STAR_TEXT = 'Browser Use has only 30,000 GitHub stars.'
LANGUAGE_TEXT = 'Browser Use is primarily written in Python.'


class FakeChatModel:
	"""Records every call and answers with whatever the test's reply function returns."""

	def __init__(self, reply: Callable[[str, int], Any]) -> None:
		self.model = 'fake-reranking-model'
		self.provider = 'fake'
		self.name = 'fake-reranking-model'
		self.model_name = 'fake-reranking-model'
		self._verified_api_keys = True
		self.calls: list[dict] = []
		self._reply = reply

	async def ainvoke(self, messages, output_format=None, **kwargs) -> ChatInvokeCompletion:
		self.calls.append({'messages': messages, 'output_format': output_format, 'kwargs': kwargs})
		completion = self._reply(messages[-1].text, len(self.calls) - 1)
		return ChatInvokeCompletion(completion=completion, usage=None)

	def prompts(self) -> list[str]:
		return [call['messages'][-1].text for call in self.calls]


def _never_called() -> Callable[[str, int], Any]:
	"""Reply that fails the test if the reranker spends a call it should have skipped."""

	def _reply(_prompt: str, _index: int) -> Any:
		raise AssertionError('the reranker should not have called the model')

	return _reply


def _score_each(scores: dict[str, float]) -> Callable[[str, int], RawSemanticReranking]:
	"""Reply that scores exactly the candidates present in the prompt, looking scores up by id."""

	def _reply(prompt: str, _index: int) -> RawSemanticReranking:
		return RawSemanticReranking(
			scores=[
				RawSemanticEvidenceScore(evidence_id=evidence_id, relevance_score=scores[evidence_id])
				for evidence_id in _CANDIDATE_ID_PATTERN.findall(prompt)
			]
		)

	return _reply


def _reply_with(*pairs: tuple[str, float]) -> Callable[[str, int], RawSemanticReranking]:
	"""Reply with a fixed score list, ignoring the prompt, to break id integrity on purpose."""

	def _reply(_prompt: str, _index: int) -> RawSemanticReranking:
		return RawSemanticReranking(
			scores=[RawSemanticEvidenceScore(evidence_id=evidence_id, relevance_score=score) for evidence_id, score in pairs]
		)

	return _reply


def _node(step_number: int, text: str, *, evidence_id: str, title: str = '') -> EvidenceNode:
	return EvidenceNode(
		evidence_id=evidence_id,
		task_id='task-1',
		step_number=step_number,
		url=f'https://example.com/{step_number}',
		title=title,
		text=text,
	)


def _claim_set(*texts: str, task_id: str = 'task-1') -> ClaimSet:
	return ClaimSet(
		task_id=task_id,
		task='How popular is Browser Use?',
		answer=' '.join(texts),
		claims=[Claim(claim_id=f'claim-{index}', order=index, text=text) for index, text in enumerate(texts, start=1)],
	)


def _alignment(per_claim: dict[str, list[tuple[str, float]]], *, task_id: str = 'task-1') -> AlignmentResult:
	"""Build a Phase 4A-shaped alignment result: claim_id -> [(evidence_id, lexical_score)]."""
	return AlignmentResult(
		task_id=task_id,
		alignments=[
			ClaimAlignment(
				claim_id=claim_id,
				matches=[
					EvidenceMatch(evidence_id=evidence_id, score=score, rank=rank)
					for rank, (evidence_id, score) in enumerate(matches, start=1)
				],
			)
			for claim_id, matches in per_claim.items()
		],
	)


def _three_candidates():
	"""One claim, three candidates with descending lexical scores 0.5 / 0.4 / 0.3."""
	nodes = [
		_node(1, HIGH_STAR_TEXT, evidence_id='evidence-high', title='GitHub'),
		_node(2, LANGUAGE_TEXT, evidence_id='evidence-language', title='Docs'),
		_node(3, LOW_STAR_TEXT, evidence_id='evidence-low', title='GitHub'),
	]
	claim_set = _claim_set(STARS_CLAIM)
	alignment = _alignment({'claim-1': [('evidence-high', 0.5), ('evidence-language', 0.4), ('evidence-low', 0.3)]})
	return claim_set, alignment, nodes


def _by_id(reranking: ClaimReranking) -> dict[str, RerankedEvidenceMatch]:
	return {match.evidence_id: match for match in reranking.matches}


class TestRerankingModels:
	def test_semantic_score_is_bounded(self):
		for invalid in (-0.01, 1.01):
			with pytest.raises(ValidationError):
				RawSemanticEvidenceScore(evidence_id='e1', relevance_score=invalid)

	def test_reranked_match_keeps_both_scores_and_bounds_the_combination(self):
		match = RerankedEvidenceMatch(evidence_id='e1', lexical_score=0.4, semantic_score=0.9, combined_score=0.75, rank=1)

		assert (match.lexical_score, match.semantic_score, match.combined_score) == (0.4, 0.9, 0.75)
		with pytest.raises(ValidationError):
			RerankedEvidenceMatch(evidence_id='e1', lexical_score=0.4, semantic_score=0.9, combined_score=1.4, rank=1)

	def test_list_fields_default_to_empty(self):
		assert ClaimReranking(claim_id='claim-1').matches == []
		assert RerankingResult(task_id='task-1').rerankings == []


class TestSemanticEvidenceReranker:
	async def test_three_candidates_produce_three_reranked_matches(self):
		claim_set, alignment, nodes = _three_candidates()
		reranker = SemanticEvidenceReranker(
			FakeChatModel(_score_each({'evidence-high': 0.9, 'evidence-language': 0.2, 'evidence-low': 0.95}))
		)

		result = await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		assert isinstance(result, RerankingResult)
		assert len(result.rerankings) == 1
		assert len(result.rerankings[0].matches) == 3
		assert all(isinstance(match, RerankedEvidenceMatch) for match in result.rerankings[0].matches)

	async def test_semantic_scores_are_stored_per_candidate(self):
		claim_set, alignment, nodes = _three_candidates()
		reranker = SemanticEvidenceReranker(
			FakeChatModel(_score_each({'evidence-high': 0.9, 'evidence-language': 0.2, 'evidence-low': 0.95}))
		)

		result = await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		semantic = {match.evidence_id: match.semantic_score for match in result.rerankings[0].matches}
		assert semantic == {'evidence-high': 0.9, 'evidence-language': 0.2, 'evidence-low': 0.95}

	async def test_lexical_scores_are_carried_over_unchanged(self):
		claim_set, alignment, nodes = _three_candidates()
		reranker = SemanticEvidenceReranker(
			FakeChatModel(_score_each({'evidence-high': 0.1, 'evidence-language': 0.2, 'evidence-low': 0.3}))
		)

		result = await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		lexical = {match.evidence_id: match.lexical_score for match in result.rerankings[0].matches}
		assert lexical == {'evidence-high': 0.5, 'evidence-language': 0.4, 'evidence-low': 0.3}

	async def test_combined_score_is_the_weighted_blend(self):
		claim_set, alignment, nodes = _three_candidates()
		reranker = SemanticEvidenceReranker(
			FakeChatModel(_score_each({'evidence-high': 0.9, 'evidence-language': 0.2, 'evidence-low': 0.95})),
			semantic_weight=0.7,
		)

		result = await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		combined = {match.evidence_id: match.combined_score for match in result.rerankings[0].matches}
		assert combined['evidence-high'] == pytest.approx(0.3 * 0.5 + 0.7 * 0.9)
		assert combined['evidence-language'] == pytest.approx(0.3 * 0.4 + 0.7 * 0.2)
		assert combined['evidence-low'] == pytest.approx(0.3 * 0.3 + 0.7 * 0.95)

	def test_default_weights_are_seventy_semantic(self):
		reranker = SemanticEvidenceReranker(FakeChatModel(_never_called()))

		assert reranker.semantic_weight == 0.7
		assert reranker.lexical_weight == pytest.approx(0.3)

	async def test_weight_of_zero_reproduces_the_lexical_baseline(self):
		claim_set, alignment, nodes = _three_candidates()
		reranker = SemanticEvidenceReranker(
			FakeChatModel(_score_each({'evidence-high': 0.0, 'evidence-language': 1.0, 'evidence-low': 1.0})),
			semantic_weight=0.0,
		)

		result = await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		assert [match.combined_score for match in result.rerankings[0].matches] == pytest.approx([0.5, 0.4, 0.3])

	async def test_weight_of_one_ignores_the_lexical_scores(self):
		claim_set, alignment, nodes = _three_candidates()
		reranker = SemanticEvidenceReranker(
			FakeChatModel(_score_each({'evidence-high': 0.1, 'evidence-language': 0.9, 'evidence-low': 0.5})),
			semantic_weight=1.0,
		)

		result = await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		assert [match.evidence_id for match in result.rerankings[0].matches] == [
			'evidence-language',
			'evidence-low',
			'evidence-high',
		]
		assert [match.combined_score for match in result.rerankings[0].matches] == pytest.approx([0.9, 0.5, 0.1])

	def test_out_of_range_weights_and_budgets_are_rejected(self):
		for invalid_weight in (-0.1, 1.5):
			with pytest.raises(ValueError, match='semantic_weight'):
				SemanticEvidenceReranker(FakeChatModel(_never_called()), semantic_weight=invalid_weight)
		with pytest.raises(ValueError, match='max_evidence_chars'):
			SemanticEvidenceReranker(FakeChatModel(_never_called()), max_evidence_chars=0)

	async def test_ranking_follows_combined_score_not_lexical_order(self):
		claim_set, alignment, nodes = _three_candidates()
		reranker = SemanticEvidenceReranker(
			FakeChatModel(_score_each({'evidence-high': 0.2, 'evidence-language': 0.99, 'evidence-low': 0.3}))
		)

		result = await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		assert [match.evidence_id for match in result.rerankings[0].matches] == [
			'evidence-language',
			'evidence-low',
			'evidence-high',
		]

	async def test_ranks_are_renumbered_from_one(self):
		claim_set, alignment, nodes = _three_candidates()
		reranker = SemanticEvidenceReranker(
			FakeChatModel(_score_each({'evidence-high': 0.1, 'evidence-language': 0.2, 'evidence-low': 0.3}))
		)

		result = await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		assert [match.rank for match in result.rerankings[0].matches] == [1, 2, 3]

	async def test_equal_combined_score_prefers_the_higher_semantic_score(self):
		nodes = [
			_node(1, HIGH_STAR_TEXT, evidence_id='evidence-lex-strong'),
			_node(2, LOW_STAR_TEXT, evidence_id='evidence-sem-strong'),
		]
		claim_set = _claim_set(STARS_CLAIM)
		# both land on combined 0.5: 0.5 * 0.8 + 0.5 * 0.2 versus 0.5 * 0.2 + 0.5 * 0.8
		alignment = _alignment({'claim-1': [('evidence-lex-strong', 0.8), ('evidence-sem-strong', 0.2)]})
		reranker = SemanticEvidenceReranker(
			FakeChatModel(_score_each({'evidence-lex-strong': 0.2, 'evidence-sem-strong': 0.8})),
			semantic_weight=0.5,
		)

		result = await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		combined = {match.evidence_id: match.combined_score for match in result.rerankings[0].matches}
		assert combined['evidence-lex-strong'] == pytest.approx(combined['evidence-sem-strong'])
		assert [match.evidence_id for match in result.rerankings[0].matches] == ['evidence-sem-strong', 'evidence-lex-strong']

	async def test_full_tie_falls_back_to_evidence_id_order(self):
		nodes = [
			_node(3, HIGH_STAR_TEXT, evidence_id='evidence-c'),
			_node(1, LOW_STAR_TEXT, evidence_id='evidence-a'),
			_node(2, LANGUAGE_TEXT, evidence_id='evidence-b'),
		]
		claim_set = _claim_set(STARS_CLAIM)
		alignment = _alignment({'claim-1': [('evidence-c', 0.5), ('evidence-a', 0.5), ('evidence-b', 0.5)]})
		reranker = SemanticEvidenceReranker(FakeChatModel(_score_each({'evidence-a': 0.5, 'evidence-b': 0.5, 'evidence-c': 0.5})))

		result = await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		assert [match.evidence_id for match in result.rerankings[0].matches] == ['evidence-a', 'evidence-b', 'evidence-c']

	async def test_one_call_covers_every_candidate_of_a_claim(self):
		llm = FakeChatModel(_score_each({'evidence-high': 0.5, 'evidence-language': 0.5, 'evidence-low': 0.5}))
		claim_set, alignment, nodes = _three_candidates()

		await SemanticEvidenceReranker(llm).rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		assert len(llm.calls) == 1
		prompt = llm.prompts()[0]
		for evidence_id in ('evidence-high', 'evidence-language', 'evidence-low'):
			assert f'evidence_id: {evidence_id}' in prompt

	async def test_two_claims_cost_two_calls_not_six(self):
		nodes = [
			_node(1, HIGH_STAR_TEXT, evidence_id='evidence-high'),
			_node(2, LANGUAGE_TEXT, evidence_id='evidence-language'),
			_node(3, LOW_STAR_TEXT, evidence_id='evidence-low'),
		]
		claim_set = _claim_set(STARS_CLAIM, 'Browser Use is written in Python.')
		alignment = _alignment(
			{
				'claim-1': [('evidence-high', 0.5), ('evidence-language', 0.4), ('evidence-low', 0.3)],
				'claim-2': [('evidence-language', 0.6), ('evidence-high', 0.2), ('evidence-low', 0.1)],
			}
		)
		llm = FakeChatModel(_score_each({'evidence-high': 0.5, 'evidence-language': 0.5, 'evidence-low': 0.5}))

		result = await SemanticEvidenceReranker(llm).rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		assert len(llm.calls) == 2
		assert [reranking.claim_id for reranking in result.rerankings] == ['claim-1', 'claim-2']

	async def test_claim_without_candidates_makes_no_call(self):
		llm = FakeChatModel(_never_called())
		claim_set = _claim_set('A claim about stars.', 'Another claim about stars.')
		alignment = _alignment({'claim-1': [('evidence-1', 0.5)], 'claim-2': []})

		# the reply raises if the reranker spends a call for the second claim
		result = await SemanticEvidenceReranker(_scored_model(llm, {'evidence-1': 0.5})).rerank(
			claim_set=claim_set, alignment_result=alignment, evidence_nodes=[_node(1, HIGH_STAR_TEXT, evidence_id='evidence-1')]
		)

		assert len(result.rerankings) == 2
		assert result.rerankings[1].matches == []
		assert result.rerankings[1].claim_id == 'claim-2'

	async def test_empty_claim_set_makes_no_call(self):
		llm = FakeChatModel(_never_called())

		result = await SemanticEvidenceReranker(llm).rerank(
			claim_set=_claim_set(),
			alignment_result=AlignmentResult(task_id='task-1'),
			evidence_nodes=[_node(1, 'text', evidence_id='evidence-1')],
		)

		assert result.rerankings == []
		assert llm.calls == []


def _scored_model(llm: FakeChatModel, scores: dict[str, float]) -> FakeChatModel:
	"""Swap a never-call reply for real scores so a partial test can still exercise one claim."""
	llm._reply = _score_each(scores)
	return llm


class TestIdIntegrity:
	async def test_alignment_pointing_at_unknown_evidence_is_rejected(self):
		reranker = SemanticEvidenceReranker(FakeChatModel(_never_called()))

		with pytest.raises(EvidenceRerankingError, match='unknown evidence_id'):
			await reranker.rerank(
				claim_set=_claim_set(STARS_CLAIM),
				alignment_result=_alignment({'claim-1': [('evidence-ghost', 0.5)]}),
				evidence_nodes=[_node(1, HIGH_STAR_TEXT, evidence_id='evidence-1')],
			)

	async def test_task_id_mismatch_is_rejected(self):
		reranker = SemanticEvidenceReranker(FakeChatModel(_never_called()))

		with pytest.raises(EvidenceRerankingError, match='Task mismatch'):
			await reranker.rerank(
				claim_set=_claim_set(STARS_CLAIM),
				alignment_result=_alignment({'claim-1': []}, task_id='task-other'),
				evidence_nodes=[_node(1, HIGH_STAR_TEXT, evidence_id='evidence-1')],
			)

	async def test_alignment_for_unknown_claim_is_rejected(self):
		reranker = SemanticEvidenceReranker(FakeChatModel(_never_called()))

		with pytest.raises(EvidenceRerankingError, match='unknown claim_id'):
			await reranker.rerank(
				claim_set=_claim_set(STARS_CLAIM),
				alignment_result=_alignment({'claim-nowhere': [('evidence-1', 0.5)]}),
				evidence_nodes=[_node(1, HIGH_STAR_TEXT, evidence_id='evidence-1')],
			)

	async def test_claim_without_alignment_is_rejected(self):
		reranker = SemanticEvidenceReranker(FakeChatModel(_score_each({'evidence-1': 0.5})))

		with pytest.raises(EvidenceRerankingError, match='no alignment entry'):
			await reranker.rerank(
				claim_set=_claim_set(STARS_CLAIM, 'A second claim about stars.'),
				alignment_result=_alignment({'claim-1': [('evidence-1', 0.5)]}),
				evidence_nodes=[_node(1, HIGH_STAR_TEXT, evidence_id='evidence-1')],
			)

	async def test_duplicate_claim_alignment_is_rejected(self):
		reranker = SemanticEvidenceReranker(FakeChatModel(_never_called()))
		match = EvidenceMatch(evidence_id='evidence-1', score=0.5, rank=1)
		alignment = AlignmentResult(
			task_id='task-1',
			alignments=[ClaimAlignment(claim_id='claim-1', matches=[match]), ClaimAlignment(claim_id='claim-1', matches=[match])],
		)

		with pytest.raises(EvidenceRerankingError, match='more than once'):
			await reranker.rerank(
				claim_set=_claim_set(STARS_CLAIM),
				alignment_result=alignment,
				evidence_nodes=[_node(1, HIGH_STAR_TEXT, evidence_id='evidence-1')],
			)

	async def test_duplicate_evidence_id_in_the_node_list_is_rejected(self):
		reranker = SemanticEvidenceReranker(FakeChatModel(_never_called()))

		with pytest.raises(EvidenceRerankingError, match='more than once'):
			await reranker.rerank(
				claim_set=_claim_set(STARS_CLAIM),
				alignment_result=_alignment({'claim-1': [('evidence-1', 0.5)]}),
				evidence_nodes=[_node(1, 'a', evidence_id='evidence-1'), _node(2, 'b', evidence_id='evidence-1')],
			)

	async def test_unknown_evidence_id_from_the_model_is_rejected(self):
		reranker = SemanticEvidenceReranker(FakeChatModel(_reply_with(('evidence-invented', 0.9))))
		claim_set, alignment, nodes = _three_candidates()

		with pytest.raises(EvidenceRerankingError, match='unknown evidence_id'):
			await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

	async def test_duplicate_evidence_id_from_the_model_is_rejected(self):
		reranker = SemanticEvidenceReranker(FakeChatModel(_reply_with(('evidence-high', 0.9), ('evidence-high', 0.8))))
		claim_set, alignment, nodes = _three_candidates()

		with pytest.raises(EvidenceRerankingError, match='duplicate evidence_id'):
			await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

	async def test_omitted_candidate_is_rejected(self):
		reranker = SemanticEvidenceReranker(FakeChatModel(_reply_with(('evidence-high', 0.9), ('evidence-low', 0.8))))
		claim_set, alignment, nodes = _three_candidates()

		with pytest.raises(EvidenceRerankingError, match='omitted'):
			await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

	async def test_extra_candidate_beyond_the_prompt_is_rejected(self):
		reranker = SemanticEvidenceReranker(
			FakeChatModel(
				_reply_with(('evidence-high', 0.9), ('evidence-language', 0.5), ('evidence-low', 0.8), ('evidence-bonus', 0.1))
			)
		)
		claim_set, alignment, nodes = _three_candidates()

		with pytest.raises(EvidenceRerankingError, match='unknown evidence_id'):
			await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

	async def test_non_structured_completion_is_rejected(self):
		reranker = SemanticEvidenceReranker(FakeChatModel(lambda prompt, index: 'the candidates look fine to me'))
		claim_set, alignment, nodes = _three_candidates()

		with pytest.raises(EvidenceRerankingError) as excinfo:
			await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		assert 'expected RawSemanticReranking' in str(excinfo.value)
		assert 'look fine' not in str(excinfo.value)


class TestFailureSemantics:
	async def test_provider_failure_keeps_the_cause_but_hides_its_text(self):
		class _FailingModel(FakeChatModel):
			async def ainvoke(self, messages, output_format=None, **kwargs):
				self.calls.append({'messages': messages, 'output_format': output_format, 'kwargs': kwargs})
				raise RuntimeError('secret-api-key=abc123 USER_PRIVATE_ANSWER')

		claim_set, alignment, nodes = _three_candidates()

		with pytest.raises(EvidenceRerankingError) as excinfo:
			await SemanticEvidenceReranker(_FailingModel(_never_called())).rerank(
				claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes
			)

		message = str(excinfo.value)
		for secret in ('abc123', 'secret-api-key', 'USER_PRIVATE_ANSWER', STARS_CLAIM, HIGH_STAR_TEXT):
			assert secret not in message
		assert isinstance(excinfo.value.__cause__, RuntimeError)
		assert 'abc123' in str(excinfo.value.__cause__)

	async def test_integrity_error_messages_do_not_contain_evidence_text(self):
		reranker = SemanticEvidenceReranker(FakeChatModel(_reply_with(('evidence-invented', 0.9))))
		claim_set, alignment, nodes = _three_candidates()

		with pytest.raises(EvidenceRerankingError) as excinfo:
			await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		assert HIGH_STAR_TEXT not in str(excinfo.value)
		assert LANGUAGE_TEXT not in str(excinfo.value)


class TestPromptContract:
	def test_system_prompt_defines_semantic_relevance_only(self):
		for phrase in (
			'You are a semantic evidence reranker.',
			'This is relevance, not verification:',
			'Do not decide whether the claim is true, false, supported, contradicted',
			'no candidate may be dropped because of them',
			'Contradicting evidence can be highly relevant',
			'is only weakly relevant',
			'Copy each evidence_id verbatim',
			'Return exactly one score for every candidate',
			'Do not explain, and do not show your reasoning.',
		):
			assert phrase in _SEMANTIC_RERANKING_SYSTEM_PROMPT

	def test_verdict_words_only_appear_inside_prohibitions_or_examples(self):
		sentences = [
			line
			for line in _SEMANTIC_RERANKING_SYSTEM_PROMPT.splitlines()
			if re.search(r'\b(supported|contradicted|true|false)\b', line)
		]

		assert sentences, 'the prompt is expected to name the verdicts it forbids'
		for sentence in sentences:
			assert 'Do not decide' in sentence or 'highly relevant' in sentence

	async def test_request_uses_the_structured_output_schema(self):
		llm = FakeChatModel(_score_each({'evidence-high': 0.5, 'evidence-language': 0.5, 'evidence-low': 0.5}))
		claim_set, alignment, nodes = _three_candidates()

		await SemanticEvidenceReranker(llm).rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		assert llm.calls[0]['output_format'] is RawSemanticReranking
		assert isinstance(llm.calls[0]['messages'][0].content, str)

	async def test_user_prompt_carries_claim_title_and_content(self):
		llm = FakeChatModel(_score_each({'evidence-high': 0.5, 'evidence-language': 0.5, 'evidence-low': 0.5}))
		claim_set, alignment, nodes = _three_candidates()

		await SemanticEvidenceReranker(llm).rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		prompt = llm.prompts()[0]
		assert STARS_CLAIM in prompt
		assert HIGH_STAR_TEXT in prompt
		assert 'title: GitHub' in prompt

	async def test_long_candidate_text_is_clipped_deterministically(self):
		long_text = 'stars ' * 2000
		nodes = [_node(1, long_text, evidence_id='evidence-long'), _node(2, LANGUAGE_TEXT, evidence_id='evidence-language')]
		claim_set = _claim_set(STARS_CLAIM)
		alignment = _alignment({'claim-1': [('evidence-long', 0.5), ('evidence-language', 0.4)]})

		prompts = []
		for _ in range(2):
			llm = FakeChatModel(_score_each({'evidence-long': 0.5, 'evidence-language': 0.5}))
			await SemanticEvidenceReranker(llm, max_evidence_chars=40).rerank(
				claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes
			)
			prompts.append(llm.prompts()[0])

		assert prompts[0] == prompts[1]
		assert _TRUNCATION_MARKER in prompts[0]
		assert long_text not in prompts[0]
		assert 'stars stars stars' in prompts[0]


class TestRelevanceIsNotSupport:
	async def test_conflicting_evidence_keeps_a_high_semantic_score(self):
		# Both candidates state a star count, so both stay highly relevant even though one disagrees.
		nodes = [_node(1, HIGH_STAR_TEXT, evidence_id='evidence-high'), _node(2, LOW_STAR_TEXT, evidence_id='evidence-low')]
		claim_set = _claim_set(STARS_CLAIM)
		alignment = _alignment({'claim-1': [('evidence-high', 0.6), ('evidence-low', 0.6)]})
		reranker = SemanticEvidenceReranker(FakeChatModel(_score_each({'evidence-high': 0.98, 'evidence-low': 0.96})))

		result = await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		semantic = {match.evidence_id: match.semantic_score for match in result.rerankings[0].matches}
		assert semantic == pytest.approx({'evidence-high': 0.98, 'evidence-low': 0.96})
		assert min(semantic.values()) > 0.9
		assert [match.evidence_id for match in result.rerankings[0].matches] == ['evidence-high', 'evidence-low']


class TestPurity:
	async def test_inputs_are_not_mutated(self):
		claim_set, alignment, nodes = _three_candidates()
		claims_before = [claim.model_dump() for claim in claim_set.claims]
		alignment_before = alignment.model_dump()
		nodes_before = [node.model_dump() for node in nodes]
		reranker = SemanticEvidenceReranker(
			FakeChatModel(_score_each({'evidence-high': 0.5, 'evidence-language': 0.5, 'evidence-low': 0.5}))
		)

		await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		assert [claim.model_dump() for claim in claim_set.claims] == claims_before
		assert alignment.model_dump() == alignment_before
		assert [node.model_dump() for node in nodes] == nodes_before

	async def test_claim_ids_are_copied_verbatim_and_claim_order_is_kept(self):
		nodes = [_node(1, HIGH_STAR_TEXT, evidence_id='evidence-high')]
		claim_set = _claim_set('First claim about stars.', 'Second claim about stars.')
		alignment = _alignment({'claim-2': [('evidence-high', 0.5)], 'claim-1': [('evidence-high', 0.5)]})
		reranker = SemanticEvidenceReranker(FakeChatModel(_score_each({'evidence-high': 0.5})))

		result = await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		assert [reranking.claim_id for reranking in result.rerankings] == ['claim-1', 'claim-2']

	async def test_result_round_trips_through_json(self):
		claim_set, alignment, nodes = _three_candidates()
		reranker = SemanticEvidenceReranker(
			FakeChatModel(_score_each({'evidence-high': 0.5, 'evidence-language': 0.5, 'evidence-low': 0.5}))
		)

		result = await reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		assert RerankingResult.model_validate_json(result.model_dump_json()) == result


class TestPipelineComposition:
	async def test_phase_4a_output_feeds_the_reranker_directly(self):
		claim_set = _claim_set(STARS_CLAIM)
		nodes = [
			_node(1, HIGH_STAR_TEXT, evidence_id='evidence-high', title='GitHub'),
			_node(2, LANGUAGE_TEXT, evidence_id='evidence-language', title='Docs'),
		]

		alignment = EvidenceAligner(top_k=2).align(claim_set=claim_set, evidence_nodes=nodes)
		result = await SemanticEvidenceReranker(
			FakeChatModel(_score_each({'evidence-high': 0.95, 'evidence-language': 0.3}))
		).rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

		assert [match.evidence_id for match in result.rerankings[0].matches] == ['evidence-high', 'evidence-language']
		assert result.rerankings[0].matches[0].semantic_score == pytest.approx(0.95)


_QWEN_MODEL = 'qwen3.8-flash'
_API_KEY_ENV = 'ALIBABA_CLOUD'
# Default matches examples/models/qwen.py. Keys are region-scoped, so ALIBABA_CLOUD_BASE_URL can
# point at the Beijing endpoint when the account was created there.
_DEFAULT_BASE_URL = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
_BASE_URL_ENV = 'ALIBABA_CLOUD_BASE_URL'
# A key that happens to sit in a local .env must not make the ordinary unit run spend real quota.
_OPT_IN_ENV = 'RUN_LLM_INTEGRATION_TESTS'
_INTEGRATION_ENABLED = bool(os.getenv(_OPT_IN_ENV)) and bool(os.getenv(_API_KEY_ENV))


@pytest.mark.integration
@pytest.mark.skipif(
	not _INTEGRATION_ENABLED,
	reason=f'makes a real {_QWEN_MODEL} call; set {_OPT_IN_ENV}=1 with {_API_KEY_ENV} configured to run it',
)
async def test_qwen38_flash_returns_a_complete_structured_reranking():
	"""One real call proving BaseChatModel + structured output works for qwen3.8-flash."""
	from browser_use.llm.openai.chat import ChatOpenAI

	nodes = [
		_node(1, 'Browser Use has 111,799 GitHub stars.', evidence_id='evidence-a', title='browser-use/browser-use'),
		_node(2, LANGUAGE_TEXT, evidence_id='evidence-b', title='Docs'),
		_node(3, 'Tomorrow will be sunny.', evidence_id='evidence-c', title='Weather'),
	]
	claim_set = _claim_set('Browser Use has more than 100,000 GitHub stars.')
	# Hand-built so all three candidates reach the model, including ones lexical scoring would drop.
	alignment = _alignment({'claim-1': [('evidence-a', 0.5), ('evidence-b', 0.4), ('evidence-c', 0.1)]})

	llm = ChatOpenAI(
		model=_QWEN_MODEL,
		api_key=os.getenv(_API_KEY_ENV),
		base_url=os.getenv(_BASE_URL_ENV, _DEFAULT_BASE_URL),
		temperature=0.0,
	)

	result = await SemanticEvidenceReranker(llm).rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)

	matches = result.rerankings[0].matches
	assert {match.evidence_id for match in matches} == {'evidence-a', 'evidence-b', 'evidence-c'}
	assert len(matches) == 3
	assert [match.rank for match in matches] == [1, 2, 3]
	assert all(0.0 <= match.semantic_score <= 1.0 for match in matches)
	assert all(0.0 <= match.lexical_score <= 1.0 for match in matches)
	by_id = {match.evidence_id: match for match in matches}
	assert by_id['evidence-a'].semantic_score > by_id['evidence-c'].semantic_score
