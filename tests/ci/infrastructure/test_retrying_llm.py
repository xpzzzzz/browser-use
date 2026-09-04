"""Tests for the Phase 9C retry wrapper.

A fake model only: no network, no Qwen, no browser. The point of these tests is that a bounded number of
extra attempts is all the wrapper does, so every one of them asserts either what the wrapper counted or
what it refused to touch. Test names carry the spec 23 letter (A-U) they pin down.

The exception messages used here deliberately contain a secret and prompt-shaped text. Several tests exist
only to prove that none of it reaches telemetry.
"""

import asyncio
import re
from typing import Any

import pytest
from pydantic import ValidationError

from browser_use.evidence import (
	ClaimExtractionError,
	ClaimExtractor,
	ClaimVerifier,
	EvidenceAligner,
	EvidenceNode,
	EvidenceRelation,
	LLMRetryPolicy,
	LLMRetryStats,
	RerankingResult,
	RetryingChatModel,
	SemanticEvidenceReranker,
	VerificationResult,
	VerificationStatus,
	stats_delta,
)
from browser_use.evidence.claim_extractor import RawClaim, RawClaimExtraction
from browser_use.evidence.reranking import RawSemanticEvidenceScore, RawSemanticReranking
from browser_use.evidence.verification import RawClaimEvidenceAssessment, RawEvidenceAssessment
from browser_use.llm.base import BaseChatModel
from browser_use.llm.messages import UserMessage
from browser_use.llm.views import ChatInvokeCompletion

SECRET = 'api-key=sk-should-never-be-logged'
CLAIM = 'Browser Use is primarily written in Python.'
EVIDENCE_TEXT = 'Browser Use is primarily written in Python.'
TASK = 'What is Browser Use written in?'
_CANDIDATE_PATTERN = re.compile(r'^evidence_id: (.+)$', re.MULTILINE)


class ScriptedChatModel:
	"""BaseChatModel-shaped fake whose first attempts raise the exceptions it was handed.

	Every attempt is recorded with the arguments it received, so a pass-through test can assert the wrapper
	forwarded exactly what it was given rather than something equivalent.
	"""

	def __init__(self, *, fail_with: tuple[Exception, ...] = (), completion: Any = None) -> None:
		self.model = 'scripted-fake'
		self.provider = 'fake'
		self.name = 'scripted-fake'
		self.model_name = 'scripted-fake'
		self._verified_api_keys = False
		self.fail_with = list(fail_with)
		self.completion = completion if completion is not None else RawClaimExtraction(claims=[RawClaim(text=CLAIM)])
		self.attempt_count = 0
		self.seen: list[tuple[Any, Any, dict[str, Any]]] = []

	async def ainvoke(self, messages, output_format=None, **kwargs) -> ChatInvokeCompletion:
		self.attempt_count += 1
		self.seen.append((messages, output_format, kwargs))
		if self.attempt_count <= len(self.fail_with):
			raise self.fail_with[self.attempt_count - 1]
		return ChatInvokeCompletion(completion=self.completion, usage=None)


class StageChatModel:
	"""Answers every structured request the three post-processing stages make.

	``fail_schemas`` names the stages whose first attempt explodes. Each stage makes one call per claim, so
	a repeated schema can only be that call's retry, and it succeeds: exactly what a transient timeout looks
	like. Counting attempts instead would let one stage spend the whole retry budget of another.
	"""

	def __init__(self, *, fail_schemas: tuple[str, ...] = (), relation: EvidenceRelation = EvidenceRelation.SUPPORTS) -> None:
		self.model = 'stage-fake'
		self.provider = 'fake'
		self.name = 'stage-fake'
		self.model_name = 'stage-fake'
		self._verified_api_keys = False
		self._pending_failures = set(fail_schemas)
		self.relation = relation
		self.attempt_count = 0
		self.calls: list[str] = []

	async def ainvoke(self, messages, output_format=None, **kwargs) -> ChatInvokeCompletion:
		schema = getattr(output_format, '__name__', '')
		self.attempt_count += 1
		self.calls.append(schema)
		if schema in self._pending_failures:
			self._pending_failures.discard(schema)
			raise TimeoutError(f'{SECRET} in {schema}')

		candidate_ids = _CANDIDATE_PATTERN.findall(messages[-1].text)
		if schema == 'RawClaimExtraction':
			completion: Any = RawClaimExtraction(claims=[RawClaim(text=CLAIM)])
		elif schema == 'RawSemanticReranking':
			completion = RawSemanticReranking(
				scores=[RawSemanticEvidenceScore(evidence_id=evidence_id, relevance_score=0.9) for evidence_id in candidate_ids]
			)
		elif schema == 'RawClaimEvidenceAssessment':
			completion = RawClaimEvidenceAssessment(
				assessments=[
					RawEvidenceAssessment(
						evidence_id=evidence_id,
						relation=self.relation,
						explanation=f'{evidence_id} states it directly.',
					)
					for evidence_id in candidate_ids
				]
			)
		else:
			raise AssertionError(f'a stage asked for an unexpected schema: {schema!r}')

		return ChatInvokeCompletion(completion=completion, usage=None)


class RecordingSleeper:
	"""Captures the delays instead of spending them, which is why the constructor accepts a sleeper."""

	def __init__(self) -> None:
		self.delays: list[float] = []

	async def __call__(self, delay: float) -> None:
		self.delays.append(delay)


def _timeout(message: str = SECRET) -> TimeoutError:
	return TimeoutError(f'{message} prompt echoed back')


def _node(evidence_id: str, text: str) -> EvidenceNode:
	return EvidenceNode(
		evidence_id=evidence_id, task_id='task-1', step_number=1, url='https://example.com', title='Example', text=text
	)


async def _stage_inputs(wrapper: RetryingChatModel):
	"""The real Phase 3 and 4A outputs a stage needs, produced by the deterministic aligner."""
	claim_set = await ClaimExtractor(wrapper).extract(task_id='task-1', task=TASK, answer=CLAIM)
	nodes = [_node('evidence-1', EVIDENCE_TEXT)]
	alignment = EvidenceAligner(top_k=5).align(claim_set=claim_set, evidence_nodes=nodes)
	return claim_set, alignment, nodes


class TestRetryPolicy:
	def test_defaults_match_the_phase_contract(self):
		policy = LLMRetryPolicy()

		assert policy.max_attempts == 3
		assert policy.initial_delay_seconds == 1.0
		assert policy.backoff_multiplier == 2.0
		assert policy.max_delay_seconds == 8.0

	@pytest.mark.parametrize('overrides', [{'max_attempts': 0}, {'initial_delay_seconds': -0.1}, {'backoff_multiplier': 0.5}])
	def test_impossible_budgets_are_refused(self, overrides: dict[str, Any]):
		with pytest.raises(ValidationError):
			LLMRetryPolicy(**overrides)

	def test_a_ceiling_below_the_first_delay_is_refused(self):
		"""A ceiling lower than the first delay would silently defeat the backoff it claims to bound."""
		with pytest.raises(ValidationError, match='max_delay_seconds'):
			LLMRetryPolicy(initial_delay_seconds=5.0, max_delay_seconds=1.0)

	def test_policy_is_frozen(self):
		policy = LLMRetryPolicy()

		with pytest.raises(ValidationError):
			policy.max_attempts = 99

	def test_delays_are_geometric_and_clamped(self):
		policy = LLMRetryPolicy(max_attempts=6, initial_delay_seconds=1.0, backoff_multiplier=3.0, max_delay_seconds=4.0)

		assert policy.retry_delays() == (1.0, 3.0, 4.0, 4.0, 4.0)

	def test_delay_is_one_based(self):
		policy = LLMRetryPolicy()

		assert policy.delay_before_retry(1) == 1.0
		assert policy.delay_before_retry(2) == 2.0
		with pytest.raises(ValueError, match='1-based'):
			policy.delay_before_retry(0)


class TestAttemptAccounting:
	async def test_first_attempt_success_costs_one_attempt(self):  # A, U
		fake = ScriptedChatModel()
		wrapper = RetryingChatModel(fake, sleep=RecordingSleeper())

		await wrapper.ainvoke([UserMessage(content='hi')], output_format=RawClaimExtraction)
		stats = wrapper.snapshot_stats()

		assert stats.logical_invocation_count == 1
		assert stats.attempt_count == 1
		assert stats.retry_count == 0
		assert stats.failed_invocation_count == 0
		# A call that never failed is not "recovered".
		assert stats.recovered_invocation_count == 0

	async def test_one_failure_then_success(self):  # B
		fake = ScriptedChatModel(fail_with=(_timeout(),))
		wrapper = RetryingChatModel(fake, sleep=RecordingSleeper())

		await wrapper.ainvoke([UserMessage(content='hi')], output_format=RawClaimExtraction)
		stats = wrapper.snapshot_stats()

		assert (stats.logical_invocation_count, stats.attempt_count, stats.retry_count) == (1, 2, 1)
		assert (stats.recovered_invocation_count, stats.failed_invocation_count) == (1, 0)

	async def test_two_failures_then_success_inside_three_attempts(self):  # C
		fake = ScriptedChatModel(fail_with=(_timeout(), _timeout()))
		wrapper = RetryingChatModel(fake, sleep=RecordingSleeper())

		await wrapper.ainvoke([UserMessage(content='hi')], output_format=RawClaimExtraction)
		stats = wrapper.snapshot_stats()

		assert (stats.logical_invocation_count, stats.attempt_count, stats.retry_count) == (1, 3, 2)
		assert stats.recovered_invocation_count == 1

	async def test_exhausted_budget_reraises_the_last_original_exception(self):  # D, T
		first, second, last = _timeout('first'), _timeout('second'), _timeout('last')
		fake = ScriptedChatModel(fail_with=(first, second, last))
		wrapper = RetryingChatModel(fake, sleep=RecordingSleeper())

		with pytest.raises(TimeoutError) as raised:
			await wrapper.ainvoke([UserMessage(content='hi')])

		# The wrapper invents no RetryError: the caller's own stage error handling still sees this object.
		assert raised.value is last
		stats = wrapper.snapshot_stats()
		assert (stats.attempt_count, stats.retry_count) == (3, 2)
		assert (stats.recovered_invocation_count, stats.failed_invocation_count) == (0, 1)

	async def test_one_attempt_never_retries(self):  # E
		fake = ScriptedChatModel(fail_with=(_timeout(), _timeout()))
		sleeper = RecordingSleeper()
		wrapper = RetryingChatModel(fake, policy=LLMRetryPolicy(max_attempts=1), sleep=sleeper)

		with pytest.raises(TimeoutError):
			await wrapper.ainvoke([UserMessage(content='hi')])

		assert fake.attempt_count == 1
		assert sleeper.delays == []
		stats = wrapper.snapshot_stats()
		assert (stats.logical_invocation_count, stats.attempt_count, stats.retry_count) == (1, 1, 0)
		assert stats.failed_invocation_count == 1

	async def test_logical_and_failed_counts_are_per_invocation_not_per_attempt(self):
		# Three invocations at the three-attempt budget, so every attempt of every call has to be loaded.
		fake = ScriptedChatModel(fail_with=tuple(_timeout() for _ in range(9)))
		wrapper = RetryingChatModel(fake, policy=LLMRetryPolicy(max_attempts=3), sleep=RecordingSleeper())

		for _ in range(3):
			with pytest.raises(TimeoutError):
				await wrapper.ainvoke([UserMessage(content='hi')])

		stats = wrapper.snapshot_stats()
		assert (stats.logical_invocation_count, stats.attempt_count, stats.retry_count) == (3, 9, 6)
		assert stats.failed_invocation_count == 3


class TestBackoff:
	async def test_sleeper_sees_the_geometric_sequence(self):  # F
		fake = ScriptedChatModel(fail_with=tuple(_timeout() for _ in range(4)))
		sleeper = RecordingSleeper()
		wrapper = RetryingChatModel(fake, policy=LLMRetryPolicy(max_attempts=5), sleep=sleeper)

		await wrapper.ainvoke([UserMessage(content='hi')])

		assert sleeper.delays == [1.0, 2.0, 4.0, 8.0]

	async def test_no_sleep_happens_before_the_ceiling_with_zero_delay(self):
		fake = ScriptedChatModel(fail_with=(_timeout(),))
		sleeper = RecordingSleeper()
		policy = LLMRetryPolicy(initial_delay_seconds=0.0, max_delay_seconds=0.0)

		await RetryingChatModel(fake, policy=policy, sleep=sleeper).ainvoke([UserMessage(content='hi')])

		assert sleeper.delays == [0.0]

	async def test_every_delay_respects_the_ceiling(self):  # G
		fake = ScriptedChatModel(fail_with=tuple(_timeout() for _ in range(5)))
		sleeper = RecordingSleeper()
		policy = LLMRetryPolicy(max_attempts=6, initial_delay_seconds=1.0, backoff_multiplier=10.0, max_delay_seconds=8.0)

		await RetryingChatModel(fake, policy=policy, sleep=sleeper).ainvoke([UserMessage(content='hi')])

		assert sleeper.delays == [1.0, 8.0, 8.0, 8.0, 8.0]

	def test_production_default_sleep_is_asyncio_sleep(self):
		"""The injected sleeper is a test seam, not something production silently waits on."""
		wrapper = RetryingChatModel(ScriptedChatModel())

		assert wrapper._sleep is asyncio.sleep


class TestPassThrough:
	async def test_messages_reach_the_model_untouched(self):  # H
		messages = [UserMessage(content='the answer under verification')]
		fake = ScriptedChatModel()
		wrapper = RetryingChatModel(fake, sleep=RecordingSleeper())

		await wrapper.ainvoke(messages, output_format=RawClaimExtraction)

		assert fake.seen[0][0] is messages

	async def test_messages_are_reidentical_on_every_retry(self):
		fake = ScriptedChatModel(fail_with=(_timeout(), _timeout()))
		wrapper = RetryingChatModel(fake, sleep=RecordingSleeper())
		messages = [UserMessage(content='same prompt')]

		await wrapper.ainvoke(messages, output_format=RawClaimExtraction)

		assert [seen[0] for seen in fake.seen] == [messages, messages, messages]

	async def test_output_format_is_forwarded_unchanged(self):  # I, S
		fake = ScriptedChatModel()
		wrapper = RetryingChatModel(fake, sleep=RecordingSleeper())

		await wrapper.ainvoke([UserMessage(content='hi')], output_format=RawClaimExtraction)

		assert fake.seen[0][1] is RawClaimExtraction

	async def test_extra_kwargs_are_forwarded_unchanged(self):  # J
		fake = ScriptedChatModel()
		wrapper = RetryingChatModel(fake, sleep=RecordingSleeper())

		await wrapper.ainvoke([UserMessage(content='hi')], output_format=RawClaimExtraction, timeout=12, seed=7)

		assert fake.seen[0][2] == {'timeout': 12, 'seed': 7}

	async def test_completion_is_returned_as_the_model_produced_it(self):  # K, S
		payload = RawClaimExtraction(claims=[RawClaim(text=CLAIM)])
		wrapper = RetryingChatModel(ScriptedChatModel(completion=payload), sleep=RecordingSleeper())

		result = await wrapper.ainvoke([UserMessage(content='hi')], output_format=RawClaimExtraction)

		assert result.completion is payload
		assert type(result.completion) is RawClaimExtraction

	async def test_a_retry_returns_the_first_real_completion(self):
		payload = RawClaimExtraction(claims=[RawClaim(text=CLAIM)])
		fake = ScriptedChatModel(fail_with=(_timeout(),), completion=payload)
		wrapper = RetryingChatModel(fake, sleep=RecordingSleeper())

		result = await wrapper.ainvoke([UserMessage(content='hi')], output_format=RawClaimExtraction)

		assert result.completion is payload
		assert fake.attempt_count == 2

	async def test_interface_is_forwarded_so_callers_see_a_chat_model(self):
		fake = ScriptedChatModel()
		wrapper = RetryingChatModel(fake)

		assert isinstance(wrapper, BaseChatModel)
		assert wrapper.model == 'scripted-fake'
		assert wrapper.provider == 'fake'
		assert wrapper.name == 'scripted-fake'
		assert wrapper.model_name == 'scripted-fake'
		assert wrapper.wrapped_model is fake
		assert wrapper.policy == LLMRetryPolicy()


class TestBaseExceptionsAreNotRetried:
	async def test_keyboard_interrupt_passes_straight_through(self):
		class Interrupting(ScriptedChatModel):
			async def ainvoke(self, messages, output_format=None, **kwargs):
				self.attempt_count += 1
				raise KeyboardInterrupt

		wrapper = RetryingChatModel(Interrupting(), sleep=RecordingSleeper())

		with pytest.raises(KeyboardInterrupt):
			await wrapper.ainvoke([UserMessage(content='hi')])

		assert wrapper.snapshot_stats().attempt_count == 1
		assert wrapper.snapshot_stats().retry_count == 0

	async def test_task_cancellation_is_not_swallowed(self):
		class Cancelling(ScriptedChatModel):
			async def ainvoke(self, messages, output_format=None, **kwargs):
				self.attempt_count += 1
				raise asyncio.CancelledError

		wrapper = RetryingChatModel(Cancelling(), sleep=RecordingSleeper())

		with pytest.raises(asyncio.CancelledError):
			await wrapper.ainvoke([UserMessage(content='hi')])

		assert wrapper.snapshot_stats().attempt_count == 1


class TestStats:
	def test_exception_counts_use_class_names_only(self):  # L
		wrapper = RetryingChatModel(ScriptedChatModel(), sleep=RecordingSleeper())
		wrapper._exception_type_counts = {'TimeoutError': 3}

		assert wrapper.snapshot_stats().exception_type_counts == {'TimeoutError': 3}

	async def test_mixed_exception_types_are_counted_separately(self):  # L
		fake = ScriptedChatModel(fail_with=(_timeout(), ValueError('bad'), _timeout(), ValueError('bad')))
		wrapper = RetryingChatModel(fake, policy=LLMRetryPolicy(max_attempts=5), sleep=RecordingSleeper())

		await wrapper.ainvoke([UserMessage(content='hi')])

		assert wrapper.snapshot_stats().exception_type_counts == {'TimeoutError': 2, 'ValueError': 2}

	async def test_a_secret_in_an_exception_message_never_reaches_telemetry(self):  # O
		fake = ScriptedChatModel(fail_with=(_timeout(f'{SECRET} and the answer under verification'),))
		wrapper = RetryingChatModel(fake, sleep=RecordingSleeper())

		await wrapper.ainvoke([UserMessage(content='hi')])
		dumped = wrapper.snapshot_stats().model_dump_json()

		assert 'api-key' not in dumped
		assert 'verification' not in dumped
		assert SECRET not in dumped
		assert 'TimeoutError' in dumped

	def test_snapshot_is_frozen_and_isolated_from_internal_state(self):  # M
		wrapper = RetryingChatModel(ScriptedChatModel(), sleep=RecordingSleeper())
		wrapper._exception_type_counts = {'TimeoutError': 1}

		snapshot = wrapper.snapshot_stats()
		snapshot.exception_type_counts['Bogus'] = 99

		with pytest.raises(ValidationError):
			snapshot.retry_count = 5
		assert wrapper.snapshot_stats().exception_type_counts == {'TimeoutError': 1}

	async def test_reset_stats_clears_the_counters_without_touching_the_model(self):
		wrapper = RetryingChatModel(ScriptedChatModel(fail_with=(_timeout(),)), sleep=RecordingSleeper())
		await wrapper.ainvoke([UserMessage(content='hi')])

		wrapper.reset_stats()

		assert wrapper.snapshot_stats() == LLMRetryStats()

	def test_two_wrappers_keep_independent_counters(self):  # N
		first = RetryingChatModel(ScriptedChatModel(fail_with=(_timeout(),)), sleep=RecordingSleeper())
		second = RetryingChatModel(ScriptedChatModel(), sleep=RecordingSleeper())
		first._exception_type_counts = {'TimeoutError': 2}

		assert second.snapshot_stats().exception_type_counts == {}
		assert first.snapshot_stats().retry_count == 0
		assert second.snapshot_stats().logical_invocation_count == 0


class TestStatsDelta:
	def test_delta_subtracts_two_snapshots(self):
		before = LLMRetryStats(
			logical_invocation_count=2, attempt_count=3, retry_count=1, exception_type_counts={'TimeoutError': 1}
		)
		after = LLMRetryStats(
			logical_invocation_count=5,
			attempt_count=9,
			retry_count=4,
			recovered_invocation_count=2,
			failed_invocation_count=1,
			exception_type_counts={'TimeoutError': 3, 'ValueError': 1},
		)

		delta = stats_delta(before, after)

		assert delta.logical_invocation_count == 3
		assert delta.attempt_count == 6
		assert delta.retry_count == 3
		assert delta.recovered_invocation_count == 2
		assert delta.failed_invocation_count == 1
		assert delta.exception_type_counts == {'TimeoutError': 2, 'ValueError': 1}

	def test_identical_snapshots_delta_to_zero(self):
		snapshot = LLMRetryStats(logical_invocation_count=4, attempt_count=7, retry_count=3)

		assert stats_delta(snapshot, snapshot) == LLMRetryStats()

	def test_a_backwards_counter_names_the_field(self):
		"""Counters only increase, so this can only mean snapshots from different wrappers or out of order."""
		before = LLMRetryStats(retry_count=2)
		after = LLMRetryStats(retry_count=1)

		with pytest.raises(ValueError, match='retry_count went backwards'):
			stats_delta(before, after)

	def test_a_backwards_exception_type_count_is_refused(self):
		with pytest.raises(ValueError, match='ValueError'):
			stats_delta(
				LLMRetryStats(exception_type_counts={'ValueError': 4}), LLMRetryStats(exception_type_counts={'ValueError': 1})
			)

	async def test_a_run_delta_ignores_the_calls_of_earlier_runs(self):
		"""The benchmark shares one wrapper, so a per-run record must be a difference, not a total."""
		wrapper = RetryingChatModel(
			ScriptedChatModel(fail_with=tuple(_timeout() for _ in range(9))),
			policy=LLMRetryPolicy(max_attempts=3),
			sleep=RecordingSleeper(),
		)

		before_first = wrapper.snapshot_stats()
		for _ in range(2):
			with pytest.raises(TimeoutError):
				await wrapper.ainvoke([UserMessage(content='hi')])
		first_run = stats_delta(before_first, wrapper.snapshot_stats())
		second_before = wrapper.snapshot_stats()

		with pytest.raises(TimeoutError):
			await wrapper.ainvoke([UserMessage(content='hi')])

		second_run = stats_delta(second_before, wrapper.snapshot_stats())
		assert (first_run.logical_invocation_count, second_run.logical_invocation_count) == (2, 1)
		assert (first_run.attempt_count, second_run.attempt_count) == (6, 3)


class TestStageCompatibility:
	"""The three stages must accept the wrapper without knowing it exists (spec 21, 22)."""

	async def test_claim_extractor_accepts_the_wrapper(self):  # P
		fake = StageChatModel(fail_schemas=('RawClaimExtraction',))
		wrapper = RetryingChatModel(fake, sleep=RecordingSleeper())

		claim_set = await ClaimExtractor(wrapper).extract(task_id='task-1', task=TASK, answer=CLAIM)

		assert [claim.text for claim in claim_set.claims] == [CLAIM]
		assert wrapper.snapshot_stats().recovered_invocation_count == 1

	async def test_reranker_accepts_the_wrapper(self):  # Q
		wrapper = RetryingChatModel(
			StageChatModel(fail_schemas=('RawClaimExtraction', 'RawSemanticReranking')), sleep=RecordingSleeper()
		)
		claim_set, alignment, nodes = await _stage_inputs(wrapper)

		result = await SemanticEvidenceReranker(wrapper).rerank(
			claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes
		)

		assert isinstance(result, RerankingResult)
		assert [match.evidence_id for match in result.rerankings[0].matches] == ['evidence-1']
		# One attempt recovered for the extraction and one for the reranking itself.
		assert wrapper.snapshot_stats().recovered_invocation_count == 2

	async def test_verifier_accepts_the_wrapper(self):  # R
		wrapper = RetryingChatModel(
			StageChatModel(fail_schemas=('RawClaimExtraction', 'RawSemanticReranking', 'RawClaimEvidenceAssessment')),
			sleep=RecordingSleeper(),
		)
		claim_set, alignment, nodes = await _stage_inputs(wrapper)
		reranking = await SemanticEvidenceReranker(wrapper).rerank(
			claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes
		)

		verification = await ClaimVerifier(wrapper).verify(claim_set=claim_set, reranking_result=reranking, evidence_nodes=nodes)

		assert isinstance(verification, VerificationResult)
		assert verification.verifications[0].status is VerificationStatus.SUPPORTED
		assert wrapper.snapshot_stats().recovered_invocation_count == 3

	async def test_a_refuting_page_still_verifies_through_the_wrapper(self):
		wrapper = RetryingChatModel(StageChatModel(relation=EvidenceRelation.CONTRADICTS), sleep=RecordingSleeper())
		claim_set, alignment, nodes = await _stage_inputs(wrapper)
		reranking = await SemanticEvidenceReranker(wrapper).rerank(
			claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes
		)

		verification = await ClaimVerifier(wrapper).verify(claim_set=claim_set, reranking_result=reranking, evidence_nodes=nodes)

		assert verification.verifications[0].status is VerificationStatus.CONTRADICTED
		assert wrapper.snapshot_stats().recovered_invocation_count == 0

	async def test_the_stage_error_still_surfaces_when_the_budget_runs_out(self):
		"""Retry must not soften strictness: ClaimExtractionError still owns the failure (spec 20)."""
		fake = ScriptedChatModel(fail_with=(_timeout(),) * 3)
		wrapper = RetryingChatModel(fake, sleep=RecordingSleeper())

		with pytest.raises(ClaimExtractionError):
			await ClaimExtractor(wrapper).extract(task_id='task-1', task=TASK, answer=CLAIM)

		stats = wrapper.snapshot_stats()
		assert (stats.attempt_count, stats.retry_count, stats.failed_invocation_count) == (3, 2, 1)
