"""Tests for the fixed-input replay benchmark: fixtures, signatures, run records and aggregates.

No browser, no network and no live model. The replay tests drive the real Phases 3 to 7 through a fake that
answers the three structured requests those stages make, because a harness tested only against a fake
pipeline would not notice it drifting away from what a live run does.

The fake's exception messages deliberately contain secret-shaped text and a slice of the prompt. A provider
error echoes the request it was given, and that request is the answer under verification plus scraped page
text, so several tests here exist only to prove that none of it reaches an artifact someone might share.
"""

import json
import re
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from browser_use.evidence import (
	Claim,
	ClaimExtractor,
	ClaimSet,
	ClaimVerification,
	ClaimVerifier,
	EvidenceAligner,
	EvidenceNode,
	EvidenceOrganizer,
	EvidenceRelation,
	EvidenceReplayBenchmarkResult,
	EvidenceReportBuilder,
	LLMRetryPolicy,
	MarkdownReportRenderer,
	PipelineStage,
	ReplayBenchmarkError,
	ReplayBenchmarkSummary,
	ReplayComparison,
	ReplayFixture,
	ReplayRunResult,
	RetryingChatModel,
	SemanticEvidenceReranker,
	VerificationResult,
	VerificationStatus,
	WebEvidencePipeline,
	claim_signature,
	compare_replay_results,
	load_replay_fixtures,
	normalize_claim_text,
	run_replay,
	select_fixtures,
	status_signature,
	summarize_replay_runs,
)
from browser_use.evidence.claim_extractor import RawClaim, RawClaimExtraction
from browser_use.evidence.replay_benchmark import ordered_claims
from browser_use.evidence.reranking import RawSemanticEvidenceScore, RawSemanticReranking
from browser_use.evidence.verification import RawClaimEvidenceAssessment, RawEvidenceAssessment
from browser_use.llm.views import ChatInvokeCompletion

TASK_ID = '06a9a554-21ee-79cc-8000-b39424bedc96'
OTHER_TASK_ID = '06a9a616-0cf4-7e98-8000-b22567d8b249'
CLAIM = 'example.com is reserved for documentation purposes.'
SECOND_CLAIM = 'example.com is managed by IANA.'
ANSWER = f'{CLAIM} {SECOND_CLAIM}'
NODE_TEXT = f'{CLAIM} {SECOND_CLAIM}'
SECRET = 'sk-secret-prompt-echo'
_CANDIDATE_PATTERN = re.compile(r'^evidence_id: (.+)$', re.MULTILINE)
DATASET = Path('benchmarks/webevidence/replay/manifest.jsonl')


class TransientProviderError(RuntimeError):
	"""A provider hiccup, named like one so the recorded type name means something on its own."""


class StageChatModel:
	"""Answers every structured request the three model stages make, and fails on scheduled attempts.

	``fail_on`` holds 1-based attempt numbers rather than schema names. One wrapper serves several replays in
	a test, so "the fifth attempt of this benchmark exploded" is the statement a test needs to make, and
	attempt numbers are also what stops one call's retry budget from paying for another call's failure.
	"""

	def __init__(
		self,
		*,
		claim_texts: Sequence[str] | None = None,
		fail_on: set[int] | frozenset[int] = frozenset(),
		relation: EvidenceRelation = EvidenceRelation.SUPPORTS,
	) -> None:
		self.model = 'stage-fake'
		self.provider = 'fake'
		self.name = 'stage-fake'
		self.model_name = 'stage-fake'
		self._verified_api_keys = False
		self.claim_texts = list(claim_texts or [CLAIM])
		self.fail_on = set(fail_on)
		self.relation = relation
		self.attempt_count = 0
		self.calls: list[str] = []

	async def ainvoke(self, messages, output_format=None, **kwargs) -> ChatInvokeCompletion:
		self.attempt_count += 1
		if self.attempt_count in self.fail_on:
			raise TransientProviderError(f'attempt {self.attempt_count} of {SECRET} refused for {messages[-1].text[:40]}')

		schema = getattr(output_format, '__name__', '')
		self.calls.append(schema)
		candidate_ids = _CANDIDATE_PATTERN.findall(messages[-1].text)
		completion: Any
		if schema == 'RawClaimExtraction':
			completion = RawClaimExtraction(claims=[RawClaim(text=text) for text in self.claim_texts])
		elif schema == 'RawSemanticReranking':
			completion = RawSemanticReranking(
				scores=[RawSemanticEvidenceScore(evidence_id=evidence_id, relevance_score=0.9) for evidence_id in candidate_ids]
			)
		elif schema == 'RawClaimEvidenceAssessment':
			completion = RawClaimEvidenceAssessment(
				assessments=[
					RawEvidenceAssessment(
						evidence_id=evidence_id, relation=self.relation, explanation=f'{evidence_id} states it.'
					)
					for evidence_id in candidate_ids
				]
			)
		else:
			raise AssertionError(f'a stage asked for an unexpected schema: {schema!r}')
		return ChatInvokeCompletion(completion=completion, usage=None)


async def _no_sleep(_seconds: float) -> None:
	"""Stands in for asyncio.sleep, so a retry test asserts the budget was spent without waiting for it."""


def _node(evidence_id: str = 'ev-1', *, task_id: str = TASK_ID, text: str = NODE_TEXT, **overrides: Any) -> EvidenceNode:
	"""An explicitly identified node: a generated ``evidence_id`` is a fixture bug, so these supply one."""
	fields: dict[str, Any] = {
		'evidence_id': evidence_id,
		'task_id': task_id,
		'step_number': 1,
		'url': 'https://example.com/',
		'title': 'Example Domain',
		'text': text,
		'screenshot_path': None,
	}
	fields.update(overrides)
	return EvidenceNode(**fields)


def _fixture(**overrides: Any) -> ReplayFixture:
	fixture: dict[str, Any] = {
		'fixture_id': 'example-heading',
		'task_id': TASK_ID,
		'task': 'Report what example.com is for.',
		'answer': ANSWER,
		'evidence_nodes': [_node()],
	}
	fixture.update(overrides)
	return ReplayFixture(**fixture)


def _pipeline(llm: Any) -> WebEvidencePipeline:
	"""The real Phases 3 to 7, wired the way the live scripts wire them."""
	return WebEvidencePipeline(
		claim_extractor=ClaimExtractor(llm),
		aligner=EvidenceAligner(top_k=5),
		reranker=SemanticEvidenceReranker(llm),
		verifier=ClaimVerifier(llm),
		organizer=EvidenceOrganizer(),
		report_builder=EvidenceReportBuilder(),
		markdown_renderer=MarkdownReportRenderer(),
	)


def _replayed(fixture: ReplayFixture, *, model: StageChatModel, max_attempts: int = 3, repeat_index: int = 1):
	"""One replay through a wrapper of its own, with the clock fixed so elapsed_seconds is assertable."""
	wrapper = RetryingChatModel(model, policy=LLMRetryPolicy(max_attempts=max_attempts), sleep=_no_sleep)
	return run_replay(
		fixture,
		pipeline=_pipeline(wrapper),
		repeat_index=repeat_index,
		max_attempts=max_attempts,
		stats=wrapper.snapshot_stats,
		clock=iter([0.0, 4.0]).__next__,
	)


def _status_counts(statuses: Sequence[str]) -> dict[str, int]:
	"""All six status counts for a status sequence, keyed by the field that holds each one."""
	return {f'{status.name.lower()}_claim_count': statuses.count(status.value) for status in VerificationStatus}


def _completed_run(**overrides: Any) -> ReplayRunResult:
	"""A replay that reached a report: one claim judged SUPPORTED, three calls, all on the first attempt."""
	run: dict[str, Any] = {
		'fixture_id': 'f1',
		'repeat_index': 1,
		'max_attempts': 3,
		'pipeline_completed': True,
		'elapsed_seconds': 2.0,
		'claim_count': 1,
		'claim_signature': [normalize_claim_text(CLAIM)],
		'status_signature': [VerificationStatus.SUPPORTED.value],
		'postprocess_llm_logical_calls': 3,
		'postprocess_llm_attempts': 3,
	}
	run.update(overrides)
	if isinstance(run['status_signature'], list):
		# The six counts are the signature restated, so a test states verdicts and not arithmetic.
		run.update(_status_counts(run['status_signature']))
	return ReplayRunResult(**run)


def _failed_run(**overrides: Any) -> ReplayRunResult:
	"""A replay that stopped at claim extraction after one attempt, having measured nothing else."""
	run: dict[str, Any] = {
		'fixture_id': 'f1',
		'repeat_index': 1,
		'max_attempts': 1,
		'pipeline_completed': False,
		'failure_stage': PipelineStage.CLAIM_EXTRACTION,
		'failure_type': f'WebEvidencePipelineError:{PipelineStage.CLAIM_EXTRACTION.value}',
		'elapsed_seconds': 1.0,
		'postprocess_llm_logical_calls': 1,
		'postprocess_llm_attempts': 1,
		'postprocess_llm_failed_calls': 1,
		'exception_type_counts': {'TransientProviderError': 1},
	}
	run.update(overrides)
	return ReplayRunResult(**run)


def _run_for(
	fixture_id: str,
	repeat_index: int,
	*,
	completed: bool = True,
	claims: list[str] | None = None,
	statuses: list[str] | None = None,
	max_attempts: int = 3,
	elapsed_seconds: float = 2.0,
	**overrides: Any,
) -> ReplayRunResult:
	"""A replay record for the aggregation tests, where only completion and the signatures matter.

	``claim_count`` is always derived from the claim signature here: these tests are about aggregation, and a
	record whose own counts and signatures disagree is rejected by the model before it can be summed.
	"""
	if not completed:
		return _failed_run(
			fixture_id=fixture_id,
			repeat_index=repeat_index,
			max_attempts=max_attempts,
			elapsed_seconds=elapsed_seconds,
			**overrides,
		)

	claim_signature_value = claims or [f'claim {fixture_id}']
	status_signature_value = statuses or [VerificationStatus.SUPPORTED.value] * len(claim_signature_value)
	if len(claim_signature_value) != len(status_signature_value):
		raise AssertionError('a replay record needs exactly one status per claim')
	return _completed_run(
		fixture_id=fixture_id,
		repeat_index=repeat_index,
		max_attempts=max_attempts,
		elapsed_seconds=elapsed_seconds,
		claim_count=len(claim_signature_value),
		claim_signature=claim_signature_value,
		status_signature=status_signature_value,
		**overrides,
	)


def _fixture_payload(fixture_id: str, **overrides: Any) -> dict[str, Any]:
	payload: dict[str, Any] = {
		'fixture_id': fixture_id,
		'task_id': TASK_ID,
		'task': 't',
		'answer': ANSWER,
		'evidence_nodes': [_node().model_dump(mode='json')],
	}
	payload.update(overrides)
	return payload


def _write_dataset(tmp_path: Path, lines: list[str]) -> Path:
	path = tmp_path / 'manifest.jsonl'
	path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
	return path


class TestReplayFixture:
	def test_a_frozen_input_loads_with_its_defaults(self):
		fixture = _fixture()

		assert fixture.tags == []
		assert fixture.description == ''
		assert [node.evidence_id for node in fixture.evidence_nodes] == ['ev-1']

	def test_blank_identifiers_are_refused(self):
		for overrides in ({'fixture_id': ''}, {'fixture_id': '   '}, {'task_id': '\n\t'}):
			with pytest.raises(Exception):
				_fixture(**overrides)

	def test_duplicate_evidence_id_is_refused(self):
		with pytest.raises(ReplayBenchmarkError, match='duplicate evidence_id'):
			_fixture(evidence_nodes=[_node('ev-1'), _node('ev-1', text='a second capture with the same id')])

	def test_generated_evidence_id_is_refused(self):
		"""A regenerated id makes two repeats of a fixture different inputs, and evidence ids key the results."""
		with pytest.raises(ReplayBenchmarkError, match='generated evidence_id'):
			_fixture(evidence_nodes=[EvidenceNode(task_id=TASK_ID, step_number=1, url='https://example.com/', text='t')])

	def test_evidence_from_another_task_is_refused(self):
		with pytest.raises(ReplayBenchmarkError, match='another task_id'):
			_fixture(evidence_nodes=[_node('ev-1'), _node('ev-2', task_id=OTHER_TASK_ID)])

	def test_empty_evidence_is_a_legal_fixture(self):
		"""Nothing captured is a finding the pipeline reports as NO_EVIDENCE, not a broken fixture."""
		assert _fixture(evidence_nodes=[]).evidence_nodes == []


class TestFixtureLoader:
	def test_jsonl_loads_in_file_order(self, tmp_path):
		path = _write_dataset(tmp_path, [json.dumps(_fixture_payload('b-first')), json.dumps(_fixture_payload('a-second'))])

		assert [fixture.fixture_id for fixture in load_replay_fixtures(path)] == ['b-first', 'a-second']

	def test_a_trailing_blank_line_is_not_a_fixture(self, tmp_path):
		path = tmp_path / 'manifest.jsonl'
		path.write_text(json.dumps(_fixture_payload('only')) + '\n\n   \n', encoding='utf-8')

		assert len(load_replay_fixtures(path)) == 1

	def test_a_malformed_line_is_reported_with_its_line_number(self, tmp_path):
		path = _write_dataset(tmp_path, [json.dumps(_fixture_payload('first')), 'not json at all'])

		with pytest.raises(ReplayBenchmarkError, match='line 2.*ValidationError'):
			load_replay_fixtures(path)

	def test_a_thawed_fixture_is_reported_with_its_line_number(self, tmp_path):
		payload = _fixture_payload('bad')
		payload['evidence_nodes'].append(_node('ev-1').model_dump(mode='json'))
		path = _write_dataset(tmp_path, [json.dumps(_fixture_payload('good')), json.dumps(payload)])

		with pytest.raises(ReplayBenchmarkError, match='line 2.*duplicate evidence_id'):
			load_replay_fixtures(path)

	def test_a_bad_line_names_no_line_content(self, tmp_path):
		"""The line number and the exception type only: the offending value here would be page text."""
		payload = _fixture_payload('bad', answer=f'{SECRET} and every word of the captured page')
		del payload['task']
		path = _write_dataset(tmp_path, [json.dumps(payload)])

		with pytest.raises(ReplayBenchmarkError) as excinfo:
			load_replay_fixtures(path)
		assert SECRET not in str(excinfo.value)

	def test_duplicate_fixture_id_names_both_lines(self, tmp_path):
		path = _write_dataset(
			tmp_path,
			[json.dumps(_fixture_payload('same')), json.dumps(_fixture_payload('other')), json.dumps(_fixture_payload('same'))],
		)

		with pytest.raises(ReplayBenchmarkError, match="'same'.*lines 1 and 3"):
			load_replay_fixtures(path)

	def test_an_empty_dataset_is_refused(self, tmp_path):
		path = tmp_path / 'manifest.jsonl'
		path.write_text('\n   \n', encoding='utf-8')

		with pytest.raises(ReplayBenchmarkError, match='contains no fixtures'):
			load_replay_fixtures(path)

	def test_an_unreadable_dataset_says_so(self, tmp_path):
		with pytest.raises(ReplayBenchmarkError, match='Cannot read replay fixture dataset'):
			load_replay_fixtures(tmp_path / 'nope.jsonl')

	def test_seeded_dataset_loads_and_covers_the_planned_shapes(self):
		fixtures = load_replay_fixtures(DATASET)

		by_id = {fixture.fixture_id: fixture for fixture in fixtures}
		assert len(fixtures) == len(by_id) == 5
		assert by_id.keys() == {
			'example-heading-single-claim',
			'python-stdlib-three-claims',
			'github-primary-language-claims',
			'example-purpose-claim-extraction-failure',
			'heading-plus-manager-reranking-failure',
		}

		tags = {tag for fixture in fixtures for tag in fixture.tags}
		assert {
			'claim-count:1',
			'claim-count:3',
			'high-claim-count',
			'prior-failure:CLAIM_EXTRACTION',
			'prior-failure:SEMANTIC_RERANKING',
		} <= tags
		assert all(fixture.description.strip() for fixture in fixtures)

	def test_seeded_dataset_is_sanitized_and_task_consistent(self):
		for fixture in load_replay_fixtures(DATASET):
			assert fixture.evidence_nodes, f'{fixture.fixture_id} would replay an empty capture'
			assert fixture.task.strip() and fixture.answer.strip()
			ids = [node.evidence_id for node in fixture.evidence_nodes]
			assert len(set(ids)) == len(ids)
			for node in fixture.evidence_nodes:
				# The captures had screenshots; the artifact that gets shared must not point at them.
				assert node.screenshot_path is None
				assert node.task_id == fixture.task_id


class TestSelectFixtures:
	def test_narrowing_keeps_file_order_and_dedupes_repeated_names(self):
		fixtures = load_replay_fixtures(DATASET)

		selected = select_fixtures(
			fixtures, ['heading-plus-manager-reranking-failure', 'example-heading-single-claim', 'example-heading-single-claim']
		)

		assert [fixture.fixture_id for fixture in selected] == [
			'example-heading-single-claim',
			'heading-plus-manager-reranking-failure',
		]

	def test_a_typo_says_so(self):
		with pytest.raises(ReplayBenchmarkError, match='Unknown --fixture value'):
			select_fixtures(load_replay_fixtures(DATASET), ['nope'])

	def test_nothing_selected_is_refused(self):
		with pytest.raises(ReplayBenchmarkError, match='empty'):
			select_fixtures([], None)


class TestSignatures:
	def test_formatting_differences_are_not_claim_differences(self):
		assert normalize_claim_text('  Example   Domain\n\tis here.  ') == 'example domain is here.'

	def test_compatibility_characters_are_folded(self):
		"""A page that writes 'ﬁxed' and a model that writes 'fixed' state the same claim."""
		assert normalize_claim_text('A ﬁxed ½ answer') == normalize_claim_text('A fixed 1⁄2 answer')

	def test_claim_signature_holds_no_identifiers(self):
		"""Claim ids are generated fresh every run, so an id-based comparison would always disagree."""
		first = ClaimSet(task_id=TASK_ID, task='t', answer='a', claims=[Claim(text=CLAIM, order=1)])
		second = ClaimSet(task_id=TASK_ID, task='t', answer='a', claims=[Claim(text=CLAIM, order=1)])

		assert first.claims[0].claim_id != second.claims[0].claim_id
		assert claim_signature(first) == claim_signature(second)

	def test_claim_order_not_list_position_fixes_the_sequence(self):
		claim_set = ClaimSet(
			task_id=TASK_ID,
			task='t',
			answer='a',
			claims=[Claim(text='second claim.', order=2), Claim(text='first claim.', order=1)],
		)

		assert [claim.order for claim in ordered_claims(claim_set)] == [1, 2]
		assert claim_signature(claim_set) == ['first claim.', 'second claim.']

	def test_status_sequence_follows_claims_not_response_order(self):
		claims = [Claim(text='first claim.', order=1), Claim(text='second claim.', order=2)]
		claim_set = ClaimSet(task_id=TASK_ID, task='t', answer='a', claims=claims)
		verification_result = VerificationResult(
			task_id=TASK_ID,
			verifications=[
				ClaimVerification(claim_id=claims[1].claim_id, status=VerificationStatus.UNSUPPORTED),
				ClaimVerification(claim_id=claims[0].claim_id, status=VerificationStatus.SUPPORTED),
			],
		)

		assert status_signature(claim_set, verification_result) == ['SUPPORTED', 'UNSUPPORTED']

	def test_a_claim_with_no_status_is_an_error_not_a_gap(self):
		claim = Claim(text='only claim.', order=1)
		verification_result = VerificationResult(
			task_id=TASK_ID, verifications=[ClaimVerification(claim_id=claim.claim_id, status=VerificationStatus.SUPPORTED)]
		)

		with pytest.raises(ReplayBenchmarkError, match='no verification status'):
			status_signature(
				ClaimSet(task_id=TASK_ID, task='t', answer='a', claims=[Claim(text='a different claim.', order=1)]),
				verification_result,
			)

	def test_a_claim_verified_twice_is_an_error(self):
		claim = Claim(text='only claim.', order=1)
		claim_set = ClaimSet(task_id=TASK_ID, task='t', answer='a', claims=[claim])
		verification_result = VerificationResult(
			task_id=TASK_ID,
			verifications=[
				ClaimVerification(claim_id=claim.claim_id, status=VerificationStatus.SUPPORTED),
				ClaimVerification(claim_id=claim.claim_id, status=VerificationStatus.UNSUPPORTED),
			],
		)

		with pytest.raises(ReplayBenchmarkError, match='twice'):
			status_signature(claim_set, verification_result)


class TestReplayRunResult:
	def test_a_completed_replay_needs_no_failure_fields(self):
		run = _completed_run()

		assert run.pipeline_completed and run.failure_stage is None and run.failure_type is None
		assert run.exception_type_counts == {}

	def test_a_failed_replay_reports_no_claim_results(self):
		run = _failed_run()

		assert run.claim_count is None and run.claim_signature is None and run.status_signature is None
		assert run.supported_claim_count is None and run.no_evidence_claim_count is None

	def test_completed_and_failed_cannot_be_combined(self):
		for overrides in (
			{'failure_type': 'TransientProviderError'},
			{'failure_stage': PipelineStage.VERIFICATION},
			{'claim_count': None},
			{'claim_signature': None},
			{'status_signature': None},
		):
			with pytest.raises(Exception):
				_completed_run(**overrides)

	def test_a_failed_replay_must_say_why(self):
		with pytest.raises(Exception, match='failure type'):
			_failed_run(failure_type=None)

	def test_a_stopped_replay_cannot_carry_a_verdict(self):
		"""The Phase 9D guarantee: a transport failure is never recorded as UNSUPPORTED or NO_EVIDENCE."""
		for overrides in ({'claim_count': 1}, {'status_signature': ['UNSUPPORTED']}, {'supported_claim_count': 0}):
			with pytest.raises(Exception, match='never verified'):
				_failed_run(**overrides)

	def test_signatures_must_have_one_entry_per_claim(self):
		with pytest.raises(Exception, match='but signs'):
			_completed_run(claim_count=2)
		with pytest.raises(Exception, match='statuses'):
			_completed_run(status_signature=['SUPPORTED', 'SUPPORTED'])

	def test_attempts_are_calls_plus_retries(self):
		with pytest.raises(Exception, match='attempts that are not calls plus retries'):
			_completed_run(postprocess_llm_attempts=4)

	def test_no_more_resolved_calls_than_calls_made(self):
		with pytest.raises(Exception, match='resolved more calls'):
			_completed_run(postprocess_llm_recovered_calls=2, postprocess_llm_failed_calls=2)

	def test_exception_counts_must_account_for_every_failed_attempt(self):
		"""Each failed attempt either bought a retry or exhausted a budget, so a mismatch is a wrong snapshot."""
		with pytest.raises(Exception, match='do not match its retries'):
			_completed_run(postprocess_llm_retry_count=1, postprocess_llm_attempts=4)

	def test_status_counts_must_restate_the_status_signature(self):
		"""The six statuses partition a claim set, so counts and signature are the same fact twice."""
		fields = _completed_run().model_dump()

		with pytest.raises(Exception, match='does not match its status signature'):
			ReplayRunResult.model_validate({**fields, 'supported_claim_count': 0})
		with pytest.raises(Exception, match='does not match its status signature'):
			ReplayRunResult.model_validate({**fields, 'no_evidence_claim_count': 3})

	def test_a_status_that_is_not_a_status_is_refused(self):
		fields = _completed_run().model_dump()

		with pytest.raises(Exception, match='are not'):
			ReplayRunResult.model_validate(
				{
					**fields,
					'claim_count': 2,
					'claim_signature': ['a.', 'b.'],
					'status_signature': ['SUPPORTED', 'ALMOST_SUPPORTED'],
				}
			)

	def test_a_failure_type_is_a_type_name_plus_stage_and_nothing_else(self):
		for failure_type in (
			f'TimeoutError: {SECRET} for url https://example.com',
			'WebEvidencePipelineError: timeout',
			'gave up',
		):
			with pytest.raises(Exception, match='not a bare type name'):
				_failed_run(failure_type=failure_type)

	def test_an_exception_key_must_be_bare(self):
		with pytest.raises(Exception, match='not bare'):
			_failed_run(exception_type_counts={f'TimeoutError: {SECRET}': 1})

	def test_an_exception_count_must_be_positive(self):
		with pytest.raises(Exception, match='counts'):
			_failed_run(
				postprocess_llm_retry_count=1, postprocess_llm_attempts=2, exception_type_counts={'TransientProviderError': 0}
			)

	def test_repeat_index_is_one_based(self):
		with pytest.raises(Exception):
			_completed_run(repeat_index=0)

	def test_a_run_survives_a_json_round_trip(self):
		run = _completed_run(
			postprocess_llm_retry_count=1,
			postprocess_llm_attempts=4,
			postprocess_llm_recovered_calls=1,
			exception_type_counts={'TimeoutError': 1},
		)

		assert ReplayRunResult.model_validate_json(run.model_dump_json()) == run


class TestRunReplay:
	async def test_a_frozen_input_reaches_a_report_and_records_both_readings(self):
		run = await _replayed(_fixture(), model=StageChatModel())

		assert run.pipeline_completed and run.failure_stage is None and run.failure_type is None
		assert run.claim_count == 1
		assert run.claim_signature == [normalize_claim_text(CLAIM)]
		assert run.status_signature == [VerificationStatus.SUPPORTED.value]
		assert run.supported_claim_count == 1
		assert run.elapsed_seconds == 4.0
		assert (run.postprocess_llm_logical_calls, run.postprocess_llm_attempts, run.postprocess_llm_retry_count) == (3, 3, 0)

	async def test_two_claims_of_one_answer_produce_two_entries_in_each_signature(self):
		model = StageChatModel(claim_texts=[CLAIM, SECOND_CLAIM])

		run = await _replayed(_fixture(evidence_nodes=[_node('ev-1'), _node('ev-2')]), model=model)

		assert run.claim_count == 2
		assert run.claim_signature == [normalize_claim_text(CLAIM), normalize_claim_text(SECOND_CLAIM)]
		assert run.status_signature == [VerificationStatus.SUPPORTED.value, VerificationStatus.SUPPORTED.value]
		assert model.calls.count('RawClaimEvidenceAssessment') == 2

	async def test_a_contradicting_verdict_is_recorded_as_such(self):
		model = StageChatModel(relation=EvidenceRelation.CONTRADICTS)

		run = await _replayed(_fixture(), model=model)

		assert run.status_signature == [VerificationStatus.CONTRADICTED.value]
		assert run.contradicted_claim_count == 1 and run.supported_claim_count == 0

	async def test_a_transient_failure_that_recovers_is_recorded_as_a_success_with_a_cost(self):
		run = await _replayed(_fixture(), model=StageChatModel(fail_on={2}))

		assert run.pipeline_completed
		assert (run.postprocess_llm_logical_calls, run.postprocess_llm_attempts, run.postprocess_llm_retry_count) == (3, 4, 1)
		assert (run.postprocess_llm_recovered_calls, run.postprocess_llm_failed_calls) == (1, 0)
		assert run.exception_type_counts == {'TransientProviderError': 1}

	async def test_an_exhausted_budget_stops_the_replay_and_says_where(self):
		run = await _replayed(_fixture(), model=StageChatModel(fail_on={1}), max_attempts=1)

		assert not run.pipeline_completed
		assert run.failure_stage is PipelineStage.CLAIM_EXTRACTION
		assert run.failure_type == 'WebEvidencePipelineError:CLAIM_EXTRACTION'
		assert run.claim_count is None and run.status_signature is None
		assert (run.postprocess_llm_logical_calls, run.postprocess_llm_attempts, run.postprocess_llm_failed_calls) == (1, 1, 1)
		assert run.exception_type_counts == {'TransientProviderError': 1}

	async def test_a_later_stage_fails_under_its_own_stage_name(self):
		run = await _replayed(_fixture(), model=StageChatModel(fail_on={2, 3, 4}), max_attempts=3)

		assert run.failure_stage is PipelineStage.SEMANTIC_RERANKING
		assert run.failure_type == 'WebEvidencePipelineError:SEMANTIC_RERANKING'
		# Extraction spent one clean call, reranking spent three attempts and gave up.
		assert run.postprocess_llm_logical_calls == 2
		assert (run.postprocess_llm_attempts, run.postprocess_llm_retry_count) == (4, 2)
		assert run.exception_type_counts == {'TransientProviderError': 3}

	async def test_attempts_spent_before_a_failure_are_still_recorded(self):
		"""A stopped replay consumed real provider calls, and hiding that would understate what retry costs."""
		run = await _replayed(_fixture(), model=StageChatModel(fail_on={3, 4, 5}), max_attempts=3)

		assert not run.pipeline_completed
		assert run.failure_stage is PipelineStage.VERIFICATION
		assert (run.postprocess_llm_logical_calls, run.postprocess_llm_attempts, run.postprocess_llm_retry_count) == (3, 5, 2)
		assert run.postprocess_llm_failed_calls == 1
		assert run.exception_type_counts == {'TransientProviderError': 3}

	async def test_no_message_prompt_or_answer_text_reaches_the_record(self):
		run = await _replayed(_fixture(), model=StageChatModel(fail_on={1, 2, 3}), max_attempts=3)

		persisted = run.model_dump_json()
		assert SECRET not in persisted
		assert ANSWER not in persisted
		assert 'TransientProviderError' in persisted

	async def test_one_wrapper_serving_two_replays_still_reports_per_replay_counters(self):
		"""Replay 1 recovers from a retry, replay 2 is clean; a cumulative total would land on both records."""
		fixture = _fixture()
		model = StageChatModel(fail_on={2})
		wrapper = RetryingChatModel(model, policy=LLMRetryPolicy(max_attempts=3), sleep=_no_sleep)
		pipeline = _pipeline(wrapper)

		first = await run_replay(fixture, pipeline=pipeline, repeat_index=1, max_attempts=3, stats=wrapper.snapshot_stats)
		model.fail_on = set()
		second = await run_replay(fixture, pipeline=pipeline, repeat_index=2, max_attempts=3, stats=wrapper.snapshot_stats)

		assert (first.postprocess_llm_logical_calls, first.postprocess_llm_attempts, first.postprocess_llm_retry_count) == (
			3,
			4,
			1,
		)
		assert (second.postprocess_llm_logical_calls, second.postprocess_llm_attempts, second.postprocess_llm_retry_count) == (
			3,
			3,
			0,
		)
		assert second.postprocess_llm_recovered_calls == 0
		assert wrapper.snapshot_stats().logical_invocation_count == 6

	async def test_replaying_a_fixture_never_thaws_it(self):
		fixture = _fixture(evidence_nodes=[_node('ev-1'), _node('ev-2')])
		before = deepcopy(fixture)

		await _replayed(fixture, model=StageChatModel(claim_texts=[CLAIM, SECOND_CLAIM]))

		assert fixture == before
		assert [node.evidence_id for node in fixture.evidence_nodes] == ['ev-1', 'ev-2']

	async def test_a_replay_with_no_evidence_verifies_nothing_but_still_completes(self):
		run = await _replayed(_fixture(evidence_nodes=[]), model=StageChatModel())

		assert run.pipeline_completed
		assert run.claim_count == 1
		assert run.no_evidence_claim_count == 1
		assert run.status_signature == [VerificationStatus.NO_EVIDENCE.value]
		# Nothing to rerank or verify, so only the extraction call reached the provider.
		assert run.postprocess_llm_logical_calls == 1

	async def test_repeat_index_and_budget_are_carried_onto_the_record(self):
		run = await _replayed(_fixture(), model=StageChatModel(), max_attempts=1, repeat_index=7)

		assert (run.repeat_index, run.max_attempts) == (7, 1)


class TestSummarizeReplayRuns:
	def test_nothing_measured_is_none_rather_than_zero(self):
		summary = summarize_replay_runs([])

		assert summary.run_count == 0 and summary.fixture_count == 0
		assert summary.pipeline_completion_rate is None
		assert summary.mean_elapsed_seconds is None and summary.mean_completed_elapsed_seconds is None
		assert summary.mean_claim_count is None
		assert summary.max_attempts is None
		assert summary.exception_type_counts == {} and summary.failure_stage_counts == {}

	def test_completion_is_over_every_replay(self):
		summary = summarize_replay_runs([_run_for('f1', 1), _run_for('f1', 2, completed=False), _run_for('f1', 3)])

		assert summary.run_count == 3
		assert summary.failed_run_count == 1
		assert summary.pipeline_completion_rate == pytest.approx(2 / 3)

	def test_retry_totals_and_the_two_run_counts(self):
		runs = [
			_run_for(
				'f1',
				1,
				postprocess_llm_logical_calls=3,
				postprocess_llm_attempts=4,
				postprocess_llm_retry_count=1,
				postprocess_llm_recovered_calls=1,
				exception_type_counts={'TimeoutError': 1},
			),
			_run_for('f1', 2),
			_run_for('f1', 3, completed=False),
		]

		summary = summarize_replay_runs(runs)

		assert (summary.total_logical_calls, summary.total_provider_attempts, summary.total_retries) == (7, 8, 1)
		assert (summary.total_recovered_calls, summary.total_failed_calls) == (1, 1)
		assert summary.runs_with_retry_count == 1
		assert summary.runs_recovered_by_retry_count == 1

	def test_a_stopped_replay_that_recovered_a_call_is_not_counted_as_recovered(self):
		"""runs_recovered_by_retry_count counts reports that exist; a replay that stopped produced none."""
		summary = summarize_replay_runs(
			[
				_run_for(
					'f1',
					1,
					completed=False,
					postprocess_llm_retry_count=1,
					postprocess_llm_attempts=2,
					postprocess_llm_failed_calls=0,
					postprocess_llm_recovered_calls=1,
					exception_type_counts={'TransientProviderError': 1},
				)
			]
		)

		assert summary.runs_with_retry_count == 1
		assert summary.runs_recovered_by_retry_count == 0

	def test_exception_types_are_merged_across_replays_and_sorted(self):
		runs = [
			_run_for(
				'f1',
				1,
				completed=False,
				postprocess_llm_retry_count=1,
				postprocess_llm_attempts=2,
				exception_type_counts={'ValidationError': 2},
			),
			_run_for(
				'f1',
				2,
				completed=False,
				postprocess_llm_retry_count=2,
				postprocess_llm_attempts=3,
				exception_type_counts={'TimeoutError': 3},
			),
		]

		summary = summarize_replay_runs(runs)

		assert summary.exception_type_counts == {'TimeoutError': 3, 'ValidationError': 2}
		assert list(summary.exception_type_counts) == ['TimeoutError', 'ValidationError']

	def test_failures_are_attributed_to_the_stage_that_stopped_them(self):
		runs = [
			_run_for('f1', 1, completed=False),
			_run_for(
				'f1',
				2,
				completed=False,
				failure_stage=PipelineStage.SEMANTIC_RERANKING,
				failure_type='WebEvidencePipelineError:SEMANTIC_RERANKING',
			),
			_run_for(
				'f1',
				3,
				completed=False,
				failure_stage=PipelineStage.SEMANTIC_RERANKING,
				failure_type='WebEvidencePipelineError:SEMANTIC_RERANKING',
			),
			_run_for('f1', 4),
		]

		summary = summarize_replay_runs(runs)

		assert summary.failure_stage_counts == {'CLAIM_EXTRACTION': 1, 'SEMANTIC_RERANKING': 2}

	def test_timing_separates_all_replays_from_the_completed_ones(self):
		runs = [
			_run_for('f1', 1, elapsed_seconds=2.0),
			_run_for('f1', 2, elapsed_seconds=4.0),
			_run_for('f1', 3, completed=False, elapsed_seconds=30.0),
		]

		summary = summarize_replay_runs(runs)

		assert summary.mean_elapsed_seconds == pytest.approx(12.0)
		assert summary.mean_completed_elapsed_seconds == pytest.approx(3.0)

	def test_claim_averages_ignore_replays_that_never_reached_claims(self):
		runs = [_run_for('f1', 1, claims=['a.', 'b.', 'c.', 'd.']), _run_for('f1', 2, completed=False)]

		assert summarize_replay_runs(runs).mean_claim_count == pytest.approx(4.0)

	def test_identical_replays_of_a_fixture_count_as_one_signature(self):
		summary = summarize_replay_runs([_run_for('f1', index) for index in (1, 2, 3)])

		assert summary.claim_signature_unique_count_by_fixture == {'f1': 1}
		assert summary.status_signature_unique_count_by_fixture == {'f1': 1}

	def test_a_different_claim_split_is_a_second_signature_of_its_own_kind(self):
		runs = [
			_run_for('f1', 1, claims=['one claim.', 'two claim.'], statuses=['SUPPORTED', 'SUPPORTED']),
			_run_for('f1', 2, claims=['a different split.'], statuses=['SUPPORTED']),
		]

		summary = summarize_replay_runs(runs)

		assert summary.claim_signature_unique_count_by_fixture == {'f1': 2}
		assert summary.status_signature_unique_count_by_fixture == {'f1': 2}

	def test_the_same_claims_can_disagree_only_on_the_verdict(self):
		"""Reliability and semantic stability are separate readings, and this is the case that proves it."""
		runs = [_run_for('f1', 1, statuses=['SUPPORTED']), _run_for('f1', 2, statuses=['CONTRADICTED'])]

		summary = summarize_replay_runs(runs)

		assert summary.claim_signature_unique_count_by_fixture == {'f1': 1}
		assert summary.status_signature_unique_count_by_fixture == {'f1': 2}

	def test_a_fixture_with_no_completed_replay_is_listed_as_zero_not_one(self):
		"""0 means "nothing to compare", which is a different statement from "perfectly unstable"."""
		summary = summarize_replay_runs([_run_for('f1', 1, completed=False), _run_for('f2', 1)])

		assert summary.claim_signature_unique_count_by_fixture == {'f1': 0, 'f2': 1}
		assert summary.status_signature_unique_count_by_fixture == {'f1': 0, 'f2': 1}

	def test_the_budget_is_reported_only_when_every_replay_shared_one(self):
		assert summarize_replay_runs([_run_for('f1', 1), _run_for('f1', 2)]).max_attempts == 3
		assert summarize_replay_runs([_run_for('f1', 1), _run_for('f1', 2, max_attempts=1)]).max_attempts is None

	def test_run_count_and_fixture_count_are_different_numbers(self):
		summary = summarize_replay_runs([_run_for('f1', index) for index in (1, 2)] + [_run_for('f2', 1)])

		assert (summary.run_count, summary.fixture_count) == (3, 2)

	def test_a_summary_survives_a_json_round_trip(self):
		summary = summarize_replay_runs([_run_for('f1', 1), _run_for('f1', 2, completed=False)])

		assert ReplayBenchmarkSummary.model_validate_json(summary.model_dump_json()) == summary


class TestCompareReplayResults:
	def _summaries(self) -> tuple[ReplayBenchmarkSummary, ReplayBenchmarkSummary]:
		baseline = summarize_replay_runs([_run_for('f1', 1, max_attempts=1), _run_for('f1', 2, max_attempts=1, completed=False)])
		candidate = summarize_replay_runs([_run_for('f1', 1, max_attempts=3), _run_for('f1', 2, max_attempts=3)])
		return baseline, candidate

	def test_deltas_are_candidate_minus_baseline(self):
		baseline, candidate = self._summaries()

		comparison = compare_replay_results(baseline, candidate)

		assert (comparison.max_attempts_a, comparison.max_attempts_b) == (1, 3)
		assert (comparison.run_count_a, comparison.run_count_b) == (2, 2)
		assert comparison.completion_rate_a == pytest.approx(0.5)
		assert comparison.completion_rate_b == pytest.approx(1.0)
		assert comparison.completion_rate_delta == pytest.approx(0.5)
		assert comparison.mean_elapsed_delta == pytest.approx(comparison.mean_elapsed_b - comparison.mean_elapsed_a)
		assert (comparison.failed_runs_a, comparison.failed_runs_b) == (1, 0)

	def test_exception_types_and_recovery_stay_on_their_own_side(self):
		baseline, candidate = self._summaries()
		candidate = candidate.model_copy(update={'exception_type_counts': {'TimeoutError': 2}, 'total_recovered_calls': 2})

		comparison = compare_replay_results(baseline, candidate)

		assert comparison.exception_type_counts_a == {'TransientProviderError': 1}
		assert comparison.exception_type_counts_b == {'TimeoutError': 2}
		assert comparison.recovered_calls_b == 2
		assert comparison.exception_type_counts_a != comparison.exception_type_counts_b

	def test_an_empty_side_is_refused_rather_than_reported_as_zero(self):
		baseline, candidate = self._summaries()
		empty = summarize_replay_runs([])

		with pytest.raises(ReplayBenchmarkError, match='the baseline side has no replays'):
			compare_replay_results(empty, candidate)
		with pytest.raises(ReplayBenchmarkError, match='the candidate side has no replays'):
			compare_replay_results(baseline, empty)

	def test_a_side_with_two_budgets_describes_no_single_configuration(self):
		baseline, candidate = self._summaries()
		mixed = summarize_replay_runs([_run_for('f1', 1, max_attempts=1), _run_for('f1', 2, max_attempts=3)])

		with pytest.raises(ReplayBenchmarkError, match='share one retry budget'):
			compare_replay_results(mixed, candidate)
		with pytest.raises(ReplayBenchmarkError, match='share one retry budget'):
			compare_replay_results(baseline, mixed)

	def test_the_comparison_reports_no_statistics_it_cannot_support(self):
		fields = set(ReplayComparison.model_fields)

		assert not any(word in name.lower() for name in fields for word in ('p_value', 'significance', 'confidence', 'interval'))


class TestBenchmarkArtifact:
	def test_a_whole_benchmark_result_survives_a_json_round_trip(self):
		runs = [_run_for('f1', 1), _run_for('f1', 2, completed=False)]
		result = EvidenceReplayBenchmarkResult(summary=summarize_replay_runs(runs), runs=runs)

		assert EvidenceReplayBenchmarkResult.model_validate_json(result.model_dump_json()) == result

	def test_the_summary_is_the_caller_aggregate_not_a_derived_field(self):
		"""A result built without a summary says so with empty aggregates; the script passes real ones."""
		runs = [_run_for('f1', 1)]

		assert EvidenceReplayBenchmarkResult(runs=runs).summary.run_count == 0
		assert EvidenceReplayBenchmarkResult(summary=summarize_replay_runs(runs), runs=runs).summary.run_count == 1

	def test_a_run_record_keeps_signatures_not_intermediates(self):
		"""The replay keeps what a reader needs to judge it, and drops everything that carries page text."""
		fields = set(ReplayRunResult.model_fields)

		assert not fields & {
			'answer',
			'task',
			'evidence_nodes',
			'alignment_result',
			'reranking_result',
			'verification_result',
			'evidence_graph',
			'report',
			'markdown',
		}
