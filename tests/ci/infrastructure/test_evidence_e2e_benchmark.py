"""Tests for the paired browser benchmark: models, answer checker and aggregation.

No browser, no network and no live model here. The script that runs the real benchmark is deliberately
not imported: a pytest run must never start Chromium, and the parts worth testing are the pure ones.
"""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from browser_use.evidence import (
	AlignmentResult,
	ClaimSet,
	EvidenceGraph,
	EvidenceRelation,
	LLMRetryStats,
	RerankingResult,
	VerificationResult,
	VerificationStatus,
	stats_delta,
)
from browser_use.evidence.e2e_benchmark import (
	AnswerCheck,
	AnswerMatchMode,
	BrowserBenchmarkCase,
	BrowserBenchmarkError,
	BrowserBenchmarkFailureStage,
	BrowserBenchmarkRunResult,
	BrowserBenchmarkSummary,
	EvidenceBrowserBenchmarkResult,
	evaluate_answer,
	load_browser_benchmark_cases,
	normalize_answer_text,
	run_result_with_pipeline,
	run_result_with_retry_stats,
	summarize_browser_runs,
)
from browser_use.evidence.pipeline import WebEvidencePipelineResult
from browser_use.evidence.reporting import (
	ClaimReportSection,
	EvidenceGroundedReport,
	ReportClaimEvidence,
	ReportSummary,
)

DATASET = Path('benchmarks/webevidence/browser_cases.jsonl')


def _case(**overrides) -> BrowserBenchmarkCase:
	fixture = {
		'case_id': 'c1',
		'task': 'Open https://example.com and report the page heading.',
		'expected_answer_patterns': ['example domain'],
	}
	fixture.update(overrides)
	return BrowserBenchmarkCase(**fixture)


def _run(**overrides) -> BrowserBenchmarkRunResult:
	fixture = {
		'case_id': 'c1',
		'task': 't',
		'agent_completed': True,
		'final_answer_present': True,
		'answer_check_passed': True,
		'evidence_count': 2,
		'pipeline_completed': True,
		'claim_count': 2,
		'supported_claim_count': 2,
		'browser_step_count': 3,
	}
	fixture.update(overrides)
	return BrowserBenchmarkRunResult(**fixture)


def _failed_run(case_id: str, stage: BrowserBenchmarkFailureStage, failure_type: str) -> BrowserBenchmarkRunResult:
	"""A run that stopped early: nothing was answered, checked or verified."""
	return _run(
		case_id=case_id,
		agent_completed=stage is not BrowserBenchmarkFailureStage.AGENT_RUN,
		final_answer_present=False,
		answer_check_passed=None,
		pipeline_completed=False,
		claim_count=None,
		supported_claim_count=None,
		browser_step_count=None,
		evidence_count=0,
		failure_stage=stage,
		failure_type=failure_type,
	)


def _statuses(*statuses: VerificationStatus) -> BrowserBenchmarkRunResult:
	"""Counts for a run whose claims carry exactly these statuses, with no evidence for the rest."""
	counts = {status: 0 for status in VerificationStatus}
	for status in statuses:
		counts[status] += 1
	return _run(
		claim_count=len(statuses),
		supported_claim_count=counts[VerificationStatus.SUPPORTED],
		partial_claim_count=counts[VerificationStatus.PARTIAL],
		unsupported_claim_count=counts[VerificationStatus.UNSUPPORTED],
		contradicted_claim_count=counts[VerificationStatus.CONTRADICTED],
		conflicted_claim_count=counts[VerificationStatus.CONFLICTED],
		no_evidence_claim_count=counts[VerificationStatus.NO_EVIDENCE],
		evidence_coverage_rate=(len(statuses) - counts[VerificationStatus.NO_EVIDENCE]) / len(statuses) if statuses else 0.0,
		fully_supported=bool(statuses) and all(status is VerificationStatus.SUPPORTED for status in statuses),
	)


def _pipeline_result(statuses: list[VerificationStatus], evidence_count: int = 2) -> WebEvidencePipelineResult:
	counts = {status: 0 for status in VerificationStatus}
	for status in statuses:
		counts[status] += 1
	claim_count = len(statuses)
	covered = claim_count - counts[VerificationStatus.NO_EVIDENCE]
	report = EvidenceGroundedReport(
		task_id='task-1',
		task='t',
		summary=ReportSummary(
			claim_count=claim_count,
			evidence_count=evidence_count,
			unique_source_count=1,
			supported_claim_count=counts[VerificationStatus.SUPPORTED],
			partial_claim_count=counts[VerificationStatus.PARTIAL],
			unsupported_claim_count=counts[VerificationStatus.UNSUPPORTED],
			contradicted_claim_count=counts[VerificationStatus.CONTRADICTED],
			conflicted_claim_count=counts[VerificationStatus.CONFLICTED],
			no_evidence_claim_count=counts[VerificationStatus.NO_EVIDENCE],
			evidence_covered_claim_count=covered,
		),
		claims=[
			ClaimReportSection(
				claim_id=f'claim-{index}',
				order=index,
				claim_text=f'claim {index}',
				status=status,
				evidence=[]
				if status is VerificationStatus.NO_EVIDENCE
				else [ReportClaimEvidence(evidence_id='ev-1', relation=EvidenceRelation.SUPPORTS, explanation='states it.')],
			)
			for index, status in enumerate(statuses, start=1)
		],
	)
	return WebEvidencePipelineResult(
		task_id='task-1',
		task='t',
		answer='a',
		evidence_count=evidence_count,
		claim_set=ClaimSet(task_id='task-1', task='t', answer='a'),
		alignment_result=AlignmentResult(task_id='task-1'),
		reranking_result=RerankingResult(task_id='task-1'),
		verification_result=VerificationResult(task_id='task-1'),
		evidence_graph=EvidenceGraph(task_id='task-1'),
		report=report,
		markdown='# report',
	)


_TELEMETRY_FIELDS = (
	'postprocess_llm_logical_calls',
	'postprocess_llm_attempts',
	'postprocess_llm_retry_count',
	'postprocess_llm_recovered_calls',
	'postprocess_llm_failed_calls',
)


def _retry_telemetry(
	*, logical: int = 3, attempts: int = 3, retries: int = 0, recovered: int = 0, failed: int = 0
) -> dict[str, int]:
	"""The five counters as keyword arguments.

	The defaults are a single-claim run that asked for one extraction, one reranking and one verification
	and got all three on the first attempt, so a test states only the interesting part.
	"""
	return {
		'postprocess_llm_logical_calls': logical,
		'postprocess_llm_attempts': attempts,
		'postprocess_llm_retry_count': retries,
		'postprocess_llm_recovered_calls': recovered,
		'postprocess_llm_failed_calls': failed,
	}


class TestCaseModel:
	def test_defaults_are_all_matching_and_a_short_budget(self):
		case = _case()

		assert case.answer_match_mode is AnswerMatchMode.ALL
		assert case.max_steps == 8
		assert case.tags == []
		assert case.description == ''

	def test_blank_identifiers_and_patterns_are_refused(self):
		for overrides in (
			{'case_id': '   '},
			{'task': ''},
			{'expected_answer_patterns': []},
			{'expected_answer_patterns': ['  ']},
		):
			fields = {'case_id': 'c1', 'task': 't', 'expected_answer_patterns': ['x']}
			fields.update(overrides)
			with pytest.raises(Exception):
				BrowserBenchmarkCase(**fields)

	@pytest.mark.parametrize('max_steps', [0, -3])
	def test_a_run_with_no_steps_cannot_be_scheduled(self, max_steps):
		with pytest.raises(Exception, match='max_steps'):
			_case(max_steps=max_steps)

	def test_patterns_must_be_non_blank_strings(self):
		with pytest.raises(Exception):
			_case(expected_answer_patterns=['ok', ''])


class TestDatasetLoader:
	def test_shipped_dataset_loads_and_covers_the_planned_shapes(self):
		cases = load_browser_benchmark_cases(DATASET)

		assert 6 <= len(cases) <= 8
		by_id = {case.case_id: case for case in cases}
		assert {'example-heading', 'example-purpose', 'two-page-link', 'heading-plus-manager'} <= set(by_id)
		# The multi-page task and the extra-claim trap are both required by the phase.
		assert 'multi-page' in by_id['two-page-link'].tags
		assert 'extra-claim-trap' in by_id['heading-plus-manager'].tags

	def test_the_dataset_avoids_fast_changing_gold(self):
		patterns = ' '.join(
			pattern for case in load_browser_benchmark_cases(DATASET) for pattern in case.expected_answer_patterns
		)

		for volatile in ('star', 'price', 'weather', 'rank', 'million', 'thousand'):
			assert volatile not in patterns.lower()

	def test_duplicate_case_id_names_both_lines(self, tmp_path):
		path = tmp_path / 'cases.jsonl'
		path.write_text('\n'.join([_case().model_dump_json(), _case(task='other').model_dump_json()]) + '\n', encoding='utf-8')

		with pytest.raises(BrowserBenchmarkError, match='duplicate case_id .c1. on lines 1 and 2'):
			load_browser_benchmark_cases(path)

	def test_malformed_line_reports_its_line_number(self, tmp_path):
		path = tmp_path / 'cases.jsonl'
		path.write_text(_case().model_dump_json() + '\n' + '{"case_id": "broken"\n', encoding='utf-8')

		with pytest.raises(BrowserBenchmarkError, match='line 2'):
			load_browser_benchmark_cases(path)

	def test_blank_lines_are_skipped_and_an_empty_dataset_is_refused(self, tmp_path):
		path = tmp_path / 'cases.jsonl'
		path.write_text('\n\n   \n', encoding='utf-8')

		with pytest.raises(BrowserBenchmarkError, match='contains no cases'):
			load_browser_benchmark_cases(path)

	def test_a_missing_file_is_reported(self, tmp_path):
		with pytest.raises(BrowserBenchmarkError, match='Cannot read'):
			load_browser_benchmark_cases(tmp_path / 'absent.jsonl')

	def test_the_dataset_file_is_not_modified(self):
		before = DATASET.read_bytes()

		load_browser_benchmark_cases(DATASET)

		assert DATASET.read_bytes() == before


class TestDeterministicAnswerChecker:
	def test_literal_patterns_match_as_normalized_substrings(self):
		check = evaluate_answer(_case(), 'The page heading is “Example    Domain”.')

		assert check.answer_check_passed
		assert check.matched_patterns == ['example domain']
		assert check.missing_patterns == []

	def test_unicode_and_fullwidth_forms_agree(self):
		assert normalize_answer_text('ＧＰＬ  License') == 'gpl license'
		assert evaluate_answer(_case(expected_answer_patterns=['ｇｐｌ']), 'a GPL license').answer_check_passed

	def test_all_mode_requires_every_pattern(self):
		case = _case(expected_answer_patterns=['example domain', 'illustrative'])

		assert evaluate_answer(case, 'Example Domain is illustrative.').answer_check_passed
		partial = evaluate_answer(case, 'Example Domain only.')
		assert not partial.answer_check_passed
		assert partial.matched_patterns == ['example domain']
		assert partial.missing_patterns == ['illustrative']

	def test_any_mode_needs_only_one(self):
		case = _case(expected_answer_patterns=['python', 'rust'], answer_match_mode=AnswerMatchMode.ANY)

		assert evaluate_answer(case, 'mostly Rust code').answer_check_passed
		assert evaluate_answer(case, 'mostly Go code').answer_check_passed is False

	def test_regex_patterns_are_supported_and_matched_against_normalized_text(self):
		case = _case(expected_answer_patterns=['regex:example\\s+domain'])

		assert evaluate_answer(case, 'the heading is example\n domain').answer_check_passed

	def test_a_broken_regex_is_a_dataset_typo_not_a_failed_answer(self):
		with pytest.raises(BrowserBenchmarkError, match='Invalid regex expected answer pattern'):
			evaluate_answer(_case(expected_answer_patterns=['regex:(']), 'anything')

	def test_no_answer_is_a_failed_check_here_and_a_none_in_the_record(self):
		"""The checker always has a verdict for a string; the run record keeps None for no check at all."""
		check = evaluate_answer(_case(), None)

		assert check.answer_check_passed is False
		assert check.missing_patterns == ['example domain']

		blank = evaluate_answer(_case(), '   ')
		assert blank.answer_check_passed is False

	def test_a_returned_check_never_leaves_the_answer_out(self):
		with pytest.raises(Exception):
			AnswerCheck()


class TestRunRecord:
	def test_patterns_without_a_verdict_are_refused(self):
		with pytest.raises(Exception, match='without an answer check'):
			_run(answer_check_passed=None, matched_patterns=['example domain'])

	def test_a_completed_pipeline_must_report_its_claim_count(self):
		with pytest.raises(Exception, match='no claim count'):
			_run(claim_count=None)

	def test_fully_supported_needs_a_pipeline_result_behind_it(self):
		with pytest.raises(Exception, match='without a pipeline result'):
			_run(pipeline_completed=False, fully_supported=True)

	def test_no_claims_can_never_be_fully_supported(self):
		with pytest.raises(Exception, match='no claims'):
			_run(claim_count=0, supported_claim_count=0, fully_supported=True)

	def test_an_upstream_failure_cannot_coexist_with_a_completed_pipeline(self):
		with pytest.raises(Exception, match='completed the pipeline'):
			_run(failure_stage=BrowserBenchmarkFailureStage.PIPELINE, failure_type='X')

	def test_a_failed_write_may_follow_a_completed_pipeline(self):
		"""The analysis exists and the artifact does not, which is a real and separate outcome."""
		run = _run(failure_stage=BrowserBenchmarkFailureStage.OUTPUT_WRITE, failure_type='OSError', report_json_path=None)

		assert run.pipeline_completed
		assert run.report_json_path is None

	def test_failure_type_is_a_name_never_a_message(self):
		run = _failed_run('c1', BrowserBenchmarkFailureStage.AGENT_RUN, type(RuntimeError('api-key=secret123')).__name__)

		assert run.failure_type == 'RuntimeError'
		assert 'secret123' not in run.model_dump_json()

	def test_the_record_holds_no_answer_text_or_dom(self):
		fields = set(BrowserBenchmarkRunResult.model_fields)

		assert not fields & {'answer', 'final_answer', 'dom', 'screenshot', 'metadata', 'api_key', 'notes'}
		assert {'report_json_path', 'report_markdown_path'} <= fields


class TestPipelineProjection:
	def test_claim_counts_and_flags_are_copied_from_the_report(self):
		run = _run(pipeline_completed=False, claim_count=None, supported_claim_count=None, fully_supported=None, evidence_count=0)
		pipeline = _pipeline_result([VerificationStatus.SUPPORTED, VerificationStatus.NO_EVIDENCE])

		filled = run_result_with_pipeline(run, pipeline)

		assert filled.pipeline_completed
		assert filled.claim_count == 2
		assert filled.supported_claim_count == 1
		assert filled.no_evidence_claim_count == 1
		assert filled.evidence_coverage_rate == 0.5
		assert filled.fully_supported is False
		assert filled.evidence_count == 2

	def test_the_input_record_is_not_mutated(self):
		run = _run(pipeline_completed=False, claim_count=None, supported_claim_count=None, fully_supported=None)
		before = deepcopy(run)

		run_result_with_pipeline(run, _pipeline_result([VerificationStatus.SUPPORTED]))

		assert run == before

	def test_zero_claims_is_not_fully_supported_even_when_every_claim_agrees(self):
		filled = run_result_with_pipeline(_run(pipeline_completed=False, claim_count=None), _pipeline_result([]))

		assert filled.claim_count == 0
		assert filled.fully_supported is False
		assert filled.pipeline_completed


class TestRetryTelemetry:
	def test_a_run_that_never_reached_the_pipeline_has_nothing_to_count(self):
		"""None, not 0: no post-processing call was made, which is a different claim from making three cleanly."""
		run = _failed_run('c1', BrowserBenchmarkFailureStage.AGENT_RUN, 'TimeoutError')

		assert all(getattr(run, field) is None for field in _TELEMETRY_FIELDS)

	def test_a_partial_counter_block_is_refused(self):
		"""A single filled counter would feed a phantom zero into the summary totals."""
		block = _retry_telemetry()
		block.pop('postprocess_llm_failed_calls')

		with pytest.raises(Exception, match='partial post-processing retry telemetry'):
			_run(**block)

	def test_a_clean_run_reports_zero_retries_as_zero(self):
		summary = summarize_browser_runs([_run(**_retry_telemetry())])

		assert summary.total_postprocess_llm_logical_calls == 3
		assert summary.total_postprocess_llm_attempts == 3
		assert summary.total_postprocess_llm_retries == 0
		assert summary.total_postprocess_llm_recovered_calls == 0
		assert summary.runs_with_postprocess_retry_count == 0
		assert summary.runs_recovered_by_retry_count == 0

	def test_the_counters_pool_over_the_runs_that_reported_them(self):
		runs = [
			_run(case_id='retried', **_retry_telemetry(logical=3, attempts=5, retries=2, recovered=2)),
			_run(case_id='clean', **_retry_telemetry(logical=4, attempts=4)),
		]

		summary = summarize_browser_runs(runs)

		assert summary.total_postprocess_llm_logical_calls == 7
		assert summary.total_postprocess_llm_attempts == 9
		assert summary.total_postprocess_llm_retries == 2
		assert summary.total_postprocess_llm_recovered_calls == 2
		assert summary.total_postprocess_llm_failed_calls == 0
		assert summary.runs_with_postprocess_retry_count == 1
		assert summary.runs_recovered_by_retry_count == 1

	def test_no_run_reported_telemetry_so_the_totals_stay_none(self):
		summary = summarize_browser_runs([_run(), _failed_run('c2', BrowserBenchmarkFailureStage.PIPELINE, 'X')])

		for field in (
			'total_postprocess_llm_logical_calls',
			'total_postprocess_llm_attempts',
			'total_postprocess_llm_retries',
			'total_postprocess_llm_recovered_calls',
			'total_postprocess_llm_failed_calls',
		):
			assert getattr(summary, field) is None, field
		assert summary.runs_with_postprocess_retry_count == 0

	def test_the_projection_fills_the_counters_and_leaves_the_rest_alone(self):
		run = _run(pipeline_completed=False, claim_count=None, fully_supported=None)
		before = deepcopy(run)

		filled = run_result_with_retry_stats(
			run,
			LLMRetryStats(
				logical_invocation_count=3,
				attempt_count=6,
				retry_count=3,
				recovered_invocation_count=2,
				failed_invocation_count=1,
				exception_type_counts={'TimeoutError': 3},
			),
		)

		assert run == before
		assert (filled.postprocess_llm_logical_calls, filled.postprocess_llm_attempts) == (3, 6)
		assert (filled.postprocess_llm_retry_count, filled.postprocess_llm_recovered_calls) == (3, 2)
		assert filled.postprocess_llm_failed_calls == 1
		assert filled.claim_count is None and not filled.pipeline_completed

	def test_an_exception_type_name_never_reaches_the_shared_artifact(self):
		"""The record is the file that gets handed around, so it carries counts and nothing else."""
		filled = run_result_with_retry_stats(
			_run(**_retry_telemetry()),
			LLMRetryStats(logical_invocation_count=1, attempt_count=3, retry_count=2, exception_type_counts={'TimeoutError': 2}),
		)
		dumped = filled.model_dump_json()

		assert 'TimeoutError' not in dumped
		assert 'exception_type' not in dumped

	def test_a_run_records_its_own_delta_not_the_shared_wrappers_total(self):
		"""One wrapper serves every run, so the pair of snapshots has to be subtracted before it is stored."""
		before = LLMRetryStats(logical_invocation_count=2, attempt_count=4, retry_count=2, recovered_invocation_count=2)
		after = LLMRetryStats(
			logical_invocation_count=5,
			attempt_count=9,
			retry_count=4,
			recovered_invocation_count=4,
			failed_invocation_count=1,
		)

		run = run_result_with_retry_stats(_run(**_retry_telemetry()), stats_delta(before, after))

		assert (run.postprocess_llm_logical_calls, run.postprocess_llm_attempts, run.postprocess_llm_retry_count) == (3, 5, 2)
		assert run.postprocess_llm_logical_calls != after.logical_invocation_count

	def test_a_run_that_failed_after_retrying_still_pays_for_its_attempts(self):
		"""Retry visibility, not softening: the stage failure owns the run and its attempts stay in the totals."""
		run = run_result_with_retry_stats(
			_failed_run('broken', BrowserBenchmarkFailureStage.PIPELINE, 'WebEvidencePipelineError:CLAIM_EXTRACTION'),
			LLMRetryStats(logical_invocation_count=1, attempt_count=3, retry_count=2, failed_invocation_count=1),
		)

		assert not run.pipeline_completed
		summary = summarize_browser_runs([run])

		assert summary.total_postprocess_llm_attempts == 3
		assert summary.total_postprocess_llm_retries == 2
		assert summary.total_postprocess_llm_failed_calls == 1
		assert summary.runs_with_postprocess_retry_count == 1
		# Nothing recovered it, so it is not a run that retry saved.
		assert summary.runs_recovered_by_retry_count == 0
		assert summary.pipeline_fail_case_ids == ['broken']


class TestSummary:
	def test_rates_use_pooled_claims_not_an_average_of_run_percentages(self):
		"""One claim in a first run and one of three in a second must not look like 50 percent each."""
		runs = [
			_statuses(VerificationStatus.SUPPORTED),
			_statuses(VerificationStatus.SUPPORTED, VerificationStatus.UNSUPPORTED, VerificationStatus.UNSUPPORTED),
		]

		summary = summarize_browser_runs(runs)

		assert summary.total_claims_verified == 4
		assert summary.supported_claim_rate == pytest.approx(0.5)
		# A per-run average would have said 0.667 here, which is the difference the docstring warns about.
		assert summary.supported_claim_rate != pytest.approx((1.0 + 1 / 3) / 2)
		assert summary.unsupported_claim_rate == pytest.approx(0.5)

	def test_task_success_is_measured_only_over_runs_that_answered(self):
		runs = [
			_run(case_id='answered'),
			_run(
				case_id='silent',
				final_answer_present=False,
				answer_check_passed=None,
				pipeline_completed=False,
				claim_count=None,
				supported_claim_count=None,
			),
		]

		summary = summarize_browser_runs(runs)

		assert summary.final_answer_rate == 0.5
		assert summary.answer_check_pass_rate == 1.0
		assert summary.pipeline_completion_rate == 0.5

	def test_every_denominator_is_none_rather_than_zero_when_nothing_was_measured(self):
		summary = summarize_browser_runs([])

		assert summary.run_count == 0
		assert summary.agent_completion_rate is None
		assert summary.answer_check_pass_rate is None
		assert summary.mean_browser_steps is None
		assert summary.mean_claim_count is None
		assert summary.supported_claim_rate is None
		assert summary.fully_supported_run_rate is None
		assert summary.mean_agent_elapsed_seconds is None
		assert summary.total_claims_verified == 0

	def test_step_and_timing_means_ignore_runs_that_never_reported_them(self):
		runs = [
			_run(browser_step_count=None, agent_elapsed_seconds=None),
			_run(browser_step_count=5, agent_elapsed_seconds=2.0),
		]

		summary = summarize_browser_runs(runs)

		assert summary.mean_browser_steps == 5.0
		assert summary.mean_agent_elapsed_seconds == 2.0

	def test_the_headline_diagnostic_is_a_task_pass_with_an_unsupported_claim(self):
		runs = [
			_statuses(VerificationStatus.SUPPORTED, VerificationStatus.NO_EVIDENCE),
			_statuses(VerificationStatus.SUPPORTED),
		]

		summary = summarize_browser_runs(runs)

		assert summary.answer_pass_but_not_fully_supported_case_ids == ['c1']
		assert summary.fully_supported_run_rate == 0.5
		assert summary.no_evidence_case_ids == ['c1']

	def test_status_specific_lists_stay_apart(self):
		runs = [
			_run(case_id='contra', supported_claim_count=0, contradicted_claim_count=1),
			_run(case_id='confl', supported_claim_count=0, conflicted_claim_count=1),
			_run(case_id='unsup', supported_claim_count=1),
		]

		summary = summarize_browser_runs(runs)

		assert summary.contradicted_case_ids == ['contra']
		assert summary.conflicted_case_ids == ['confl']
		assert summary.no_evidence_case_ids == []

	def test_failures_are_grouped_by_stage_and_deduplicated_across_repeats(self):
		runs = [
			_failed_run('broken', BrowserBenchmarkFailureStage.AGENT_RUN, 'TimeoutError'),
			_failed_run('broken', BrowserBenchmarkFailureStage.PIPELINE, 'WebEvidencePipelineError'),
			_failed_run('other', BrowserBenchmarkFailureStage.FINAL_ANSWER, 'MissingFinalAnswer'),
		]

		summary = summarize_browser_runs(runs)

		assert summary.failure_case_ids_by_stage == {
			'AGENT_RUN': ['broken'],
			'FINAL_ANSWER': ['other'],
			'PIPELINE': ['broken'],
		}
		assert summary.pipeline_fail_case_ids == ['broken', 'other']
		# Only the agent failure never reported done; the other two died after the agent finished.
		assert summary.agent_completion_rate == pytest.approx(2 / 3)
		assert summary.answer_check_pass_rate is None

	def test_repeats_of_a_case_are_counted_as_runs_once_per_case_list(self):
		runs = [_run(case_id='a'), _run(case_id='a'), _run(case_id='b')]

		summary = summarize_browser_runs(runs)

		assert summary.run_count == 3
		assert summary.case_count == 2
		assert summary.answer_fail_case_ids == []

	def test_the_inputs_are_not_mutated(self):
		runs = [_statuses(VerificationStatus.SUPPORTED), _statuses(VerificationStatus.UNSUPPORTED)]
		before = deepcopy(runs)

		summarize_browser_runs(runs)

		assert runs == before


class TestResultArtifact:
	def test_result_round_trips_and_holds_no_timestamp(self):
		result = EvidenceBrowserBenchmarkResult(summary=summarize_browser_runs([_run()]), runs=[_run()])

		assert set(EvidenceBrowserBenchmarkResult.model_fields) == {'summary', 'runs'}
		assert EvidenceBrowserBenchmarkResult.model_validate_json(result.model_dump_json()) == result

	def test_the_same_records_serialise_identically(self):
		first = EvidenceBrowserBenchmarkResult(summary=summarize_browser_runs([_run()]), runs=[_run()])
		second = EvidenceBrowserBenchmarkResult(summary=summarize_browser_runs([_run()]), runs=[_run()])

		assert first.model_dump_json() == second.model_dump_json()

	def test_no_field_carries_dom_or_credential_shaped_data(self):
		names = set(BrowserBenchmarkSummary.model_fields) | set(BrowserBenchmarkRunResult.model_fields)

		for forbidden in ('dom', 'html', 'screenshot', 'api_key', 'key', 'secret', 'cookie', 'chain_of_thought', 'reasoning'):
			assert not any(forbidden in name for name in names), forbidden

	def test_the_summary_exposes_every_diagnostic_list(self):
		assert set(BrowserBenchmarkSummary.model_fields) >= {
			'answer_pass_but_not_fully_supported_case_ids',
			'answer_fail_case_ids',
			'pipeline_fail_case_ids',
			'no_evidence_case_ids',
			'contradicted_case_ids',
			'conflicted_case_ids',
		}
		assert json.dumps(summarize_browser_runs([]).model_dump())
