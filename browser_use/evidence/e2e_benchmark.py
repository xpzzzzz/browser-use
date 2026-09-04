"""Paired end-to-end benchmark models: one browser run, two readings of it.

Phase 9A measures the evidence engine on frozen inputs. This module measures what the engine does on
real browser runs, and it does so without a second run: the agent's final answer is the raw Browser Use
baseline, and the same answer plus the captured evidence is what the WebEvidence pipeline turns into a
claim-level report. Two separate agent runs could not be compared that way, because model sampling, page
state and timing all drift between runs, so a difference could just as well come from the trajectory as
from the evidence layer. Sharing one trajectory leaves only the analysis.

The consequence for how results must be read: WebEvidence does not rewrite the agent's answer, so this
benchmark cannot show an accuracy gain. What it can show is whether a task that a deterministic checker
declares passed still contains claims the collected evidence does not support, which is a class of
problem ordinary task-level scoring cannot see at all.

Nothing here touches a browser, a network or a model. The deterministic pieces are the answer checker,
the dataset loader and the aggregation; the live part that starts Chromium lives in
``scripts/run_webevidence_browser_benchmark.py``. Live runs are not reproducible, so this artifact keeps
no timestamp and claims no determinism, while the same input data still serialises identically.
"""

import re
import unicodedata
from collections.abc import Iterable, Sequence
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator

from browser_use.evidence.claims import NonBlankString
from browser_use.evidence.pipeline import WebEvidencePipelineResult
from browser_use.evidence.retrying_llm import LLMRetryStats
from browser_use.evidence.verification import VerificationStatus

# Pattern matching happens on normalized text, so dataset patterns are written in plain lowercase.
_WHITESPACE_PATTERN = re.compile(r'\s+')


class BrowserBenchmarkError(RuntimeError):
	"""Raised when the browser benchmark dataset or a run record cannot be trusted.

	Case identifiers carry every aggregate, so a duplicate id or a malformed line is refused with its line
	number rather than skipped: a quietly missing case changes what a rate means while looking identical.
	"""


class AnswerMatchMode(str, Enum):
	"""How a case's expected answer patterns combine."""

	ALL = 'ALL'
	ANY = 'ANY'


class BrowserBenchmarkFailureStage(str, Enum):
	"""Where a run stopped, so a benchmark can be read without opening every log file.

	``BROWSER_START`` and ``AGENT_RUN`` are kept apart because they mean different things: the first says
	Chromium never came up, which is an environment fault, while the second says the agent ran and failed
	or refused the task, which is a result. In practice a browser that fails to launch does so inside
	``Agent.run()``, and the public API offers no way to separate that from a model failure without reading
	the exception text, so the runner records launch faults as ``AGENT_RUN`` and reserves ``BROWSER_START``
	for a browser that cannot be configured at all.
	"""

	BROWSER_START = 'BROWSER_START'
	AGENT_RUN = 'AGENT_RUN'
	FINAL_ANSWER = 'FINAL_ANSWER'
	PIPELINE = 'PIPELINE'
	OUTPUT_WRITE = 'OUTPUT_WRITE'


class BrowserBenchmarkCase(BaseModel):
	"""One real browser task with a deterministic pass condition.

	The condition is a substring or regex test over the final answer, never a claim-level verdict: this
	checker asks only whether the answer covers what the task demanded, and deliberately ignores every
	additional factual claim the answer makes. Those extras are what the pipeline's verifier is for, and
	the gap between the two readings is the point of the benchmark.
	"""

	case_id: NonBlankString = Field(description='Stable identifier, unique within the dataset')
	task: NonBlankString = Field(description='The task handed to the Browser Use agent verbatim')
	expected_answer_patterns: list[NonBlankString] = Field(description='Patterns the final answer must contain')
	answer_match_mode: AnswerMatchMode = Field(default=AnswerMatchMode.ALL, description='ALL patterns, or ANY of them')
	tags: list[str] = Field(default_factory=list, description='Free-form grouping, e.g. multi-page or extra-claim-trap')
	max_steps: int = Field(default=8, ge=1, description='Step budget passed to Agent.run(max_steps=...)')
	description: str = Field(default='', description='What this case is meant to probe')

	@model_validator(mode='after')
	def _check_task(self) -> 'BrowserBenchmarkCase':
		"""A case with no pattern can never be judged, so it is a broken case rather than a hard one."""
		if not self.expected_answer_patterns:
			raise BrowserBenchmarkError(f'Browser benchmark case {self.case_id!r} needs at least one expected answer pattern')
		if self.answer_match_mode is AnswerMatchMode.ALL and len(set(self.expected_answer_patterns)) != len(
			self.expected_answer_patterns
		):
			raise BrowserBenchmarkError(f'Browser benchmark case {self.case_id!r} lists the same expected answer pattern twice')
		return self


def normalize_answer_text(text: str) -> str:
	"""NFKC, casefolded, whitespace-collapsed answer text.

	An agent writes "Example Domain", "example domain" or a heading split over two lines, and none of
	those differences say anything about whether the task was answered. Normalizing first keeps the
	patterns in the dataset short and lowercase instead of escaped against formatting.
	"""
	collapsed = _WHITESPACE_PATTERN.sub(' ', unicodedata.normalize('NFKC', text or '').casefold())
	return collapsed.strip()


class AnswerCheck(BaseModel):
	"""The deterministic task verdict: which required patterns the answer covered, and which it did not."""

	answer_check_passed: bool = Field(description='True when the case condition is satisfied')
	matched_patterns: list[str] = Field(default_factory=list, description='Patterns found in the answer')
	missing_patterns: list[str] = Field(default_factory=list, description='Patterns the answer never stated')


def evaluate_answer(case: BrowserBenchmarkCase, answer: str | None) -> AnswerCheck:
	"""Judge one final answer with string matching only, never with another model.

	An LLM judge would add its own errors to the measurement while this benchmark is still trying to count
	the errors of a pipeline, so the pass condition is a plain containment test: literal patterns match as
	substrings of the normalized answer, and a pattern wrapped in ``regex:`` is compiled with the standard
	library.

	This function always returns a verdict, including ``passed=False`` for a blank or missing answer, because
	"does this text satisfy the case" has a definite answer for any string. The run record keeps ``None`` for
	its own reason: a run that produced no answer at all was never given the check, and reporting that as a
	failed check would fold "no answer" into "wrong answer". The runner therefore calls this only when there
	is an answer to check.

	Raises:
		BrowserBenchmarkError: when a ``regex:`` pattern does not compile, which is a dataset typo.
	"""
	text = normalize_answer_text(answer if answer is not None else '')
	matched: list[str] = []
	missing: list[str] = []

	for pattern in case.expected_answer_patterns:
		if _pattern_matches(pattern, text):
			matched.append(pattern)
		else:
			missing.append(pattern)

	if case.answer_match_mode is AnswerMatchMode.ALL:
		passed = not missing
	else:
		passed = bool(matched)

	return AnswerCheck(answer_check_passed=passed, matched_patterns=matched, missing_patterns=missing)


def _pattern_matches(pattern: str, normalized_answer: str) -> bool:
	"""Containment for a literal pattern, ``re.search`` for one prefixed with ``regex:``."""
	if not pattern.startswith('regex:'):
		return normalize_answer_text(pattern) in normalized_answer

	expression = pattern[len('regex:') :]
	try:
		return re.search(expression, normalized_answer) is not None
	except re.error as e:
		raise BrowserBenchmarkError(f'Invalid regex expected answer pattern {pattern!r}: {type(e).__name__}') from e


def load_browser_benchmark_cases(path: Path | str) -> list[BrowserBenchmarkCase]:
	"""Read a JSONL dataset of browser cases, one per line, in file order.

	Blank lines are skipped because a hand edited file usually ends with one; everything else fails with a
	line number, and no case is dropped on the floor.

	Raises:
		BrowserBenchmarkError: unreadable file, a line that is not a valid case, or a duplicate ``case_id``.
	"""
	dataset_path = Path(path)
	try:
		text = dataset_path.read_text(encoding='utf-8')
	except OSError as e:
		raise BrowserBenchmarkError(f'Cannot read browser benchmark dataset {dataset_path}: {type(e).__name__}') from e

	cases: list[BrowserBenchmarkCase] = []
	line_of_id: dict[str, int] = {}
	for line_number, line in enumerate(text.splitlines(), start=1):
		if not line.strip():
			continue
		try:
			case = BrowserBenchmarkCase.model_validate_json(line)
		except ValidationError as e:
			raise BrowserBenchmarkError(
				f'Browser benchmark dataset line {line_number} is not a valid case: {type(e).__name__}'
			) from e

		if case.case_id in line_of_id:
			raise BrowserBenchmarkError(
				f'Browser benchmark dataset has duplicate case_id {case.case_id!r} on lines {line_of_id[case.case_id]} and {line_number}'
			)
		line_of_id[case.case_id] = line_number
		cases.append(case)

	if not cases:
		raise BrowserBenchmarkError(f'Browser benchmark dataset {dataset_path} contains no cases')
	return cases


class BrowserBenchmarkRunResult(BaseModel):
	"""One live run: what the agent did, what the checker said, and what the evidence said.

	Both readings stay separate on purpose. ``answer_check_passed`` and the claim status distribution
	often disagree, and that disagreement is the finding, so neither is folded into the other and no
	combined score is offered. Evidence counts, timings and step counts ride along because a failure that
	produced no evidence at all is a different story from one that produced thirty nodes and verified none
	of them. No answer text, page content, screenshot or credential is kept here.

	The ``postprocess_llm_*`` counters are the retry telemetry of one run, in the shape described by
	:func:`browser_use.evidence.retrying_llm.stats_delta`: a difference measured across that run, never a
	total accumulated since the benchmark started, because one wrapper is shared by every run. They are
	``None`` when the pipeline never started, since no post-processing call was made to count, and that is
	not the same claim as ``0``. A pipeline that ran cleanly reports ``postprocess_llm_retry_count=0``, and
	a pipeline that died still reports the attempts it spent before it died.
	"""

	case_id: str = Field(description='BrowserBenchmarkCase.case_id')
	task: str = Field(description='Task text, copied from the case')
	tags: list[str] = Field(default_factory=list, description='Case tags, for grouping in reports')
	repeat_index: int = Field(default=1, ge=1, description='1-based index of this run within --repeats')
	agent_completed: bool = Field(default=False, description='Agent.run() returned and reported done')
	final_answer_present: bool = Field(default=False, description='history.final_result() produced an answer')
	answer_check_passed: bool | None = Field(default=None, description='None when there was no answer to check')
	matched_patterns: list[str] = Field(default_factory=list, description='Expected patterns the answer covered')
	missing_patterns: list[str] = Field(default_factory=list, description='Expected patterns the answer lacked')
	browser_step_count: int | None = Field(default=None, ge=0, description='AgentHistoryList.number_of_steps(), if reported')
	evidence_count: int = Field(default=0, ge=0, description='Evidence nodes captured for this run')
	pipeline_completed: bool = Field(default=False, description='The WebEvidence pipeline ran through to a report')
	claim_count: int | None = Field(default=None, ge=0, description='Claims extracted from the answer')
	supported_claim_count: int | None = Field(default=None, ge=0)
	partial_claim_count: int | None = Field(default=None, ge=0)
	unsupported_claim_count: int | None = Field(default=None, ge=0)
	contradicted_claim_count: int | None = Field(default=None, ge=0)
	conflicted_claim_count: int | None = Field(default=None, ge=0)
	no_evidence_claim_count: int | None = Field(default=None, ge=0)
	evidence_coverage_rate: float | None = Field(
		default=None, ge=0.0, le=1.0, description='Share of claims with at least one candidate'
	)
	fully_supported: bool | None = Field(
		default=None, description='True only when there is at least one claim and every claim is SUPPORTED'
	)
	agent_elapsed_seconds: float | None = Field(default=None, ge=0.0, description='Wall clock of the browser run')
	pipeline_elapsed_seconds: float | None = Field(default=None, ge=0.0, description='Wall clock of the post-processing')
	postprocess_llm_logical_calls: int | None = Field(
		default=None, ge=0, description='Post-processing calls this run asked for; None when the pipeline never started'
	)
	postprocess_llm_attempts: int | None = Field(
		default=None, ge=0, description='Post-processing calls that actually reached the provider'
	)
	postprocess_llm_retry_count: int | None = Field(
		default=None, ge=0, description='Extra attempts made after a failed attempt, 0 for a run that never retried'
	)
	postprocess_llm_recovered_calls: int | None = Field(
		default=None, ge=0, description='Calls that succeeded only because an earlier attempt was retried'
	)
	postprocess_llm_failed_calls: int | None = Field(
		default=None, ge=0, description='Calls that ran out of attempts and re-raised'
	)
	failure_stage: BrowserBenchmarkFailureStage | None = Field(default=None, description='Where the run stopped, if it did')
	failure_type: str | None = Field(default=None, description='Exception type name only, never its message')
	report_json_path: str | None = Field(default=None, description='Where report.json was written for this run')
	report_markdown_path: str | None = Field(default=None, description='Where report.md was written for this run')

	@model_validator(mode='after')
	def _check_consistency(self) -> 'BrowserBenchmarkRunResult':
		"""Keep an unreadable run from pretending it measured something."""
		if self.answer_check_passed is None and (self.matched_patterns or self.missing_patterns):
			raise ValueError(f'run {self.case_id!r} has patterns recorded without an answer check')
		if self.pipeline_completed and self.claim_count is None:
			raise ValueError(f'run {self.case_id!r} claims a completed pipeline with no claim count')
		if not self.pipeline_completed and self.fully_supported is not None:
			raise ValueError(f'run {self.case_id!r} reports fully_supported without a pipeline result')
		if self.claim_count == 0 and self.fully_supported:
			raise ValueError(f'run {self.case_id!r} cannot be fully supported with no claims')
		if self.failure_stage is not None and self.pipeline_completed:
			if self.failure_stage is not BrowserBenchmarkFailureStage.OUTPUT_WRITE:
				# Only a failed write can follow a completed pipeline: the analysis exists, the artifact does not.
				raise ValueError(f'run {self.case_id!r} both failed at {self.failure_stage.value} and completed the pipeline')
		telemetry = (
			self.postprocess_llm_logical_calls,
			self.postprocess_llm_attempts,
			self.postprocess_llm_retry_count,
			self.postprocess_llm_recovered_calls,
			self.postprocess_llm_failed_calls,
		)
		if any(value is None for value in telemetry) and not all(value is None for value in telemetry):
			# The counters only make sense as one delta, and a half-filled record would feed a phantom zero
			# into the summary totals.
			raise ValueError(f'run {self.case_id!r} has partial post-processing retry telemetry')
		return self


def run_result_with_pipeline(
	run: BrowserBenchmarkRunResult,
	pipeline: WebEvidencePipelineResult,
) -> BrowserBenchmarkRunResult:
	"""Copy a run record with the claim-level fields filled from a completed pipeline result.

	The input is left untouched, so a failure while writing outputs cannot leave a half-written record
	behind, and the mapping is one place rather than spread through the runner.
	"""
	summary = pipeline.report.summary
	status_counts = {status: 0 for status in VerificationStatus}
	for section in pipeline.report.claims:
		status_counts[section.status] += 1

	return run.model_copy(
		update={
			'pipeline_completed': True,
			'claim_count': summary.claim_count,
			'supported_claim_count': summary.supported_claim_count,
			'partial_claim_count': summary.partial_claim_count,
			'unsupported_claim_count': summary.unsupported_claim_count,
			'contradicted_claim_count': summary.contradicted_claim_count,
			'conflicted_claim_count': summary.conflicted_claim_count,
			'no_evidence_claim_count': summary.no_evidence_claim_count,
			'evidence_coverage_rate': summary.evidence_coverage_rate,
			'fully_supported': summary.claim_count > 0 and summary.supported_claim_count == summary.claim_count,
			'evidence_count': pipeline.evidence_count,
		}
	)


def run_result_with_retry_stats(
	run: BrowserBenchmarkRunResult,
	stats: LLMRetryStats,
) -> BrowserBenchmarkRunResult:
	"""Copy a run record with retry telemetry taken from the counters of one wrapper.

	``stats`` is expected to be a delta produced by :func:`browser_use.evidence.retrying_llm.stats_delta`
	for this run alone. Pass a cumulative snapshot instead and every run would report the whole benchmark's
	running total, which is the failure mode the per-run snapshots exist to prevent. The exception type
	counts stay out of the record: they name nothing the aggregate needs.
	"""
	return run.model_copy(
		update={
			'postprocess_llm_logical_calls': stats.logical_invocation_count,
			'postprocess_llm_attempts': stats.attempt_count,
			'postprocess_llm_retry_count': stats.retry_count,
			'postprocess_llm_recovered_calls': stats.recovered_invocation_count,
			'postprocess_llm_failed_calls': stats.failed_invocation_count,
		}
	)


def _rate(values: Sequence[bool | None]) -> float | None:
	"""Mean over the values that were measured, or None when none were."""
	measured = [value for value in values if value is not None]
	if not measured:
		return None
	return sum(1 for value in measured if value) / len(measured)


def _mean(values: Iterable[float | None]) -> float | None:
	measured = [value for value in values if value is not None]
	if not measured:
		return None
	return sum(measured) / len(measured)


def _total(values: Iterable[int | None]) -> int | None:
	"""Sum over the values that were measured, or None when none were."""
	measured = [value for value in values if value is not None]
	if not measured:
		return None
	return sum(measured)


def _unique(values: Sequence[str]) -> list[str]:
	"""De-duplicate while keeping first-seen order, so repeats list a case once."""
	seen: set[str] = set()
	unique: list[str] = []
	for value in values:
		if value not in seen:
			seen.add(value)
			unique.append(value)
	return unique


class BrowserBenchmarkSummary(BaseModel):
	"""Aggregate over runs. Claim rates use pooled claim counts, not an average of per-run percentages.

	Averaging per-run rates would let a run with one claim weigh as much as a run with eight, which hides
	exactly the pattern this benchmark looks for: a few unsupported claims inside otherwise solid answers.
	A ``None`` rate means nothing was measured, never that it scored zero.

	The retry totals add up the per-run post-processing counters over the runs that reported them, so a
	``None`` there means no run reported telemetry, not that the pipeline made no calls. Exception type
	counts stay out of these totals deliberately. ``runs_recovered_by_retry_count`` is an observed count,
	not a counterfactual: it says a call in that run failed once and then succeeded, never that the run
	would have failed without retry.
	"""

	run_count: int = Field(default=0, ge=0, description='Runs aggregated')
	case_count: int = Field(default=0, ge=0, description='Distinct cases among those runs')
	agent_completion_rate: float | None = Field(default=None, ge=0.0, le=1.0, description='Runs whose agent reported done')
	final_answer_rate: float | None = Field(default=None, ge=0.0, le=1.0, description='Runs that produced a final answer')
	answer_check_pass_rate: float | None = Field(
		default=None, ge=0.0, le=1.0, description='Raw task success over the runs that had an answer to check'
	)
	pipeline_completion_rate: float | None = Field(default=None, ge=0.0, le=1.0)
	mean_browser_steps: float | None = Field(default=None, ge=0.0, description='Mean agent steps over runs that reported a count')
	mean_evidence_count: float = Field(default=0.0, ge=0.0, description='Mean evidence nodes over all runs')
	mean_claim_count: float | None = Field(default=None, ge=0.0, description='Mean claims extracted over pipeline-complete runs')
	mean_evidence_coverage_rate: float | None = Field(default=None, ge=0.0, le=1.0)
	supported_claim_rate: float | None = Field(default=None, ge=0.0, le=1.0)
	partial_claim_rate: float | None = Field(default=None, ge=0.0, le=1.0)
	unsupported_claim_rate: float | None = Field(default=None, ge=0.0, le=1.0)
	contradicted_claim_rate: float | None = Field(default=None, ge=0.0, le=1.0)
	conflicted_claim_rate: float | None = Field(default=None, ge=0.0, le=1.0)
	no_evidence_claim_rate: float | None = Field(default=None, ge=0.0, le=1.0)
	fully_supported_run_rate: float | None = Field(default=None, ge=0.0, le=1.0)
	mean_agent_elapsed_seconds: float | None = Field(default=None, ge=0.0)
	mean_pipeline_elapsed_seconds: float | None = Field(default=None, ge=0.0)
	total_claims_verified: int = Field(default=0, ge=0, description='Pooled claim count, the denominator of the claim rates')
	total_postprocess_llm_logical_calls: int | None = Field(
		default=None, ge=0, description='Post-processing calls asked for, summed over runs that reported telemetry'
	)
	total_postprocess_llm_attempts: int | None = Field(
		default=None, ge=0, description='Provider calls actually made for post-processing, same denominator'
	)
	total_postprocess_llm_retries: int | None = Field(default=None, ge=0, description='Extra attempts spent across those runs')
	total_postprocess_llm_recovered_calls: int | None = Field(
		default=None, ge=0, description='Calls that succeeded only after a retry, across those runs'
	)
	total_postprocess_llm_failed_calls: int | None = Field(
		default=None, ge=0, description='Calls that ran out of attempts, across those runs'
	)
	runs_with_postprocess_retry_count: int = Field(
		default=0, ge=0, description='Runs that spent at least one extra attempt, completed or not'
	)
	runs_recovered_by_retry_count: int = Field(
		default=0, ge=0, description='Completed runs in which at least one call recovered through retry'
	)
	answer_pass_but_not_fully_supported_case_ids: list[str] = Field(
		default_factory=list, description='Task passed the checker while some claim lacked full support'
	)
	answer_fail_case_ids: list[str] = Field(default_factory=list, description='Runs whose answer missed a required pattern')
	pipeline_fail_case_ids: list[str] = Field(default_factory=list, description='Runs that never reached a report')
	no_evidence_case_ids: list[str] = Field(default_factory=list, description='Runs with at least one NO_EVIDENCE claim')
	contradicted_case_ids: list[str] = Field(default_factory=list, description='Runs with at least one CONTRADICTED claim')
	conflicted_case_ids: list[str] = Field(default_factory=list, description='Runs with at least one CONFLICTED claim')
	failure_case_ids_by_stage: dict[str, list[str]] = Field(default_factory=dict, description='Stage name to case ids')


class EvidenceBrowserBenchmarkResult(BaseModel):
	"""The whole live benchmark: aggregates plus the per-run detail that produced them."""

	summary: BrowserBenchmarkSummary = Field(default_factory=BrowserBenchmarkSummary)
	runs: list[BrowserBenchmarkRunResult] = Field(default_factory=list, description='One entry per run, in execution order')


def summarize_browser_runs(runs: Sequence[BrowserBenchmarkRunResult]) -> BrowserBenchmarkSummary:
	"""Aggregate run records into a summary.

	Every rate leaves its denominator explicit: task success is measured over runs that produced an answer,
	claim rates over the pooled claims of pipeline-complete runs, and ``fully_supported`` over runs whose
	pipeline finished. Those denominators differ on purpose, since a run that never answered cannot fail a
	task check it was never given.

	The retry totals are the exception to that pattern: they cover every run that reported telemetry, which
	includes a run that spent three attempts and still failed. That is the number the phase is about, and
	dropping it from the total would make a retrying benchmark look cheaper than it was.
	"""
	verified = [run for run in runs if run.pipeline_completed and run.claim_count is not None]
	total_claims = sum(run.claim_count or 0 for run in verified)

	def claim_rate(field: str) -> float | None:
		if not total_claims:
			return None
		return sum(getattr(run, field) or 0 for run in verified) / total_claims

	failures_by_stage: dict[str, list[str]] = {}
	for run in runs:
		if run.failure_stage is not None:
			failures_by_stage.setdefault(run.failure_stage.value, []).append(run.case_id)

	return BrowserBenchmarkSummary(
		run_count=len(runs),
		case_count=len({run.case_id for run in runs}),
		agent_completion_rate=_rate([run.agent_completed for run in runs]),
		final_answer_rate=_rate([run.final_answer_present for run in runs]),
		answer_check_pass_rate=_rate([run.answer_check_passed for run in runs]),
		pipeline_completion_rate=_rate([run.pipeline_completed for run in runs]),
		mean_browser_steps=_mean([run.browser_step_count for run in runs]),
		mean_evidence_count=sum(run.evidence_count for run in runs) / len(runs) if runs else 0.0,
		mean_claim_count=_mean([run.claim_count for run in verified]),
		mean_evidence_coverage_rate=_mean([run.evidence_coverage_rate for run in verified]),
		supported_claim_rate=claim_rate('supported_claim_count'),
		partial_claim_rate=claim_rate('partial_claim_count'),
		unsupported_claim_rate=claim_rate('unsupported_claim_count'),
		contradicted_claim_rate=claim_rate('contradicted_claim_count'),
		conflicted_claim_rate=claim_rate('conflicted_claim_count'),
		no_evidence_claim_rate=claim_rate('no_evidence_claim_count'),
		fully_supported_run_rate=_rate([run.fully_supported for run in runs]),
		mean_agent_elapsed_seconds=_mean([run.agent_elapsed_seconds for run in runs]),
		mean_pipeline_elapsed_seconds=_mean([run.pipeline_elapsed_seconds for run in runs]),
		total_claims_verified=total_claims,
		total_postprocess_llm_logical_calls=_total([run.postprocess_llm_logical_calls for run in runs]),
		total_postprocess_llm_attempts=_total([run.postprocess_llm_attempts for run in runs]),
		total_postprocess_llm_retries=_total([run.postprocess_llm_retry_count for run in runs]),
		total_postprocess_llm_recovered_calls=_total([run.postprocess_llm_recovered_calls for run in runs]),
		total_postprocess_llm_failed_calls=_total([run.postprocess_llm_failed_calls for run in runs]),
		runs_with_postprocess_retry_count=sum(1 for run in runs if (run.postprocess_llm_retry_count or 0) > 0),
		runs_recovered_by_retry_count=sum(
			1 for run in runs if run.pipeline_completed and (run.postprocess_llm_recovered_calls or 0) > 0
		),
		answer_pass_but_not_fully_supported_case_ids=_unique(
			[run.case_id for run in runs if run.answer_check_passed is True and run.fully_supported is False]
		),
		answer_fail_case_ids=_unique([run.case_id for run in runs if run.answer_check_passed is False]),
		pipeline_fail_case_ids=_unique([run.case_id for run in runs if not run.pipeline_completed]),
		no_evidence_case_ids=_unique([run.case_id for run in runs if (run.no_evidence_claim_count or 0) > 0]),
		contradicted_case_ids=_unique([run.case_id for run in runs if (run.contradicted_claim_count or 0) > 0]),
		conflicted_case_ids=_unique([run.case_id for run in runs if (run.conflicted_claim_count or 0) > 0]),
		failure_case_ids_by_stage={stage: _unique(ids) for stage, ids in sorted(failures_by_stage.items())},
	)
