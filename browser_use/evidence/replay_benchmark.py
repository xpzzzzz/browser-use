"""Fixed-input replay benchmark: the same frozen answer and evidence, pushed through the pipeline again.

Phases 9B and 9C both measured pipeline completion on live browser runs and both came out at 9/14, which
is not a comparison. A browser run varies in step count, page state, how much evidence got captured and
what the final answer said, and provider latency moved on its own between those two sessions, so a
difference between the two numbers could not be attributed to the retry policy. This module takes the
browser out of the measurement: a fixture pins one task, one final answer and one ``EvidenceNode`` list,
so the only things that can still change across repeats are the provider and the model's own behaviour.

Two readings stay deliberately separate:

    reliability        -> did the pipeline reach a report            (completion rate)
    semantic stability -> did the completed runs agree on claims and
                          verdicts                                    (signature counts)

A pipeline that always completes while extracting different claims each time is not stable, and a
pipeline that fails twice has not been shown to be semantically wrong. Those are different findings that
lead to different next experiments, so nothing here folds them into one score.

The retry counters of a run come from :func:`browser_use.evidence.retrying_llm.stats_delta`, so one
wrapper can serve the whole benchmark while every record still describes only its own run. Unlike Phase
9B, this artifact also keeps the exception class names per run: knowing that post-processing failed 24
times is weak evidence, knowing that 21 of them were timeouts and 3 were validation errors says what to
fix. Only class names are kept. An answer, page text, prompt, credential and provider exception message
never enter a record here, which is what lets these files be shared.

Nothing in this file constructs a model, reads a key, opens a browser or touches a network. The live half
is ``scripts/run_webevidence_replay_benchmark.py``; everything here stays runnable under pytest with a
fake chat model, which is how a fake provider's retry telemetry gets tested exactly like a real one.
"""

import re
import time
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from browser_use.evidence.claims import Claim, ClaimSet, NonBlankString
from browser_use.evidence.models import EvidenceNode
from browser_use.evidence.pipeline import PipelineStage, WebEvidencePipeline, WebEvidencePipelineResult
from browser_use.evidence.retrying_llm import LLMRetryStats, stats_delta
from browser_use.evidence.verification import VerificationResult, VerificationStatus

# Recorded identifiers are type names and stage labels, never prose. A provider exception message can
# echo the whole request it was given -- the answer under verification plus scraped page text -- so the
# shape is enforced at the model boundary rather than trusted to the caller.
_EXCEPTION_TYPE_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_.]*')
_FAILURE_TYPE_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_.]*(:[A-Z][A-Z0-9_]*)?')
_WHITESPACE_PATTERN = re.compile(r'\s+')

# The six statuses partition a claim set, so each one owns a count field on a completed replay record.
_STATUS_COUNT_FIELDS: dict[str, str] = {status.value: f'{status.name.lower()}_claim_count' for status in VerificationStatus}


class ReplayBenchmarkError(RuntimeError):
	"""Raised when a replay fixture cannot be trusted, or two artifacts cannot be compared.

	A fixture identifier carries every aggregate, so a duplicate id, a generated ``evidence_id`` or a
	malformed line is refused with its line number rather than skipped: a quietly dropped fixture changes
	what every rate in the summary means while the file still looks fine.
	"""


class ReplayFixture(BaseModel):
	"""One frozen pipeline input: a task, the answer the agent gave for it, and the captured evidence.

	The three fields are exactly what :meth:`browser_use.evidence.pipeline.WebEvidencePipeline.analyze`
	consumes, so replaying a fixture exercises the same code path a live run did. Everything the browser
	contributed is already baked in here, which is the point: nothing downstream can drift for a reason
	this file cannot see.

	Identifiers are pinned rather than generated. ``task_id`` and every ``evidence_id`` must have come
	from the fixture itself, because a regenerated id makes two repeats of the same fixture different
	inputs, and evidence ids are what a reranking or verification result is keyed by.
	"""

	fixture_id: NonBlankString = Field(description='Stable identifier, unique within a dataset')
	task_id: NonBlankString = Field(description='Task id the frozen evidence and claims belong to')
	task: str = Field(description='Task text, replayed verbatim')
	answer: str = Field(description='The final answer to verify, replayed verbatim')
	evidence_nodes: list[EvidenceNode] = Field(default_factory=list, description='Frozen evidence, in capture order')
	tags: list[str] = Field(default_factory=list, description='Free-form grouping, e.g. claim-count or failure-replay')
	description: str = Field(default='', description='What this fixture is meant to probe, and where it came from')

	@model_validator(mode='after')
	def _check_frozen_input(self) -> 'ReplayFixture':
		"""Refuse a fixture that could not be replayed identically: a generated id, a dupe, a foreign node."""
		evidence_ids = [node.evidence_id for node in self.evidence_nodes]
		if len(set(evidence_ids)) != len(evidence_ids):
			raise ReplayBenchmarkError(f'Replay fixture {self.fixture_id!r} contains duplicate evidence_id')

		generated = [node.evidence_id for node in self.evidence_nodes if 'evidence_id' not in node.model_fields_set]
		if generated:
			raise ReplayBenchmarkError(
				f'Replay fixture {self.fixture_id!r} has {len(generated)} evidence node(s) with a generated '
				f'evidence_id, first: {generated[0]!r}'
			)

		# Nodes carry the task id of the run that captured them, so a mismatch means two runs' artifacts
		# were pasted into one fixture -- an input no live run ever had.
		foreign = [node.evidence_id for node in self.evidence_nodes if node.task_id != self.task_id]
		if foreign:
			raise ReplayBenchmarkError(
				f'Replay fixture {self.fixture_id!r} has {len(foreign)} evidence node(s) from another task_id, '
				f'first: {foreign[0]!r}'
			)
		return self


def load_replay_fixtures(path: Path | str) -> list[ReplayFixture]:
	"""Read a JSONL dataset of replay fixtures, one per line, in file order.

	Blank lines are skipped because a hand edited file usually ends with one; anything else fails with its
	line number, and no fixture is dropped on the floor.

	Raises:
		ReplayBenchmarkError: unreadable file, a line that is not a valid fixture, or a duplicate
			``fixture_id``.
	"""
	dataset_path = Path(path)
	try:
		text = dataset_path.read_text(encoding='utf-8')
	except OSError as e:
		raise ReplayBenchmarkError(f'Cannot read replay fixture dataset {dataset_path}: {type(e).__name__}') from e

	fixtures: list[ReplayFixture] = []
	line_of_id: dict[str, int] = {}
	for line_number, line in enumerate(text.splitlines(), start=1):
		if not line.strip():
			continue
		try:
			fixture = ReplayFixture.model_validate_json(line)
		except ReplayBenchmarkError as e:
			# The fixture validator already kept its own message safe; only the line number is missing.
			raise ReplayBenchmarkError(f'Replay fixture dataset line {line_number}: {e}') from e
		except ValidationError as e:
			# The exception type only: a pydantic message embeds the offending value, which here is
			# captured page text.
			raise ReplayBenchmarkError(
				f'Replay fixture dataset line {line_number} is not a valid fixture: {type(e).__name__}'
			) from e

		if fixture.fixture_id in line_of_id:
			raise ReplayBenchmarkError(
				f'Replay fixture dataset has duplicate fixture_id {fixture.fixture_id!r} '
				f'on lines {line_of_id[fixture.fixture_id]} and {line_number}'
			)
		line_of_id[fixture.fixture_id] = line_number
		fixtures.append(fixture)

	if not fixtures:
		raise ReplayBenchmarkError(f'Replay fixture dataset {dataset_path} contains no fixtures')
	return fixtures


def select_fixtures(
	fixtures: Sequence[ReplayFixture],
	only: Sequence[str] | None,
) -> list[ReplayFixture]:
	"""Narrow a dataset to named fixtures for a smoke run, and say so when a name was typed wrong."""
	selected = list(fixtures)
	if only:
		wanted = list(dict.fromkeys(only))
		known = {fixture.fixture_id for fixture in selected}
		unknown = [fixture_id for fixture_id in wanted if fixture_id not in known]
		if unknown:
			raise ReplayBenchmarkError(f'Unknown --fixture value(s): {", ".join(unknown)}')
		selected = [fixture for fixture in selected if fixture.fixture_id in wanted]
	if not selected:
		raise ReplayBenchmarkError('The selected replay fixture set is empty')
	return selected


def normalize_claim_text(text: str) -> str:
	"""NFKC, casefolded, whitespace-collapsed claim text.

	A model writes "Example Domain" where the previous run wrote "example domain", or splits a sentence
	over a line break, and none of that is a different claim. Normalizing first means a signature counts
	real disagreements instead of formatting noise.
	"""
	collapsed = _WHITESPACE_PATTERN.sub(' ', unicodedata.normalize('NFKC', text or '').casefold())
	return collapsed.strip()


def ordered_claims(claim_set: ClaimSet) -> list[Claim]:
	"""The claims of a set in the order the pipeline treats them as.

	``Claim.order``, not the list position, is what ``ClaimVerifier`` iterates by, so both signatures have
	to use the same key or a run's verdict sequence would not line up with its claim sequence.
	"""
	return sorted(claim_set.claims, key=lambda claim: (claim.order, claim.claim_id))


def claim_signature(claim_set: ClaimSet) -> list[str]:
	"""Normalized claim texts in order: how the answer got split this time.

	Claim ids stay out because they are generated fresh on every run, so an id-based comparison of two
	repeats would always report total disagreement.
	"""
	return [normalize_claim_text(claim.text) for claim in ordered_claims(claim_set)]


def status_signature(claim_set: ClaimSet, verification_result: VerificationResult) -> list[str]:
	"""Claim-level statuses in claim order, e.g. ``["SUPPORTED", "PARTIAL"]``.

	The pairing is by ``claim_id`` within one run, never by position across runs: when two repeats split
	the answer differently their sequences do not mean the same thing, and this harness deliberately does
	not pretend otherwise. Cross-run claim matching is a separate design.

	Raises:
		ReplayBenchmarkError: when the verification result does not cover every claim exactly once, which
			means the two objects are not from one pipeline run.
	"""
	statuses: dict[str, str] = {}
	for verification in verification_result.verifications:
		if verification.claim_id in statuses:
			raise ReplayBenchmarkError(f'Verification result reports claim_id {verification.claim_id!r} twice')
		statuses[verification.claim_id] = verification.status.value

	missing = [claim.claim_id for claim in ordered_claims(claim_set) if claim.claim_id not in statuses]
	if missing:
		raise ReplayBenchmarkError(f'{len(missing)} claim(s) of this run have no verification status')
	return [statuses[claim.claim_id] for claim in ordered_claims(claim_set)]


class ReplayRunResult(BaseModel):
	"""One replay of one fixture: whether the pipeline finished, what it concluded, and what it cost.

	Reliability and semantics are recorded side by side but never combined. ``pipeline_completed`` is the
	reliability fact; the claim counts and the two signatures are the semantic observation, and they are
	``None`` on a run that never reached a report rather than zero, because "verification did not happen"
	and "verification found nothing" are different statements.

	``elapsed_seconds`` is pure post-processing time -- there is no browser in this benchmark -- which
	make it the cleanest latency signal the project has, though still subject to provider load and network
	conditions, so it is a range over repeats rather than a constant.

	The ``postprocess_llm_*`` counters and ``exception_type_counts`` are one
	:func:`browser_use.evidence.retrying_llm.stats_delta` over this run, never a total since the
	benchmark started, because a single wrapper serves every run.
	"""

	fixture_id: str = Field(description='ReplayFixture.fixture_id that was replayed')
	repeat_index: int = Field(ge=1, description='1-based index of this replay within --repeats')
	max_attempts: int = Field(ge=1, description='Retry budget this run was replayed under')
	pipeline_completed: bool = Field(default=False, description='The pipeline ran through to a report')
	failure_stage: PipelineStage | None = Field(default=None, description='Which stage stopped, if any')
	failure_type: str | None = Field(default=None, description='Exception type name, plus stage for a pipeline error')
	elapsed_seconds: float = Field(ge=0.0, description='Wall clock of the post-processing for this run')
	claim_count: int | None = Field(default=None, ge=0, description='Claims extracted from the frozen answer')
	supported_claim_count: int | None = Field(default=None, ge=0)
	partial_claim_count: int | None = Field(default=None, ge=0)
	unsupported_claim_count: int | None = Field(default=None, ge=0)
	contradicted_claim_count: int | None = Field(default=None, ge=0)
	conflicted_claim_count: int | None = Field(default=None, ge=0)
	no_evidence_claim_count: int | None = Field(default=None, ge=0)
	postprocess_llm_logical_calls: int = Field(default=0, ge=0)
	postprocess_llm_attempts: int = Field(default=0, ge=0, description='Post-processing calls that reached the provider')
	postprocess_llm_retry_count: int = Field(default=0, ge=0, description='Extra attempts made after a failed one')
	postprocess_llm_recovered_calls: int = Field(default=0, ge=0, description='Calls that succeeded only after a retry')
	postprocess_llm_failed_calls: int = Field(default=0, ge=0, description='Calls that ran out of attempts')
	exception_type_counts: dict[str, int] = Field(
		default_factory=dict, description='Exception class name to count for this run; never a message'
	)
	claim_signature: list[str] | None = Field(default=None, description='Normalized claim texts in claim order')
	status_signature: list[str] | None = Field(default=None, description='Status values in claim order')

	@model_validator(mode='after')
	def _check_consistency(self) -> 'ReplayRunResult':
		"""Keep a run that stopped from pretending it measured anything, and keep the counters honest."""
		if self.pipeline_completed:
			if self.failure_stage is not None or self.failure_type is not None:
				raise ValueError(f'replay {self.fixture_id!r}#{self.repeat_index} completed and reports a failure')
			if self.claim_count is None:
				raise ValueError(f'replay {self.fixture_id!r}#{self.repeat_index} completed with no claim count')
			if self.claim_signature is None or self.status_signature is None:
				raise ValueError(f'replay {self.fixture_id!r}#{self.repeat_index} completed without its signatures')
			for status_value, field in _STATUS_COUNT_FIELDS.items():
				count = getattr(self, field)
				if count is None:
					raise ValueError(f'replay {self.fixture_id!r}#{self.repeat_index} completed without its {field}')
				# The six statuses partition the claim set, so the counts are the signature restated. A record
				# that breaks that was assembled from two different runs.
				if count != self.status_signature.count(status_value):
					raise ValueError(
						f'replay {self.fixture_id!r}#{self.repeat_index} has a {field} that does not match its status signature'
					)
			unknown = [value for value in self.status_signature if value not in _STATUS_COUNT_FIELDS]
			if unknown:
				raise ValueError(
					f'replay {self.fixture_id!r}#{self.repeat_index} has {len(unknown)} status value(s) that are not '
					f'statuses, first one: {unknown[0]!r}'
				)
		else:
			if self.failure_type is None:
				raise ValueError(f'replay {self.fixture_id!r}#{self.repeat_index} failed without a failure type')
			unmeasured = (
				self.claim_count,
				self.supported_claim_count,
				self.partial_claim_count,
				self.unsupported_claim_count,
				self.contradicted_claim_count,
				self.conflicted_claim_count,
				self.no_evidence_claim_count,
				self.claim_signature,
				self.status_signature,
			)
			if any(value is not None for value in unmeasured):
				raise ValueError(f'replay {self.fixture_id!r}#{self.repeat_index} reports claim results it never verified')

		if self.claim_signature is not None and len(self.claim_signature) != self.claim_count:
			raise ValueError(
				f'replay {self.fixture_id!r}#{self.repeat_index} counts {self.claim_count} claims '
				f'but signs {len(self.claim_signature)}'
			)
		if self.status_signature is not None and len(self.status_signature) != self.claim_count:
			raise ValueError(
				f'replay {self.fixture_id!r}#{self.repeat_index} counts {self.claim_count} claims '
				f'but signs {len(self.status_signature)} statuses'
			)

		if self.failure_type is not None and not _FAILURE_TYPE_PATTERN.fullmatch(self.failure_type):
			raise ValueError(f'replay {self.fixture_id!r}#{self.repeat_index} has a failure_type that is not a bare type name')
		for name, count in self.exception_type_counts.items():
			if not _EXCEPTION_TYPE_PATTERN.fullmatch(name):
				raise ValueError(f'replay {self.fixture_id!r}#{self.repeat_index} has an exception type name that is not bare')
			if count < 1:
				raise ValueError(f'replay {self.fixture_id!r}#{self.repeat_index} counts {name!r} {count} times')

		if self.postprocess_llm_attempts < self.postprocess_llm_logical_calls:
			raise ValueError(f'replay {self.fixture_id!r}#{self.repeat_index} reached fewer providers than it asked to reach')
		if self.postprocess_llm_attempts != (self.postprocess_llm_logical_calls + self.postprocess_llm_retry_count):
			raise ValueError(f'replay {self.fixture_id!r}#{self.repeat_index} has attempts that are not calls plus retries')
		if self.postprocess_llm_recovered_calls + self.postprocess_llm_failed_calls > self.postprocess_llm_logical_calls:
			raise ValueError(f'replay {self.fixture_id!r}#{self.repeat_index} resolved more calls than it made')
		# Every failed attempt either bought a retry or exhausted the budget, so the two account for all
		# of them. A mismatch means the counters came from different snapshots.
		if sum(self.exception_type_counts.values()) != self.postprocess_llm_retry_count + self.postprocess_llm_failed_calls:
			raise ValueError(f'replay {self.fixture_id!r}#{self.repeat_index} has exception counts that do not match its retries')
		return self


def _retry_counter_fields(stats: LLMRetryStats) -> dict[str, Any]:
	return {
		'postprocess_llm_logical_calls': stats.logical_invocation_count,
		'postprocess_llm_attempts': stats.attempt_count,
		'postprocess_llm_retry_count': stats.retry_count,
		'postprocess_llm_recovered_calls': stats.recovered_invocation_count,
		'postprocess_llm_failed_calls': stats.failed_invocation_count,
		'exception_type_counts': dict(stats.exception_type_counts),
	}


def _completed_run(
	fixture: ReplayFixture,
	*,
	repeat_index: int,
	max_attempts: int,
	elapsed_seconds: float,
	pipeline: WebEvidencePipelineResult,
	stats: LLMRetryStats,
) -> ReplayRunResult:
	"""Build the record of a replay that reached a report, from that report and this replay's counters."""
	summary = pipeline.report.summary
	return ReplayRunResult(
		fixture_id=fixture.fixture_id,
		repeat_index=repeat_index,
		max_attempts=max_attempts,
		pipeline_completed=True,
		elapsed_seconds=elapsed_seconds,
		claim_count=summary.claim_count,
		supported_claim_count=summary.supported_claim_count,
		partial_claim_count=summary.partial_claim_count,
		unsupported_claim_count=summary.unsupported_claim_count,
		contradicted_claim_count=summary.contradicted_claim_count,
		conflicted_claim_count=summary.conflicted_claim_count,
		no_evidence_claim_count=summary.no_evidence_claim_count,
		claim_signature=claim_signature(pipeline.claim_set),
		status_signature=status_signature(pipeline.claim_set, pipeline.verification_result),
		**_retry_counter_fields(stats),
	)


def _failed_run(
	fixture: ReplayFixture,
	*,
	repeat_index: int,
	max_attempts: int,
	elapsed_seconds: float,
	cause: Exception,
	stats: LLMRetryStats,
) -> ReplayRunResult:
	"""Build the record of a replay that stopped, naming only the stage and the exception type.

	A pipeline error already carries the stage that raised, and that label is appended rather than the
	message: without it every post-processing failure reads as the same anonymous
	``WebEvidencePipelineError``, which is exactly the ambiguity Phase 9D exists to remove.
	"""
	failure_type = type(cause).__name__
	stage = getattr(cause, 'stage', None)
	if isinstance(stage, PipelineStage):
		failure_type = f'{failure_type}:{stage.value}'
	return ReplayRunResult(
		fixture_id=fixture.fixture_id,
		repeat_index=repeat_index,
		max_attempts=max_attempts,
		pipeline_completed=False,
		failure_stage=stage if isinstance(stage, PipelineStage) else None,
		failure_type=failure_type,
		elapsed_seconds=elapsed_seconds,
		**_retry_counter_fields(stats),
	)


async def run_replay(
	fixture: ReplayFixture,
	*,
	pipeline: WebEvidencePipeline,
	repeat_index: int,
	max_attempts: int,
	stats: Callable[[], LLMRetryStats],
	clock: Callable[[], float] = time.perf_counter,
) -> ReplayRunResult:
	"""Replay one fixture once, and record what happened without ever raising.

	The fixture's evidence list is copied before it is handed over, so nothing downstream can reach back
	into the frozen input, and the fixture object itself is never written to. ``stats`` is a snapshot
	provider -- normally ``RetryingChatModel.snapshot_stats`` -- read before and after the run, because
	one wrapper serves the whole benchmark and its totals would otherwise land on every record.

	A pipeline that fails is a measured run, not an aborted benchmark: the counters it spent before it
	died are still recorded, since those provider calls were real cost.
	"""
	before = stats()
	started = clock()
	try:
		result = await pipeline.analyze(
			task_id=fixture.task_id,
			task=fixture.task,
			answer=fixture.answer,
			evidence_nodes=list(fixture.evidence_nodes),
		)
	except Exception as cause:
		return _failed_run(
			fixture,
			repeat_index=repeat_index,
			max_attempts=max_attempts,
			elapsed_seconds=round(clock() - started, 3),
			cause=cause,
			stats=stats_delta(before, stats()),
		)

	return _completed_run(
		fixture,
		repeat_index=repeat_index,
		max_attempts=max_attempts,
		elapsed_seconds=round(clock() - started, 3),
		pipeline=result,
		stats=stats_delta(before, stats()),
	)


def _rate(values: Sequence[bool]) -> float | None:
	"""Share of ``True`` values, or ``None`` when there was nothing to average."""
	if not values:
		return None
	return sum(1 for value in values if value) / len(values)


def _mean(values: Sequence[float]) -> float | None:
	if not values:
		return None
	return sum(values) / len(values)


class ReplayBenchmarkSummary(BaseModel):
	"""Aggregate over replays of frozen inputs.

	Every rate names its denominator in :func:`summarize_replay_runs`. The completion rate uses all runs,
	because every run got the same input and is therefore comparable; the claim mean and the signature
	counts use only the runs that reached a report, since a failed run has nothing to compare. A ``None``
	means nothing was measured, never that it scored zero.

	``runs_recovered_by_retry_count`` is an observed count, not a counterfactual: it says a call in that
	run failed once and then succeeded, never that the run would have failed without retry.
	"""

	run_count: int = Field(default=0, ge=0, description='Replays aggregated')
	fixture_count: int = Field(default=0, ge=0, description='Distinct fixtures among those replays')
	max_attempts: int | None = Field(
		default=None, ge=1, description='The retry budget every replay used; None if they did not share one'
	)
	pipeline_completion_rate: float | None = Field(default=None, ge=0.0, le=1.0)
	failed_run_count: int = Field(default=0, ge=0, description='Replays that never reached a report')
	mean_elapsed_seconds: float | None = Field(
		default=None, ge=0.0, description='Mean post-processing seconds over all replays, failed ones included'
	)
	mean_completed_elapsed_seconds: float | None = Field(
		default=None, ge=0.0, description='Mean post-processing seconds over the replays that reached a report'
	)
	total_logical_calls: int = Field(default=0, ge=0)
	total_provider_attempts: int = Field(default=0, ge=0)
	total_retries: int = Field(default=0, ge=0)
	total_recovered_calls: int = Field(default=0, ge=0)
	total_failed_calls: int = Field(default=0, ge=0)
	runs_with_retry_count: int = Field(default=0, ge=0, description='Replays that spent at least one extra attempt')
	runs_recovered_by_retry_count: int = Field(
		default=0, ge=0, description='Completed replays in which at least one call recovered through retry'
	)
	exception_type_counts: dict[str, int] = Field(default_factory=dict, description='Exception class name to total count')
	failure_stage_counts: dict[str, int] = Field(default_factory=dict, description='Stage name to failed replay count')
	mean_claim_count: float | None = Field(default=None, ge=0.0, description='Mean claims over replays that reached a report')
	claim_signature_unique_count_by_fixture: dict[str, int] = Field(
		default_factory=dict, description='Fixture id to distinct claim signatures; 0 means no completed replay'
	)
	status_signature_unique_count_by_fixture: dict[str, int] = Field(
		default_factory=dict, description='Fixture id to distinct status sequences; 0 means no completed replay'
	)


def summarize_replay_runs(runs: Sequence[ReplayRunResult]) -> ReplayBenchmarkSummary:
	"""Aggregate replay records into one summary for one retry configuration.

	The denominators differ on purpose. Completion is over every replay: a fixture is a fixed input, so a
	failed replay and a completed one are the same experiment with a different outcome. Mean claim count
	and both signature counts are over the completed replays only, since a run that never reached a report
	never produced a claim set to compare. A fixture listed with ``0`` under a signature count therefore
	means "no completed replay to compare", not "perfectly unstable".
	"""
	completed = [run for run in runs if run.pipeline_completed]
	attempts = {run.max_attempts for run in runs}

	exception_type_counts: dict[str, int] = {}
	for run in runs:
		for name, count in run.exception_type_counts.items():
			exception_type_counts[name] = exception_type_counts.get(name, 0) + count

	failure_stage_counts: dict[str, int] = {}
	for run in runs:
		if run.failure_stage is not None:
			key = run.failure_stage.value
			failure_stage_counts[key] = failure_stage_counts.get(key, 0) + 1

	fixture_ids = sorted({run.fixture_id for run in runs})
	claim_unique = {
		fixture_id: len({tuple(run.claim_signature or []) for run in completed if run.fixture_id == fixture_id})
		for fixture_id in fixture_ids
	}
	status_unique = {
		fixture_id: len({tuple(run.status_signature or []) for run in completed if run.fixture_id == fixture_id})
		for fixture_id in fixture_ids
	}

	return ReplayBenchmarkSummary(
		run_count=len(runs),
		fixture_count=len(fixture_ids),
		max_attempts=attempts.pop() if len(attempts) == 1 else None,
		pipeline_completion_rate=_rate([run.pipeline_completed for run in runs]),
		failed_run_count=sum(1 for run in runs if not run.pipeline_completed),
		mean_elapsed_seconds=_mean([run.elapsed_seconds for run in runs]),
		mean_completed_elapsed_seconds=_mean([run.elapsed_seconds for run in completed]),
		total_logical_calls=sum(run.postprocess_llm_logical_calls for run in runs),
		total_provider_attempts=sum(run.postprocess_llm_attempts for run in runs),
		total_retries=sum(run.postprocess_llm_retry_count for run in runs),
		total_recovered_calls=sum(run.postprocess_llm_recovered_calls for run in runs),
		total_failed_calls=sum(run.postprocess_llm_failed_calls for run in runs),
		runs_with_retry_count=sum(1 for run in runs if run.postprocess_llm_retry_count > 0),
		runs_recovered_by_retry_count=sum(1 for run in completed if run.postprocess_llm_recovered_calls > 0),
		exception_type_counts=dict(sorted(exception_type_counts.items())),
		failure_stage_counts=dict(sorted(failure_stage_counts.items())),
		mean_claim_count=_mean([float(run.claim_count or 0) for run in completed]),
		claim_signature_unique_count_by_fixture=claim_unique,
		status_signature_unique_count_by_fixture=status_unique,
	)


class EvidenceReplayBenchmarkResult(BaseModel):
	"""A whole replay benchmark: the aggregates plus the per-replay detail that produced them."""

	summary: ReplayBenchmarkSummary = Field(default_factory=ReplayBenchmarkSummary)
	runs: list[ReplayRunResult] = Field(default_factory=list, description='One entry per replay, in execution order')


class ReplayComparison(BaseModel):
	"""Descriptive differences between two replay artifacts, in a fixed field order.

	Deliberately no p-value, no significance test and no confidence interval. Fifteen replays per side
	cannot support one, and a number that looks statistical would outrank the honest reading of what is
	just a difference between two sessions of the same frozen inputs.
	"""

	run_count_a: int = Field(ge=0)
	run_count_b: int = Field(ge=0)
	max_attempts_a: int = Field(ge=1)
	max_attempts_b: int = Field(ge=1)
	completion_rate_a: float = Field(ge=0.0, le=1.0)
	completion_rate_b: float = Field(ge=0.0, le=1.0)
	completion_rate_delta: float = Field(description='B minus A')
	mean_elapsed_a: float = Field(ge=0.0)
	mean_elapsed_b: float = Field(ge=0.0)
	mean_elapsed_delta: float = Field(description='B minus A, in post-processing seconds')
	failed_runs_a: int = Field(ge=0)
	failed_runs_b: int = Field(ge=0)
	recovered_calls_b: int = Field(ge=0, description='Calls on side B that succeeded only after a retry')
	exception_type_counts_a: dict[str, int] = Field(default_factory=dict)
	exception_type_counts_b: dict[str, int] = Field(default_factory=dict)


def _comparable(summary: ReplayBenchmarkSummary, label: str) -> tuple[int, int, float, float, int]:
	"""Pull the five numbers a comparison needs out of one summary, or refuse to compare it.

	An empty summary would produce rates that do not exist, and a summary whose replays disagreed about
	``max_attempts`` describes no single configuration, so both fail here instead of publishing a delta
	between two things that were never measured.
	"""
	if summary.run_count == 0:
		raise ReplayBenchmarkError(f'Cannot compare replay summaries: the {label} side has no replays')
	if summary.max_attempts is None:
		raise ReplayBenchmarkError(f'Cannot compare replay summaries: the {label} side does not share one retry budget')
	if summary.pipeline_completion_rate is None or summary.mean_elapsed_seconds is None:
		raise ReplayBenchmarkError(f'Cannot compare replay summaries: the {label} side has no rate or timing')

	return (
		summary.run_count,
		summary.max_attempts,
		summary.pipeline_completion_rate,
		summary.mean_elapsed_seconds,
		summary.failed_run_count,
	)


def compare_replay_results(baseline: ReplayBenchmarkSummary, candidate: ReplayBenchmarkSummary) -> ReplayComparison:
	"""Diff two replay summaries, where ``candidate`` is the side that retried.

	Both sides have to come from the same fixtures and repeat count for the difference to mean anything;
	this function checks neither, because it cannot see the datasets, only that each side is a real
	measurement of one configuration.
	"""
	run_count_a, max_attempts_a, completion_a, elapsed_a, failed_a = _comparable(baseline, 'baseline')
	run_count_b, max_attempts_b, completion_b, elapsed_b, failed_b = _comparable(candidate, 'candidate')

	return ReplayComparison(
		run_count_a=run_count_a,
		run_count_b=run_count_b,
		max_attempts_a=max_attempts_a,
		max_attempts_b=max_attempts_b,
		completion_rate_a=completion_a,
		completion_rate_b=completion_b,
		completion_rate_delta=completion_b - completion_a,
		mean_elapsed_a=elapsed_a,
		mean_elapsed_b=elapsed_b,
		mean_elapsed_delta=elapsed_b - elapsed_a,
		failed_runs_a=failed_a,
		failed_runs_b=failed_b,
		recovered_calls_b=candidate.total_recovered_calls,
		exception_type_counts_a=dict(baseline.exception_type_counts),
		exception_type_counts_b=dict(candidate.exception_type_counts),
	)


__all__ = [
	'EvidenceReplayBenchmarkResult',
	'ReplayBenchmarkError',
	'ReplayBenchmarkSummary',
	'ReplayComparison',
	'ReplayFixture',
	'ReplayRunResult',
	'claim_signature',
	'compare_replay_results',
	'load_replay_fixtures',
	'normalize_claim_text',
	'run_replay',
	'select_fixtures',
	'status_signature',
	'summarize_replay_runs',
]
