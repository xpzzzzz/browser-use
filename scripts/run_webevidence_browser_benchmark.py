"""Run the paired end-to-end browser benchmark for the WebEvidence pipeline.

Each case is one real browser run, and that single run produces both measurements being compared::

	agent run -> final answer -> raw task check (the baseline)
	                         -> evidence -> pipeline -> claim-level verification (WebEvidence)

The baseline is not a second agent run. Two trajectories would differ in steps, page state and model
randomness, and the interesting difference between "the task passed" and "these claims are unsupported"
would be impossible to attribute. Sharing one run leaves the evidence layer as the only variable, and
since the pipeline never rewrites the answer, what this measures is failure visibility, not accuracy gain.

Setup: ``ALIBABA_CLOUD`` in the environment or in the repo ``.env``, plus ``ALIBABA_CLOUD_BASE_URL`` for
a region-scoped key (a mainland account needs ``https://dashscope.aliyuncs.com/compatible-mode/v1``).

The agent runs on the plain ``ChatOpenAI`` client and only the post-processing stages run through
:class:`~browser_use.evidence.retrying_llm.RetryingChatModel`. Phase 9B showed browsing was already
reliable and post-processing was not, so keeping the browser half un-retried is what makes the retry
numbers attributable to the stage that changed.

Examples::

	python scripts/run_webevidence_browser_benchmark.py --case example-heading
	python scripts/run_webevidence_browser_benchmark.py --limit 2
	python scripts/run_webevidence_browser_benchmark.py --repeats 2 --headful
	python scripts/run_webevidence_browser_benchmark.py --repeats 2 --postprocess-max-attempts 1

Runs are sequential, and each gets its own browser profile and evidence store under ``tmp/``. This script
is the live half of Phase 9B: nothing in the test suite starts a browser or calls a model.
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError
from uuid_extensions import uuid7str

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')

from browser_use import Agent, Browser, ChatOpenAI
from browser_use.evidence import (
	ClaimExtractor,
	ClaimVerifier,
	EvidenceAligner,
	EvidenceCollector,
	EvidenceGroundedReport,
	EvidenceOrganizer,
	EvidenceReportBuilder,
	JsonlEvidenceStore,
	LLMRetryPolicy,
	MarkdownReportRenderer,
	RetryingChatModel,
	SemanticEvidenceReranker,
	WebEvidencePipeline,
	stats_delta,
)
from browser_use.evidence.e2e_benchmark import (
	AnswerCheck,
	BrowserBenchmarkCase,
	BrowserBenchmarkError,
	BrowserBenchmarkFailureStage,
	BrowserBenchmarkRunResult,
	EvidenceBrowserBenchmarkResult,
	evaluate_answer,
	load_browser_benchmark_cases,
	run_result_with_pipeline,
	run_result_with_retry_stats,
	summarize_browser_runs,
)
from browser_use.llm.base import BaseChatModel

MODEL = 'qwen3.8-flash'
API_KEY_ENV = 'ALIBABA_CLOUD'
BASE_URL_ENV = 'ALIBABA_CLOUD_BASE_URL'
# Same fallback as the Phase 8 demo and the Phase 9A CLI: keys are region scoped, so a mainland account
# points BASE_URL_ENV at the Beijing endpoint instead.
DEFAULT_BASE_URL = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'

DEFAULT_DATASET = ROOT / 'benchmarks' / 'webevidence' / 'browser_cases.jsonl'
DEFAULT_OUTPUT_ROOT = ROOT / 'tmp' / 'webevidence-browser-benchmark'


def positive_int(raw: str) -> int:
	"""A --repeats of zero would be a benchmark that measures nothing."""
	value = int(raw)
	if value < 1:
		raise argparse.ArgumentTypeError('must be at least 1')
	return value


def non_negative_float(raw: str) -> float:
	"""A negative delay is not a wait at all, and the policy rejects it with a longer message."""
	value = float(raw)
	if value < 0:
		raise argparse.ArgumentTypeError('must be at least 0')
	return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET, help='JSONL browser task dataset')
	parser.add_argument(
		'--output-dir', type=Path, default=DEFAULT_OUTPUT_ROOT, help='root for per-run artifacts and benchmark_result.json'
	)
	parser.add_argument('--headful', action='store_true', help='show the browser window instead of running headless')
	parser.add_argument(
		'--extensions',
		action=argparse.BooleanOptionalAction,
		default=False,
		help='load the ad-block and cookie-banner extensions; off by default because fetching them can stall the browser launch watchdog',
	)
	parser.add_argument('--case', action='append', help='run only this case id, repeatable')
	parser.add_argument('--limit', type=positive_int, help='run only the first N cases')
	parser.add_argument('--repeats', type=positive_int, default=1, help='independent runs per case, to observe variance')
	parser.add_argument(
		'--postprocess-max-attempts',
		type=positive_int,
		default=3,
		help='total post-processing model attempts per call, so 3 means two retries; 1 turns retry off',
	)
	parser.add_argument(
		'--postprocess-retry-delay',
		type=non_negative_float,
		default=1.0,
		help='seconds before the first post-processing retry, doubling up to an 8s ceiling',
	)
	return parser.parse_args(argv)


def select_cases(cases: list[BrowserBenchmarkCase], only: list[str] | None, limit: int | None) -> list[BrowserBenchmarkCase]:
	"""Narrow the dataset for a smoke run, and say so when a name was typed wrong."""
	if only:
		wanted = list(dict.fromkeys(only))
		known = {case.case_id for case in cases}
		unknown = [case_id for case_id in wanted if case_id not in known]
		if unknown:
			raise BrowserBenchmarkError(f'Unknown --case value(s): {", ".join(unknown)}')
		cases = [case for case in cases if case.case_id in wanted]
	if limit is not None:
		cases = cases[:limit]
	if not cases:
		raise BrowserBenchmarkError('The selected dataset is empty')
	return cases


def build_pipeline(llm: BaseChatModel) -> WebEvidencePipeline:
	"""Wire Phases 3 to 7 once and reuse the same client the agent used.

	The stages take a ``BaseChatModel``, so whether that model retries is decided by what the caller passes
	in here, not by anything the stages know about.
	"""
	return WebEvidencePipeline(
		claim_extractor=ClaimExtractor(llm),
		aligner=EvidenceAligner(top_k=5),
		reranker=SemanticEvidenceReranker(llm),
		verifier=ClaimVerifier(llm),
		organizer=EvidenceOrganizer(),
		report_builder=EvidenceReportBuilder(),
		markdown_renderer=MarkdownReportRenderer(),
	)


def _record_failure(
	run: BrowserBenchmarkRunResult,
	stage: BrowserBenchmarkFailureStage,
	cause: Exception | str,
) -> BrowserBenchmarkRunResult:
	"""Record where a run stopped, naming only the exception type.

	A provider or browser exception can carry the prompt, scraped page content or a credential in its
	message, and a benchmark result file gets shared. The original exception stays available in the
	process for whoever is debugging, which is why only its type is written down.

	A pipeline error already names the stage that died, so that stage is appended. It is a label from our
	own enum rather than exception text, and without it every post-processing failure reads as the same
	anonymous ``WebEvidencePipelineError``.
	"""
	if isinstance(cause, str):
		failure_type = cause
	else:
		failure_type = type(cause).__name__
		inner_stage = getattr(cause, 'stage', None)
		if getattr(inner_stage, 'value', None):
			failure_type = f'{failure_type}:{inner_stage.value}'
	return run.model_copy(update={'failure_stage': stage, 'failure_type': failure_type})


def _write_artifacts(run_dir: Path, report: EvidenceGroundedReport, markdown: str) -> tuple[str, str]:
	"""Persist one run's report, UTF-8 and through Pydantic, so it can be re-read without a parser."""
	report_json = run_dir / 'report.json'
	report_markdown = run_dir / 'report.md'
	report_json.write_text(report.model_dump_json(indent=2), encoding='utf-8')
	report_markdown.write_text(markdown, encoding='utf-8')
	return _display_path(report_json), _display_path(report_markdown)


def _display_path(path: Path) -> str:
	"""Repo-relative when it can be, absolute when ``--output-dir`` points outside the repository."""
	resolved = path.resolve()
	try:
		return str(resolved.relative_to(ROOT))
	except ValueError:
		return str(resolved)


async def run_case(
	case: BrowserBenchmarkCase,
	*,
	repeat: int,
	llm: ChatOpenAI,
	postprocess_llm: RetryingChatModel,
	output_root: Path,
	headless: bool,
	extensions: bool,
) -> BrowserBenchmarkRunResult:
	"""Drive one paired run. This never raises: a stopped run is still a measured run.

	The result carries the raw task check and the claim-level observation from the same trajectory, and a
	failure leaves the later fields ``None`` rather than inventing a value for a stage that never ran.
	Retry telemetry is a snapshot difference across this run only: one wrapper serves the whole benchmark,
	so its running totals would otherwise be copied onto every record.
	"""
	run_dir = output_root / case.case_id / f'run-{repeat:03d}'
	run_dir.mkdir(parents=True, exist_ok=True)

	# The collector's task id is the pipeline's task id, so evidence and claims stay in one namespace.
	task_id = uuid7str()
	store = JsonlEvidenceStore(run_dir / 'evidence.jsonl')
	collector = EvidenceCollector(task_id=task_id, store=store, screenshot_dir=run_dir / 'screenshots')

	run = BrowserBenchmarkRunResult(
		case_id=case.case_id,
		task=case.task,
		repeat_index=repeat,
		agent_completed=False,
		final_answer_present=False,
		evidence_count=0,
		pipeline_completed=False,
	)

	browser: Browser | None = None
	try:
		browser = Browser(
			headless=headless,
			# A profile per run, so no two runs and no already-open browser fight over one profile lock.
			user_data_dir=str(run_dir / 'browser-profile'),
			enable_default_extensions=extensions,
		)
	except Exception as e:
		return _record_failure(run, BrowserBenchmarkFailureStage.BROWSER_START, e)

	print(f'  {case.case_id} run-{repeat:03d}: starting agent (max {case.max_steps} steps)', flush=True)
	try:
		agent = Agent(
			task=case.task,
			llm=llm,
			browser=browser,
			register_new_step_callback=collector.collect_step,
			# A text-only model gets a text-only observation; screenshots are still captured and stored.
			use_vision=False,
		)
		agent_started = time.perf_counter()
		try:
			history = await agent.run(max_steps=case.max_steps)
		finally:
			agent_elapsed = time.perf_counter() - agent_started
			try:
				await browser.stop()
			except Exception as e:
				print(f'  warning: browser stop failed ({type(e).__name__})', file=sys.stderr, flush=True)
	except Exception as e:
		nodes = store.load_all()
		return _record_failure(
			run.model_copy(update={'evidence_count': len(nodes), 'agent_elapsed_seconds': round(agent_elapsed, 3)}),
			BrowserBenchmarkFailureStage.AGENT_RUN,
			e,
		)

	nodes = store.load_all()
	answer = history.final_result()
	present = bool(answer and answer.strip())
	check: AnswerCheck | None = evaluate_answer(case, answer) if present else None

	run = run.model_copy(
		update={
			'agent_completed': history.is_done(),
			'final_answer_present': present,
			'answer_check_passed': check.answer_check_passed if check else None,
			'matched_patterns': check.matched_patterns if check else [],
			'missing_patterns': check.missing_patterns if check else [],
			'browser_step_count': history.number_of_steps(),
			'evidence_count': len(nodes),
			'agent_elapsed_seconds': round(agent_elapsed, 3),
		}
	)
	if not present:
		# No answer means there was nothing to check, so the task verdict stays None rather than False.
		return _record_failure(run, BrowserBenchmarkFailureStage.FINAL_ANSWER, 'MissingFinalAnswer')

	print(
		f'  {case.case_id} run-{repeat:03d}: steps={run.browser_step_count} evidence={len(nodes)} answer_check={run.answer_check_passed}',
		flush=True,
	)

	stats_before_pipeline = postprocess_llm.snapshot_stats()
	try:
		pipeline_started = time.perf_counter()
		result = await build_pipeline(postprocess_llm).analyze(
			task_id=task_id, task=case.task, answer=answer or '', evidence_nodes=nodes
		)
		pipeline_elapsed = time.perf_counter() - pipeline_started
	except Exception as e:
		# Attempts spent before a failure are part of the run's cost, so record them, then fail exactly as
		# Phase 9B did: the stage error still ends the run and still names its stage.
		run = run_result_with_retry_stats(run, stats_delta(stats_before_pipeline, postprocess_llm.snapshot_stats()))
		return _record_failure(run, BrowserBenchmarkFailureStage.PIPELINE, e)

	run = run_result_with_retry_stats(
		run_result_with_pipeline(run, result).model_copy(update={'pipeline_elapsed_seconds': round(pipeline_elapsed, 3)}),
		stats_delta(stats_before_pipeline, postprocess_llm.snapshot_stats()),
	)

	try:
		report_json, report_markdown = _write_artifacts(run_dir, result.report, result.markdown)
	except OSError as e:
		return _record_failure(run, BrowserBenchmarkFailureStage.OUTPUT_WRITE, e)

	summary = result.report.summary
	run = run.model_copy(
		update={
			'report_json_path': report_json,
			'report_markdown_path': report_markdown,
		}
	)
	(run_dir / 'run_result.json').write_text(run.model_dump_json(indent=2), encoding='utf-8')

	print(
		f'  {case.case_id} run-{repeat:03d}: claims={summary.claim_count} supported={summary.supported_claim_count} '
		f'no_evidence={summary.no_evidence_claim_count} fully_supported={run.fully_supported} '
		f'retries={run.postprocess_llm_retry_count} recovered={run.postprocess_llm_recovered_calls}',
		flush=True,
	)
	return run


def print_summary(result: EvidenceBrowserBenchmarkResult, *, dataset: Path, output: Path) -> None:
	"""Print the headline numbers. Never prints an answer, page text, or a key."""
	summary = result.summary
	print()
	print(f'dataset: {dataset}')
	print(f'runs: {summary.run_count} over {summary.case_count} cases')
	_print('Agent completion', summary.agent_completion_rate)
	_print('Final answer rate', summary.final_answer_rate)
	_print('Answer check pass', summary.answer_check_pass_rate)
	_print('Pipeline completion', summary.pipeline_completion_rate)
	print('Post-processing LLM retry:')
	print(f'  logical calls: {_count(summary.total_postprocess_llm_logical_calls)}')
	print(f'  provider attempts: {_count(summary.total_postprocess_llm_attempts)}')
	print(f'  retries: {_count(summary.total_postprocess_llm_retries)}')
	print(f'  recovered calls: {_count(summary.total_postprocess_llm_recovered_calls)}')
	print(f'  failed calls: {_count(summary.total_postprocess_llm_failed_calls)}')
	print(f'  runs with retry: {summary.runs_with_postprocess_retry_count}')
	print(f'  runs recovered by retry: {summary.runs_recovered_by_retry_count}')
	print(f'Mean browser steps: {_num(summary.mean_browser_steps)}')
	print(f'Mean evidence count: {_num(summary.mean_evidence_count)}')
	print(f'Mean claim count: {_num(summary.mean_claim_count)}')
	print(f'Mean evidence coverage: {_pct(summary.mean_evidence_coverage_rate)}')
	print(f'Pooled claims verified: {summary.total_claims_verified}')
	for label, value in (
		('SUPPORTED', summary.supported_claim_rate),
		('PARTIAL', summary.partial_claim_rate),
		('UNSUPPORTED', summary.unsupported_claim_rate),
		('CONTRADICTED', summary.contradicted_claim_rate),
		('CONFLICTED', summary.conflicted_claim_rate),
		('NO_EVIDENCE', summary.no_evidence_claim_rate),
	):
		_print(f'Claim rate {label}', value)
	_print('Fully supported runs', summary.fully_supported_run_rate)
	_print_list('Answer passed but not fully supported', summary.answer_pass_but_not_fully_supported_case_ids)
	_print_list('Answer check failed', summary.answer_fail_case_ids)
	_print_list('Pipeline failed', summary.pipeline_fail_case_ids)
	for stage, case_ids in summary.failure_case_ids_by_stage.items():
		print(f'  failures at {stage}: {", ".join(case_ids)}')
	print(f'Mean agent seconds: {_num(summary.mean_agent_elapsed_seconds)}')
	print(f'Mean pipeline seconds: {_num(summary.mean_pipeline_elapsed_seconds)}')
	print(f'result: {output}')


def _print(label: str, value: float | None) -> None:
	print(f'{label}: {_pct(value)}')


def _print_list(label: str, case_ids: list[str]) -> None:
	print(f'{label}: {", ".join(case_ids) if case_ids else "none"}')


def _pct(value: float | None) -> str:
	return 'unavailable' if value is None else f'{value:.4f}'


def _num(value: float | None) -> str:
	return 'unavailable' if value is None else f'{value:.2f}'


def _count(value: int | None) -> str:
	"""No telemetry is not the same claim as a total of zero."""
	return 'unavailable' if value is None else str(value)


async def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)
	try:
		retry_policy = LLMRetryPolicy(
			max_attempts=args.postprocess_max_attempts, initial_delay_seconds=args.postprocess_retry_delay
		)
	except ValidationError:
		# Before the dataset read and the key check, so this message is never masked by an unrelated one.
		print('--postprocess-retry-delay must not exceed the 8s retry ceiling', file=sys.stderr)
		return 1
	# A relative --output-dir means "relative to the repository", so recorded paths stay meaningful.
	if not args.output_dir.is_absolute():
		args.output_dir = ROOT / args.output_dir

	try:
		cases = select_cases(load_browser_benchmark_cases(args.dataset), args.case, args.limit)
	except BrowserBenchmarkError as e:
		print(f'benchmark refused: {e}', file=sys.stderr)
		return 1

	api_key = os.getenv(API_KEY_ENV)
	if not api_key:
		print(f'{API_KEY_ENV} is not set, so the browser benchmark cannot run', file=sys.stderr)
		return 1

	base_url = os.getenv(BASE_URL_ENV) or DEFAULT_BASE_URL
	# One client shared by the agent and every pipeline stage, sequentially, as the Phase 8 demo proved.
	llm = ChatOpenAI(model=MODEL, api_key=api_key, base_url=base_url, temperature=0.0)
	# The wrapper adds attempts to a call; it never changes which model answers. Only post-processing is
	# wrapped so the browsing baseline stays what Phase 9B measured, and one wrapper instance serves every
	# run, which is why each run records a snapshot difference instead of its totals.
	postprocess_llm = RetryingChatModel(llm, policy=retry_policy)
	print(f'model: {MODEL} via {base_url}')
	delays = ', '.join(f'{delay}s' for delay in retry_policy.retry_delays()) or 'retry disabled'
	print(f'post-processing retry: {retry_policy.max_attempts} attempts per call, delays: {delays}')
	print(f'cases: {", ".join(case.case_id for case in cases)}')
	print(f'repeats per case: {args.repeats}')

	args.output_dir.mkdir(parents=True, exist_ok=True)
	runs: list[BrowserBenchmarkRunResult] = []
	for repeat in range(1, args.repeats + 1):
		for case in cases:
			run = await run_case(
				case,
				repeat=repeat,
				llm=llm,
				postprocess_llm=postprocess_llm,
				output_root=args.output_dir,
				headless=not args.headful,
				extensions=args.extensions,
			)
			runs.append(run)

	result = EvidenceBrowserBenchmarkResult(summary=summarize_browser_runs(runs), runs=runs)
	output = args.output_dir / 'benchmark_result.json'
	output.write_text(result.model_dump_json(indent=2), encoding='utf-8')
	print_summary(result, dataset=args.dataset, output=output)
	return 0


if __name__ == '__main__':
	raise SystemExit(asyncio.run(main()))
