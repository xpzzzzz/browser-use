"""Run the fixed-input replay benchmark: frozen answers and evidence, pushed through the pipeline again.

This is the controlled half of the retry question. The Phase 9B and 9C browser benchmarks both reported
pipeline completion of 9/14, and that is not a comparison: a browser run varies in step count, page state
and captured evidence between sessions, and provider latency moved on its own. A fixture pins one task, one
final answer and one evidence list, so the only things that can still change between two replays are the
provider and the model. Nothing here opens a browser or runs an agent.

Each invocation measures one retry configuration and writes one artifact::

	python scripts/run_webevidence_replay_benchmark.py --repeats 1 --max-attempts 1 --fixture example-heading-single-claim
	python scripts/run_webevidence_replay_benchmark.py --max-attempts 1 --output tmp/p9d-attempt1.json
	python scripts/run_webevidence_replay_benchmark.py --max-attempts 3 --output tmp/p9d-attempt3.json
	python scripts/run_webevidence_replay_benchmark.py --compare tmp/p9d-attempt1.json tmp/p9d-attempt3.json

The two --max-attempts runs are the comparison, and they are separate invocations on purpose: a retry
budget cannot be A/B-ed inside one process, because a call either retried or it did not. Diffing them is
descriptive, not statistical; fifteen replays per side support a difference, never a significance test.

Setup: ``ALIBABA_CLOUD`` in the environment or in the repo ``.env``, plus ``ALIBABA_CLOUD_BASE_URL`` for a
region-scoped key (a mainland account needs ``https://dashscope.aliyuncs.com/compatible-mode/v1``).

Replays run sequentially. A replay whose pipeline fails is a measured replay, not an aborted benchmark, and
its counters stay in the artifact; a model call that never came back is recorded as the stage that stopped
it, never as an unsupported claim.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')

from browser_use import ChatOpenAI
from browser_use.evidence import (
	ClaimExtractor,
	ClaimVerifier,
	EvidenceAligner,
	EvidenceOrganizer,
	EvidenceReplayBenchmarkResult,
	EvidenceReportBuilder,
	LLMRetryPolicy,
	MarkdownReportRenderer,
	ReplayBenchmarkError,
	ReplayComparison,
	ReplayRunResult,
	RetryingChatModel,
	SemanticEvidenceReranker,
	WebEvidencePipeline,
	compare_replay_results,
	load_replay_fixtures,
	run_replay,
	select_fixtures,
	summarize_replay_runs,
)
from browser_use.llm.base import BaseChatModel

MODEL = 'qwen3.8-flash'
API_KEY_ENV = 'ALIBABA_CLOUD'
BASE_URL_ENV = 'ALIBABA_CLOUD_BASE_URL'
# Same fallback as the Phase 8 demo, the Phase 9A CLI and the Phase 9B browser benchmark: keys are region
# scoped, so a mainland account points BASE_URL_ENV at the Beijing endpoint instead.
DEFAULT_BASE_URL = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'

DEFAULT_DATASET = ROOT / 'benchmarks' / 'webevidence' / 'replay' / 'manifest.jsonl'
DEFAULT_OUTPUT_DIR = ROOT / 'tmp' / 'webevidence-replay-benchmark'


def positive_int(raw: str) -> int:
	"""A --repeats of zero or a --max-attempts of zero would be a benchmark that measures nothing."""
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
	parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET, help='JSONL replay fixture dataset')
	parser.add_argument(
		'--output',
		type=Path,
		help='result artifact path; defaults to tmp/webevidence-replay-benchmark/replay-attempts-<max-attempts>.json',
	)
	parser.add_argument('--repeats', type=positive_int, default=3, help='independent replays per fixture, to observe variance')
	parser.add_argument(
		'--max-attempts',
		type=positive_int,
		default=3,
		dest='max_attempts',
		help='total post-processing model attempts per call, so 3 means two retries; 1 turns retry off',
	)
	parser.add_argument(
		'--retry-delay',
		type=non_negative_float,
		default=1.0,
		dest='retry_delay',
		help='seconds before the first retry, doubling up to an 8s ceiling',
	)
	parser.add_argument('--fixture', action='append', help='replay only this fixture id, repeatable')
	parser.add_argument(
		'--compare',
		nargs=2,
		metavar=('BASELINE', 'CANDIDATE'),
		help='print the descriptive difference between two existing artifacts and exit',
	)
	return parser.parse_args(argv)


def build_pipeline(llm: BaseChatModel) -> WebEvidencePipeline:
	"""Wire Phases 3 to 7 exactly as the live browser benchmark wires them.

	No stage here knows that retry exists: the attempt budget is a property of the model wrapper the caller
	passed in, which is the whole reason a retry A/B can be run without touching one line of the pipeline.
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


def _display_path(path: Path) -> str:
	"""Repo-relative when it can be, absolute when --output points outside the repository."""
	resolved = path.resolve()
	try:
		return str(resolved.relative_to(ROOT))
	except ValueError:
		return str(resolved)


def _describe(run: ReplayRunResult) -> str:
	"""One line per replay. Never prints an answer, page text, or a key."""
	if not run.pipeline_completed:
		stage = run.failure_stage.value if run.failure_stage else 'UNKNOWN'
		return f'FAILED at {stage} after {run.postprocess_llm_logical_calls} calls, {run.elapsed_seconds:.1f}s'
	return (
		f'claims={run.claim_count} supported={run.supported_claim_count} no_evidence={run.no_evidence_claim_count} '
		f'retries={run.postprocess_llm_retry_count} recovered={run.postprocess_llm_recovered_calls} '
		f'failed_calls={run.postprocess_llm_failed_calls} after {run.postprocess_llm_logical_calls} calls, {run.elapsed_seconds:.1f}s'
	)


def print_summary(result: EvidenceReplayBenchmarkResult, *, dataset: Path, output: Path, repeats: int) -> None:
	summary = result.summary
	print()
	print(f'dataset: {_display_path(dataset)}')
	print(f'replays: {summary.run_count} over {summary.fixture_count} fixtures, {repeats} repeats each')
	print(f'retry budget per call: {"unavailable" if summary.max_attempts is None else summary.max_attempts} attempts')
	_print('Pipeline completion', summary.pipeline_completion_rate)
	print(f'Failed replays: {summary.failed_run_count}')
	print(f'Mean post-processing seconds: {_num(summary.mean_elapsed_seconds)}')
	print(f'Mean seconds, completed replays only: {_num(summary.mean_completed_elapsed_seconds)}')
	print('Post-processing LLM:')
	print(f'  logical calls: {_count(summary.total_logical_calls)}')
	print(f'  provider attempts: {_count(summary.total_provider_attempts)}')
	print(f'  retries: {_count(summary.total_retries)}')
	print(f'  recovered calls: {_count(summary.total_recovered_calls)}')
	print(f'  failed calls: {_count(summary.total_failed_calls)}')
	print(f'  replays with a retry: {summary.runs_with_retry_count}')
	print(f'  completed replays with a recovered call: {summary.runs_recovered_by_retry_count}')
	print('Exception types:')
	_print_counts(summary.exception_type_counts)
	print('Failure stages:')
	_print_counts(summary.failure_stage_counts)
	print(f'Mean claim count, completed replays: {_num(summary.mean_claim_count)}')
	print('Distinct claim signatures per fixture (0 means no completed replay):')
	_print_counts({key: str(value) for key, value in summary.claim_signature_unique_count_by_fixture.items()})
	print('Distinct status signatures per fixture (0 means no completed replay):')
	_print_counts({key: str(value) for key, value in summary.status_signature_unique_count_by_fixture.items()})
	print(f'result: {_display_path(output)}')


def print_comparison(comparison: ReplayComparison, *, baseline_path: Path, candidate_path: Path) -> None:
	"""Print what changed between two configurations, and nothing that looks like a statistic."""
	print(
		f'baseline:  {_display_path(baseline_path)} ({comparison.run_count_a} replays, {comparison.max_attempts_a} attempts per call)'
	)
	print(
		f'candidate: {_display_path(candidate_path)} ({comparison.run_count_b} replays, {comparison.max_attempts_b} attempts per call)'
	)
	print(
		f'completion rate: {comparison.completion_rate_a:.4f} -> {comparison.completion_rate_b:.4f} ({comparison.completion_rate_delta:+.4f})'
	)
	print(f'failed replays: {comparison.failed_runs_a} -> {comparison.failed_runs_b}')
	print(
		f'mean seconds: {comparison.mean_elapsed_a:.2f} -> {comparison.mean_elapsed_b:.2f} ({comparison.mean_elapsed_delta:+.2f})'
	)
	print('exception types (baseline -> candidate):')
	names = sorted({*comparison.exception_type_counts_a, *comparison.exception_type_counts_b})
	if not names:
		print('  none')
	for name in names:
		print(f'  {name}: {comparison.exception_type_counts_a.get(name, 0)} -> {comparison.exception_type_counts_b.get(name, 0)}')
	print(f'recovered calls on the retrying side: {comparison.recovered_calls_b}')
	print('this is a descriptive difference, not a significance test')


def _print(label: str, value: float | None) -> None:
	print(f'{label}: {_pct(value)}')


def _print_counts(counts: dict[str, int | str]) -> None:
	if not counts:
		print('  none')
	for key, value in counts.items():
		print(f'  {key}: {value}')


def _pct(value: float | None) -> str:
	return 'unavailable' if value is None else f'{value:.4f}'


def _num(value: float | None) -> str:
	return 'unavailable' if value is None else f'{value:.2f}'


def _count(value: int | None) -> str:
	"""No telemetry is not the same claim as a total of zero."""
	return 'unavailable' if value is None else str(value)


def _load_artifact(path: Path) -> EvidenceReplayBenchmarkResult:
	"""Re-read one artifact. A pydantic message is never printed: it echoes the offending value."""
	try:
		return EvidenceReplayBenchmarkResult.model_validate_json(path.read_text(encoding='utf-8'))
	except (OSError, ValidationError) as e:
		raise ReplayBenchmarkError(f'Cannot read replay artifact {path}: {type(e).__name__}') from e


def run_compare(baseline_path: Path, candidate_path: Path) -> int:
	"""Diff two existing artifacts. Reads files only: no key, no model, no network."""
	baseline_result = _load_artifact(baseline_path)
	candidate_result = _load_artifact(candidate_path)

	baseline_fixtures = set(baseline_result.summary.claim_signature_unique_count_by_fixture)
	candidate_fixtures = set(candidate_result.summary.claim_signature_unique_count_by_fixture)
	if baseline_fixtures != candidate_fixtures:
		# A difference between two different fixture sets would be a difference between two experiments.
		only_a = ', '.join(sorted(baseline_fixtures - candidate_fixtures)) or 'none'
		only_b = ', '.join(sorted(candidate_fixtures - baseline_fixtures)) or 'none'
		raise ReplayBenchmarkError(f'Artifacts cover different fixtures, only baseline: {only_a}; only candidate: {only_b}')
	if baseline_result.summary.run_count != candidate_result.summary.run_count:
		raise ReplayBenchmarkError(
			f'Artifacts have different replay counts: {baseline_result.summary.run_count} vs {candidate_result.summary.run_count}'
		)
	if baseline_result.summary.max_attempts == candidate_result.summary.max_attempts:
		# Not wrong, just not the comparison this benchmark exists to make, so say it out loud.
		print(
			f'warning: both artifacts used {baseline_result.summary.max_attempts} attempts per call, so this is a session-to-session diff',
			file=sys.stderr,
		)

	comparison = compare_replay_results(baseline_result.summary, candidate_result.summary)
	print_comparison(comparison, baseline_path=baseline_path, candidate_path=candidate_path)
	return 0


async def run_benchmark(args: argparse.Namespace, policy: LLMRetryPolicy) -> int:
	try:
		fixtures = select_fixtures(load_replay_fixtures(args.dataset), args.fixture)
	except ReplayBenchmarkError as e:
		print(f'benchmark refused: {e}', file=sys.stderr)
		return 1

	api_key = os.getenv(API_KEY_ENV)
	if not api_key:
		print(f'{API_KEY_ENV} is not set, so the replay benchmark cannot run', file=sys.stderr)
		return 1

	base_url = os.getenv(BASE_URL_ENV) or DEFAULT_BASE_URL
	llm = ChatOpenAI(model=MODEL, api_key=api_key, base_url=base_url, temperature=0.0)
	# The wrapper adds attempts to a call; it never changes which model answers. One instance serves every
	# replay, which is why each record carries a snapshot difference rather than its totals.
	wrapper = RetryingChatModel(llm, policy=policy)
	pipeline = build_pipeline(wrapper)

	print(f'model: {MODEL} via {base_url}')
	delays = ', '.join(f'{delay}s' for delay in policy.retry_delays()) or 'retry disabled'
	print(f'post-processing retry: {policy.max_attempts} attempts per call, delays: {delays}')
	print(f'fixtures: {", ".join(fixture.fixture_id for fixture in fixtures)}')
	print(f'repeats per fixture: {args.repeats}')

	output = args.output
	output.parent.mkdir(parents=True, exist_ok=True)

	runs: list[ReplayRunResult] = []
	for repeat in range(1, args.repeats + 1):
		for fixture in fixtures:
			run = await run_replay(
				fixture,
				pipeline=pipeline,
				repeat_index=repeat,
				max_attempts=policy.max_attempts,
				stats=wrapper.snapshot_stats,
			)
			runs.append(run)
			print(f'  {fixture.fixture_id} repeat-{repeat:03d}: {_describe(run)}', flush=True)
			# Written as it goes, because a run long enough to be interrupted is worth having persisted.
			output.write_text(
				EvidenceReplayBenchmarkResult(summary=summarize_replay_runs(runs), runs=runs).model_dump_json(indent=2),
				encoding='utf-8',
			)

	print_summary(
		EvidenceReplayBenchmarkResult(summary=summarize_replay_runs(runs), runs=runs),
		dataset=args.dataset,
		output=output,
		repeats=args.repeats,
	)
	return 0


async def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)

	if args.compare:
		try:
			return run_compare(Path(args.compare[0]), Path(args.compare[1]))
		except ReplayBenchmarkError as e:
			print(f'comparison refused: {e}', file=sys.stderr)
			return 1

	try:
		policy = LLMRetryPolicy(max_attempts=args.max_attempts, initial_delay_seconds=args.retry_delay)
	except ValidationError:
		# Before the dataset read and the key check, so this message is never masked by an unrelated one.
		print('--retry-delay must not exceed the 8s retry ceiling', file=sys.stderr)
		return 1

	# A relative --output means "relative to the repository", so recorded paths stay meaningful.
	if args.output is None:
		args.output = DEFAULT_OUTPUT_DIR / f'replay-attempts-{args.max_attempts}.json'
	elif not args.output.is_absolute():
		args.output = ROOT / args.output

	return await run_benchmark(args, policy)


if __name__ == '__main__':
	raise SystemExit(asyncio.run(main()))
