"""Run the offline WebEvidence benchmark over a frozen, gold-labelled dataset.

Three modes, one harness:

    lexical    Phase 4A only. No model, no key, no network, no quota.
    semantic   Phase 4A + Phase 4B, which needs a live model.
    full       Phase 4A + 4B + 5, which also needs a live model.

The two live modes refuse to run without ``--live-llm``, because a benchmark number produced by an
accidental API call is not a number anyone can reproduce later. Live scores are also not constants: the
same case can come back ``SUPPORTED`` on one run and ``PARTIAL`` on the next even at ``temperature=0``,
so repeat a live run before drawing a conclusion from it.

Examples::

    uv run python scripts/run_webevidence_benchmark.py --mode lexical
    uv run python scripts/run_webevidence_benchmark.py --mode full --live-llm
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from browser_use.evidence import EvidenceAligner  # noqa: E402
from browser_use.evidence.benchmark import (  # noqa: E402
	EvidenceBenchmarkError,
	EvidenceBenchmarkExecutionError,
	EvidenceBenchmarkRunner,
	load_benchmark_cases,
)

DEFAULT_DATASET = ROOT / 'benchmarks' / 'webevidence' / 'seed_cases.jsonl'
DEFAULT_OUTPUT = ROOT / 'tmp' / 'webevidence-benchmark' / 'result.json'

MODEL = 'qwen3.8-flash'
API_KEY_ENV = 'ALIBABA_CLOUD'
BASE_URL_ENV = 'ALIBABA_CLOUD_BASE_URL'
DEFAULT_BASE_URL = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument(
		'--dataset',
		type=Path,
		default=DEFAULT_DATASET,
		help=f'JSONL benchmark dataset (default: {DEFAULT_DATASET.relative_to(ROOT)})',
	)
	parser.add_argument('--mode', choices=('lexical', 'semantic', 'full'), default='lexical', help='which stages to measure')
	parser.add_argument(
		'--output', type=Path, default=DEFAULT_OUTPUT, help='where to write the full EvidenceBenchmarkResult JSON'
	)
	parser.add_argument('--top-k', type=int, default=5, help='candidate set size for the lexical aligner')
	parser.add_argument(
		'--live-llm', action='store_true', help='allow real qwen3.8-flash calls, required for semantic and full modes'
	)
	return parser.parse_args(argv)


def build_runner(mode: str, top_k: int, live: bool):
	"""Assemble the runner for one mode, refusing to spend quota without being told to."""
	if mode != 'lexical' and not live:
		raise EvidenceBenchmarkError(f'{mode} requires --live-llm: it would make real {MODEL} calls')

	runner = EvidenceBenchmarkRunner(aligner=EvidenceAligner(top_k=top_k))
	if mode == 'lexical':
		return runner

	from dotenv import load_dotenv

	load_dotenv(ROOT / '.env')
	api_key = os.getenv(API_KEY_ENV)
	if not api_key:
		raise EvidenceBenchmarkError(f'{API_KEY_ENV} is not set, so {mode} mode cannot run')

	from browser_use.evidence import ClaimVerifier, SemanticEvidenceReranker
	from browser_use.llm.openai.chat import ChatOpenAI

	llm = ChatOpenAI(
		model=MODEL,
		api_key=api_key,
		base_url=os.getenv(BASE_URL_ENV) or DEFAULT_BASE_URL,
		temperature=0.0,
	)
	reranker = SemanticEvidenceReranker(llm)
	if mode == 'semantic':
		return EvidenceBenchmarkRunner(aligner=EvidenceAligner(top_k=top_k), reranker=reranker)
	return EvidenceBenchmarkRunner(aligner=EvidenceAligner(top_k=top_k), reranker=reranker, verifier=ClaimVerifier(llm))


def print_summary(result, *, mode: str, dataset: Path, output: Path) -> None:
	"""Print the headline numbers and the failure lists. Never prints evidence or claim content."""
	summary = result.summary
	print(f'dataset: {dataset}')
	print(f'mode: {mode}')
	print(f'cases: {summary.case_count} (retrieval scored: {summary.retrieval_case_count})')
	_print('Lexical Hit@1', summary.lexical_hit_at_1_rate)
	_print('Lexical Hit@K', summary.lexical_hit_at_k_rate)
	_print('Lexical MRR', summary.lexical_mrr)
	if summary.semantic_hit_at_1_rate is not None:
		_print('Semantic Hit@1', summary.semantic_hit_at_1_rate)
		_print('Semantic Hit@K', summary.semantic_hit_at_k_rate)
		_print('Semantic MRR', summary.semantic_mrr)
	if summary.relation_accuracy is not None:
		_print('Relation Accuracy', summary.relation_accuracy)
		_print('Relation Macro-F1', summary.relation_macro_f1)
		_print('Status Accuracy', summary.status_accuracy)
	_print_list('Lexical misses', summary.lexical_miss_case_ids)
	_print_list('Status errors', summary.status_error_case_ids)
	_print_list('Relation errors', summary.relation_error_case_ids)
	print(f'result: {output}')


def _print(label: str, value: float | None) -> None:
	print(f'{label}: {"unavailable" if value is None else f"{value:.4f}"}')


def _print_list(label: str, case_ids: list[str]) -> None:
	print(f'{label}: {", ".join(case_ids) if case_ids else "none"}')


async def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)

	try:
		cases = load_benchmark_cases(args.dataset)
		runner = build_runner(args.mode, args.top_k, args.live_llm)
		result = await runner.run(cases)
	except EvidenceBenchmarkExecutionError as e:
		print(f'benchmark aborted: {e} (case_id={e.case_id}, stage={e.stage.value})', file=sys.stderr)
		return 1
	except EvidenceBenchmarkError as e:
		print(f'benchmark refused: {e}', file=sys.stderr)
		return 1

	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(result.model_dump_json(indent=2), encoding='utf-8')
	print_summary(result, mode=runner.mode, dataset=args.dataset, output=args.output)
	return 0


if __name__ == '__main__':
	raise SystemExit(asyncio.run(main()))
