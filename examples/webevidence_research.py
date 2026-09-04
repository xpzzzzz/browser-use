"""End-to-end WebEvidence research demo: a real browser, real evidence capture, real verification.

This is the first run of the whole chain on one task:

    user task
      -> Browser Use agent driving Chromium
      -> EvidenceCollector storing one pre-action observation per step
      -> final answer
      -> WebEvidencePipeline: claim extraction, lexical alignment, semantic reranking,
         claim-level verification, evidence organization, grounded report
      -> report.json and report.md

The evidence data flow is deliberately pre-action, so a node records the page the model was looking at
when it chose the action that follows it::

    evidence[N] -> action[N] -> evidence[N+1]

Setup: put ``ALIBABA_CLOUD`` in the environment or in a local ``.env`` (get a key at
https://modelstudio.console.alibabacloud.com/?tab=playground#/api-key). Keys are region scoped, so a
mainland account also needs ``ALIBABA_CLOUD_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1``.

Usage::

    uv run python examples/webevidence_research.py --task "Find the Browser Use GitHub repository and report its primary language."
    uv run python examples/webevidence_research.py --task "..." --postprocess-max-attempts 1

Only the three post-processing stages run through a bounded retry wrapper. The agent keeps the plain
model, because Phase 9B found the transient failures in claim extraction, reranking and verification,
not in browsing, and leaving the browser half untouched keeps that comparison honest. Setting
``--postprocess-max-attempts 1`` turns retry off and reproduces the Phase 9B behaviour.

This demo opens a browser and spends real model calls, so it is not part of the automated test suite:
``pytest`` never runs anything in ``examples/``, and the pipeline's own tests use a fake model.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError
from uuid_extensions import uuid7str

load_dotenv()

# The report markdown carries emoji, and redirected output gets the locale codepage, which is cp936 on
# Windows and cannot encode them. Setting this here rather than in main() means every printing path,
# including a caller importing this module, has the same permission the report writers already have.
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from browser_use import Agent, Browser, ChatOpenAI
from browser_use.evidence import (
	ClaimExtractor,
	ClaimVerifier,
	EvidenceAligner,
	EvidenceCollector,
	EvidenceNode,
	EvidenceOrganizer,
	EvidenceReportBuilder,
	JsonlEvidenceStore,
	LLMRetryPolicy,
	LLMRetryStats,
	MarkdownReportRenderer,
	RetryingChatModel,
	SemanticEvidenceReranker,
	WebEvidencePipeline,
	WebEvidencePipelineError,
	WebEvidencePipelineResult,
)
from browser_use.llm.base import BaseChatModel

# The model under evaluation. Kept fixed on purpose: this demo exists to show what this model does, so
# it must not quietly fall back to a different one when a stage struggles.
MODEL = 'qwen3.8-flash'
API_KEY_ENV = 'ALIBABA_CLOUD'
BASE_URL_ENV = 'ALIBABA_CLOUD_BASE_URL'
DEFAULT_BASE_URL = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'

DEFAULT_TASK = 'Find the official Browser Use GitHub repository and report whether it is primarily written in Python.'

OUTPUT_ROOT = Path('tmp') / 'webevidence'


def build_pipeline(llm: BaseChatModel) -> WebEvidencePipeline:
	"""Wire Phases 3 to 7 with the same client everywhere, injected so nothing here builds a provider.

	Each stage takes a ``BaseChatModel``, so the caller decides whether that model retries. Nothing in
	Phases 3 to 7 knows the difference.
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


async def run_agent(
	task: str,
	*,
	task_id: str,
	llm: ChatOpenAI,
	run_dir: Path,
	max_steps: int,
	headless: bool,
	extensions: bool,
) -> tuple[str, list[EvidenceNode]]:
	"""Drive the browser and collect evidence, then return the final answer and the captured nodes."""
	store = JsonlEvidenceStore(run_dir / 'evidence.jsonl')
	collector = EvidenceCollector(task_id=task_id, store=store, screenshot_dir=run_dir / 'screenshots')

	# collector.collect_step matches the step callback contract exactly: the pre-action
	# BrowserStateSummary, the AgentOutput for that step, and the 1-based step number.
	agent = Agent(
		task=task,
		llm=llm,
		browser=Browser(
			headless=headless,
			# A profile of its own inside the run directory: it cannot collide with a browser the user
			# already has open, and two demo runs never fight over one profile lock.
			user_data_dir=str(run_dir / 'profile'),
			# The automation extensions are fetched on first use, which stalls for minutes on a network
			# that cannot reach their CDN. They help with ads and cookie banners, so they stay on unless
			# the caller turns them off.
			enable_default_extensions=extensions,
		),
		register_new_step_callback=collector.collect_step,
		# A text-only model gets a text-only observation. Screenshots are still captured by the browser
		# state and still saved as evidence; they are simply not sent back to the model.
		use_vision=False,
	)

	history = await agent.run(max_steps=max_steps)

	answer = history.final_result()
	if not answer or not answer.strip():
		raise SystemExit(
			f'The agent produced no final answer (done={history.is_done()}, successful={history.is_successful()}). '
			'Nothing was verified, because there was nothing to verify.'
		)

	nodes = store.load_all()
	if not nodes:
		print(f'warning: no evidence was captured for task {task_id}; every claim will report NO_EVIDENCE', file=sys.stderr)

	return answer, nodes


def write_outputs(result: WebEvidencePipelineResult, run_dir: Path) -> tuple[Path, Path, Path]:
	"""Persist the report, its Markdown, and the full pipeline result for later inspection."""
	report_json = run_dir / 'report.json'
	report_md = run_dir / 'report.md'
	pipeline_result = run_dir / 'pipeline_result.json'

	report_json.write_text(result.report.model_dump_json(indent=2), encoding='utf-8')
	report_md.write_text(result.markdown, encoding='utf-8')
	pipeline_result.write_text(result.model_dump_json(indent=2), encoding='utf-8')
	return report_json, report_md, pipeline_result


def print_retry_stats(stats: LLMRetryStats) -> None:
	"""What the post-processing calls cost. Counts only, because a provider message can echo the answer."""
	print('post-processing LLM retry:')
	print(f'  logical calls: {stats.logical_invocation_count}')
	print(f'  provider attempts: {stats.attempt_count}')
	print(f'  retries: {stats.retry_count}')
	print(f'  recovered calls: {stats.recovered_invocation_count}')
	print(f'  failed calls: {stats.failed_invocation_count}')


def print_summary(result: WebEvidencePipelineResult, paths: tuple[Path, Path, Path], retry_stats: LLMRetryStats) -> None:
	"""Report what happened, in the order a reader asks for it. Never prints a key or raw page content."""
	summary = result.report.summary
	print(f'task_id: {result.task_id}')
	print(f'evidence captured: {result.evidence_count}')
	print(f'evidence used in verification: {result.evidence_graph.stats.evidence_count}')
	print(f'claims extracted: {summary.claim_count}')
	print('verification summary:')
	for label, count in (
		('SUPPORTED', summary.supported_claim_count),
		('PARTIAL', summary.partial_claim_count),
		('UNSUPPORTED', summary.unsupported_claim_count),
		('CONTRADICTED', summary.contradicted_claim_count),
		('CONFLICTED', summary.conflicted_claim_count),
		('NO_EVIDENCE', summary.no_evidence_claim_count),
	):
		print(f'  {label}: {count}')
	print_retry_stats(retry_stats)
	print(f'report.json: {paths[0]}')
	print(f'report.md: {paths[1]}')
	print(f'pipeline_result.json: {paths[2]}')
	print()
	print(result.markdown)


async def main() -> int:
	parser = argparse.ArgumentParser(
		description='Run a Browser Use research task and verify its answer against captured evidence.'
	)
	parser.add_argument('--task', default=DEFAULT_TASK, help='The research task to run.')
	parser.add_argument('--max-steps', type=int, default=12, help='Maximum agent steps before it must answer.')
	parser.add_argument('--headful', action='store_true', help='Show the browser window instead of running headless.')
	parser.add_argument(
		'--no-extensions',
		action='store_true',
		help='Skip the ad-block and cookie-banner extensions, for networks that cannot reach their CDN.',
	)
	parser.add_argument('--debug', action='store_true', help='Re-raise pipeline errors with their original traceback.')
	parser.add_argument(
		'--postprocess-max-attempts',
		type=int,
		default=3,
		help='Total post-processing model attempts per call, so 3 means two retries; 1 turns retry off.',
	)
	parser.add_argument(
		'--postprocess-retry-delay',
		type=float,
		default=1.0,
		help='Seconds before the first post-processing retry, doubling up to an 8s ceiling.',
	)
	args = parser.parse_args()

	try:
		retry_policy = LLMRetryPolicy(
			max_attempts=args.postprocess_max_attempts, initial_delay_seconds=args.postprocess_retry_delay
		)
	except ValidationError:
		# Before the key check, so this message is never masked by an unrelated one.
		print(
			'--postprocess-max-attempts must be at least 1, and --postprocess-retry-delay must not exceed the 8s ceiling',
			file=sys.stderr,
		)
		return 1

	api_key = os.getenv(API_KEY_ENV)
	if not api_key:
		raise SystemExit(f'{API_KEY_ENV} is not set. Add it to the environment or to a local .env file.')

	base_url = os.getenv(BASE_URL_ENV) or DEFAULT_BASE_URL
	llm = ChatOpenAI(model=MODEL, api_key=api_key, base_url=base_url, temperature=0.0)
	# The wrapper re-asks the same client rather than falling back to another one, and the agent above stays
	# unwrapped so browsing behaves exactly as it did before Phase 9C.
	postprocess_llm = RetryingChatModel(llm, policy=retry_policy)
	print(f'model: {MODEL} via {base_url}')
	print(f'post-processing retry: {retry_policy.max_attempts} attempts per call')

	task_id = uuid7str()
	run_dir = OUTPUT_ROOT / task_id
	run_dir.mkdir(parents=True, exist_ok=True)
	print(f'run directory: {run_dir}')

	try:
		answer, nodes = await run_agent(
			args.task,
			task_id=task_id,
			llm=llm,
			run_dir=run_dir,
			max_steps=args.max_steps,
			headless=not args.headful,
			extensions=not args.no_extensions,
		)
	except Exception as e:
		# The browser stage can fail for environmental reasons a model-level message cannot express,
		# so say what actually happened to the evidence before stopping.
		captured = len(JsonlEvidenceStore(run_dir / 'evidence.jsonl').load_all())
		print(
			f'agent run failed before producing an answer: {type(e).__name__}; evidence captured so far: {captured}',
			file=sys.stderr,
		)
		print(f'run directory: {run_dir}', file=sys.stderr)
		if args.debug:
			raise
		return 1

	try:
		result = await build_pipeline(postprocess_llm).analyze(
			task_id=task_id, task=args.task, answer=answer, evidence_nodes=nodes
		)
	except WebEvidencePipelineError as e:
		# The pipeline message is safe to print: it names the stage and the exception type only. The
		# original exception stays available under --debug for whoever is fixing it.
		print(f'pipeline failed at {e.stage.value}: {type(e.__cause__).__name__}', file=sys.stderr)
		# The attempts this run spent are part of what failed, so report them before stopping.
		print_retry_stats(postprocess_llm.snapshot_stats())
		if args.debug:
			raise
		return 1

	print_summary(result, write_outputs(result, run_dir), postprocess_llm.snapshot_stats())
	return 0


if __name__ == '__main__':
	raise SystemExit(asyncio.run(main()))
