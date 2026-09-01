"""End-to-end orchestration from a final answer to an evidence-grounded report.

Phases 3 to 7 are already built and each one is strict about what it accepts. This module adds no new
evidence algorithm: it runs them in the only order that makes sense, so a caller hands over a task,
the answer the agent gave, and the evidence captured while it worked, and gets back every intermediate
object plus the structured report and its Markdown rendering.

    task + answer + evidence nodes
        -> ClaimSet            (Phase 3, model)
        -> AlignmentResult     (Phase 4A, pure)
        -> RerankingResult     (Phase 4B, model)
        -> VerificationResult  (Phase 5, model)
        -> EvidenceGraph       (Phase 6, pure)
        -> report + markdown   (Phase 7, pure)

Two boundaries keep the stage honest. The pipeline never constructs a model, reads a key, or opens a
browser: components arrive injected, so which model judges the claims stays the caller's decision and
the orchestration stays testable with a fake. And the pipeline never reads the evidence store: capture
and persistence belong to the collector, while verification consumes whatever evidence the caller has
already assembled, which is also what makes an in-memory test identical in shape to a real run.

Failure is loud by design. Every stage error becomes a :class:`WebEvidencePipelineError` naming the
stage that produced it, with the original exception preserved as ``__cause__``. There is no fallback
of any kind, because "the reranker was unreachable" and "this claim is unsupported" are different
facts, and quietly substituting one for the other would put a fabricated verification in front of a
user. Retry and degradation policy belongs above this layer, where a caller can decide what a partial
answer is worth.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from enum import Enum

from pydantic import BaseModel, Field

from browser_use.evidence.alignment import AlignmentResult, EvidenceAligner
from browser_use.evidence.claim_extractor import ClaimExtractor
from browser_use.evidence.claims import ClaimSet
from browser_use.evidence.models import EvidenceNode
from browser_use.evidence.organization import EvidenceGraph, EvidenceOrganizer
from browser_use.evidence.reporting import EvidenceGroundedReport, EvidenceReportBuilder, MarkdownReportRenderer
from browser_use.evidence.reranking import RerankingResult, SemanticEvidenceReranker
from browser_use.evidence.verification import ClaimVerifier, VerificationResult


class PipelineStage(str, Enum):
	"""Which step of the pipeline a failure came from, in execution order."""

	CLAIM_EXTRACTION = 'CLAIM_EXTRACTION'
	LEXICAL_ALIGNMENT = 'LEXICAL_ALIGNMENT'
	SEMANTIC_RERANKING = 'SEMANTIC_RERANKING'
	VERIFICATION = 'VERIFICATION'
	ORGANIZATION = 'ORGANIZATION'
	REPORTING = 'REPORTING'


class WebEvidencePipelineError(RuntimeError):
	"""Raised when a stage fails or the stages disagree about which task they are working on.

	``stage`` says where it went wrong and ``__cause__`` keeps the original exception, while the
	message stays short and names only the exception type. A provider error can echo the request it
	was given, and at these stages the request contains the answer under verification, scraped page
	text, and sometimes a credential in a header -- none of which belongs in a log line or a traceback
	anyone might paste somewhere.
	"""

	def __init__(self, message: str, *, stage: PipelineStage) -> None:
		super().__init__(message)
		self.stage = stage


class WebEvidencePipelineResult(BaseModel):
	"""The outcome of one analysis, including every intermediate object that produced it.

	Nothing here is a summary the model wrote, a timestamp, or a run identifier invented for the
	occasion: the same task, answer and evidence through the same components with the same model
	outputs yields an identical object, which is what lets a report be diffed and a benchmark be
	reproduced. ``evidence_count`` is how many nodes came in, so a caller can see at a glance how much
	of the run was captured; ``result.evidence_graph.stats.evidence_count`` stays the narrower count of
	the evidence that actually took part in verification.
	"""

	task_id: str = Field(description='Task id the analysis was run for')
	task: str = Field(description='Original task prompt, passed through unchanged')
	answer: str = Field(description='Final answer that was verified, passed through unchanged')
	evidence_count: int = Field(ge=0, description='Evidence nodes handed to the pipeline')
	claim_set: ClaimSet = Field(description='Phase 3 output: the atomic claims extracted from the answer')
	alignment_result: AlignmentResult = Field(description='Phase 4A output: lexical candidates per claim')
	reranking_result: RerankingResult = Field(description='Phase 4B output: candidates rescored and ranked per claim')
	verification_result: VerificationResult = Field(description='Phase 5 output: per-evidence relations and claim statuses')
	evidence_graph: EvidenceGraph = Field(description='Phase 6 output: claims, used evidence, edges and stats')
	report: EvidenceGroundedReport = Field(description='Phase 7 output: the structured user report')
	markdown: str = Field(description='Phase 7 output: the rendered report')


class WebEvidencePipeline:
	"""Run Phases 3 to 7 in order over one task, its answer and its captured evidence.

	Every component is injected. That keeps the pipeline free of any particular provider, so a
	benchmark can swap models or thresholds without touching this file, and a unit test can drive the
	whole chain with fakes while a demo drives the same code with a real model.

	Each stage consumes exactly the previous stage's output, which is what item 7 of the phase means by
	forbidding shortcuts: the report is built from the graph, the graph from the verification, the
	verification from the reranked candidates. Reaching past an intermediate to reuse an earlier input
	would let a stale object slip into a finished report, and the whole point of the chain is that every
	number in the output can be traced through each stage that produced it.
	"""

	def __init__(
		self,
		*,
		claim_extractor: ClaimExtractor,
		aligner: EvidenceAligner,
		reranker: SemanticEvidenceReranker,
		verifier: ClaimVerifier,
		organizer: EvidenceOrganizer,
		report_builder: EvidenceReportBuilder,
		markdown_renderer: MarkdownReportRenderer,
	) -> None:
		self.claim_extractor = claim_extractor
		self.aligner = aligner
		self.reranker = reranker
		self.verifier = verifier
		self.organizer = organizer
		self.report_builder = report_builder
		self.markdown_renderer = markdown_renderer

	async def analyze(
		self,
		*,
		task_id: str,
		task: str,
		answer: str,
		evidence_nodes: Sequence[EvidenceNode],
	) -> WebEvidencePipelineResult:
		"""Verify one answer against captured evidence and render the report.

		Empty input is a result, not an error. An empty answer yields no claims and a report with zero
		claim sections; an answer with claims but no evidence still walks every stage and comes out with
		``NO_EVIDENCE`` for each claim, because "nothing was captured" is a finding the report should say
		out rather than something to crash on.

		Raises:
			WebEvidencePipelineError: when any stage raises, or when a stage reports a task id other than
				the one it was given. ``stage`` names the failing step and ``__cause__`` keeps the
				original exception.
		"""
		with self._stage(PipelineStage.CLAIM_EXTRACTION):
			claim_set = await self.claim_extractor.extract(task_id=task_id, task=task, answer=answer)

		with self._stage(PipelineStage.LEXICAL_ALIGNMENT):
			alignment_result = self.aligner.align(claim_set=claim_set, evidence_nodes=evidence_nodes)

		with self._stage(PipelineStage.SEMANTIC_RERANKING):
			reranking_result = await self.reranker.rerank(
				claim_set=claim_set, alignment_result=alignment_result, evidence_nodes=evidence_nodes
			)

		with self._stage(PipelineStage.VERIFICATION):
			verification_result = await self.verifier.verify(
				claim_set=claim_set, reranking_result=reranking_result, evidence_nodes=evidence_nodes
			)

		with self._stage(PipelineStage.ORGANIZATION):
			evidence_graph = self.organizer.organize(
				claim_set=claim_set, verification_result=verification_result, evidence_nodes=evidence_nodes
			)

		with self._stage(PipelineStage.REPORTING):
			report = self.report_builder.build(claim_set=claim_set, evidence_graph=evidence_graph)
			markdown = self.markdown_renderer.render(report)

		self._check_task_ids(
			task_id,
			[
				(PipelineStage.CLAIM_EXTRACTION, claim_set.task_id),
				(PipelineStage.LEXICAL_ALIGNMENT, alignment_result.task_id),
				(PipelineStage.SEMANTIC_RERANKING, reranking_result.task_id),
				(PipelineStage.VERIFICATION, verification_result.task_id),
				(PipelineStage.ORGANIZATION, evidence_graph.task_id),
				(PipelineStage.REPORTING, report.task_id),
			],
		)

		return WebEvidencePipelineResult(
			task_id=task_id,
			task=task,
			answer=answer,
			evidence_count=len(evidence_nodes),
			claim_set=claim_set,
			alignment_result=alignment_result,
			reranking_result=reranking_result,
			verification_result=verification_result,
			evidence_graph=evidence_graph,
			report=report,
			markdown=markdown,
		)

	@contextmanager
	def _stage(self, stage: PipelineStage) -> Iterator[None]:
		"""Translate any component failure into an error that says where it happened.

		The message carries the exception type and nothing else. ``raise ... from`` keeps the real
		exception one link away for whoever debugs it, so nothing needs to be pasted into the message to
		make the failure diagnosable.
		"""
		try:
			yield
		except Exception as e:
			raise WebEvidencePipelineError(
				f'WebEvidence pipeline failed at {stage.value}: {type(e).__name__}', stage=stage
			) from e

	@staticmethod
	def _check_task_ids(task_id: str, seen: Sequence[tuple[PipelineStage, str]]) -> None:
		"""Refuse to publish a report that mixes two tasks' data.

		Each component already checks the ids it receives, so this is the outer guarantee: an object
		arriving at the result carries the task id it was asked for, and a component that loses it fails
		here instead of producing citations that lead to another run's evidence.
		"""
		for stage, stage_task_id in seen:
			if stage_task_id != task_id:
				raise WebEvidencePipelineError(
					f'WebEvidence pipeline failed at {stage.value}: produced task_id {stage_task_id!r}, expected {task_id!r}',
					stage=stage,
				)
