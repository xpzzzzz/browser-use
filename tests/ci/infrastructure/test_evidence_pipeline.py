"""Tests for the Phase 3 to Phase 7 orchestration.

These run the real claim extractor, aligner, reranker, verifier, organizer, report builder and renderer
against a fake chat model, so the wiring between stages is covered without a browser, a network call or
a real Qwen request. Letter tags in comments map onto the Phase 8 test checklist.
"""

import ast as _ast
import re
from copy import deepcopy
from pathlib import Path as _Path
from typing import Any

import pytest

from browser_use.evidence import (
	AlignmentResult,
	ClaimExtractor,
	ClaimSet,
	ClaimVerifier,
	EvidenceAligner,
	EvidenceGraph,
	EvidenceGroundedReport,
	EvidenceNode,
	EvidenceOrganizer,
	EvidenceRelation,
	EvidenceReportBuilder,
	MarkdownReportRenderer,
	PipelineStage,
	RerankingResult,
	SemanticEvidenceReranker,
	VerificationResult,
	VerificationStatus,
	WebEvidencePipeline,
	WebEvidencePipelineError,
	WebEvidencePipelineResult,
)
from browser_use.evidence.claim_extractor import RawClaim, RawClaimExtraction
from browser_use.evidence.reranking import RawSemanticEvidenceScore, RawSemanticReranking
from browser_use.evidence.verification import RawClaimEvidenceAssessment, RawEvidenceAssessment
from browser_use.llm.views import ChatInvokeCompletion

_EVIDENCE_ID_PATTERN = re.compile(r'^evidence_id: (.+)$', re.MULTILINE)

STARS_CLAIM = 'Browser Use has more than 100,000 GitHub stars.'
LANGUAGE_CLAIM = 'Browser Use is primarily written in Python.'
HIGH_STAR_TEXT = 'Browser Use has 111,799 GitHub stars.'
LOW_STAR_TEXT = 'Browser Use has only 30,000 GitHub stars.'
LANGUAGE_TEXT = 'Browser Use is primarily written in Python.'

TASK = 'How popular is Browser Use, and what is it written in?'
ANSWER = f'{STARS_CLAIM} {LANGUAGE_CLAIM}'
SECRET = 'secret123'


def _module_imports(module_name: str) -> set[str]:
	"""The import surface of one module, which is how these tests prove what a stage may not touch."""
	_spec = __import__(module_name, fromlist=['__file__'])
	tree = _ast.parse(_Path(_spec.__file__).read_text(encoding='utf-8'))
	imported: set[str] = set()
	for node in _ast.walk(tree):
		if isinstance(node, _ast.Import):
			imported.update(alias.name.partition('.')[0] for alias in node.names)
		elif isinstance(node, _ast.ImportFrom) and node.module:
			imported.add(node.module)
	return imported


_PIPELINE_IMPORTS = _module_imports('browser_use.evidence.pipeline')


class PipelineChatModel:
	"""One fake model answering every structured request the pipeline makes.

	``fail_on`` names the schema whose call should explode, which is how the tests take down one stage at
	a time while leaving the others working.
	"""

	def __init__(
		self,
		*,
		claim_texts: tuple[str, ...] = (STARS_CLAIM,),
		relations: dict[str, EvidenceRelation] | None = None,
		default_relation: EvidenceRelation = EvidenceRelation.SUPPORTS,
		fail_on: tuple[str, ...] = (),
	) -> None:
		self.model = 'pipeline-fake-model'
		self.provider = 'fake'
		self.name = 'pipeline-fake-model'
		self.model_name = 'pipeline-fake-model'
		self._verified_api_keys = True
		self.claim_texts = claim_texts
		self.relations = dict(relations or {})
		self.default_relation = default_relation
		self.fail_on = set(fail_on)
		self.calls: list[str] = []

	@property
	def stage_names(self) -> list[str]:
		return list(self.calls)

	async def ainvoke(self, messages, output_format=None, **kwargs) -> ChatInvokeCompletion:
		schema = getattr(output_format, '__name__', '')
		self.calls.append(schema)
		if schema in self.fail_on:
			raise RuntimeError(f'provider exploded api-key={SECRET} with the whole prompt in its message')

		prompt = messages[-1].text
		candidate_ids = _EVIDENCE_ID_PATTERN.findall(prompt)
		if schema == 'RawClaimExtraction':
			completion: Any = RawClaimExtraction(claims=[RawClaim(text=text) for text in self.claim_texts])
		elif schema == 'RawSemanticReranking':
			completion = RawSemanticReranking(
				scores=[RawSemanticEvidenceScore(evidence_id=evidence_id, relevance_score=0.9) for evidence_id in candidate_ids]
			)
		elif schema == 'RawClaimEvidenceAssessment':
			completion = RawClaimEvidenceAssessment(
				assessments=[
					RawEvidenceAssessment(
						evidence_id=evidence_id,
						relation=self.relations.get(evidence_id, self.default_relation),
						explanation=f'{evidence_id} reports a figure that settles the claim.',
					)
					for evidence_id in candidate_ids
				]
			)
		else:
			raise AssertionError(f'the pipeline asked for an unexpected schema: {schema!r}')

		return ChatInvokeCompletion(completion=completion, usage=None)


def _node(
	evidence_id: str, text: str, *, url: str, title: str = '', step_number: int = 1, task_id: str = 'task-1'
) -> EvidenceNode:
	return EvidenceNode(
		evidence_id=evidence_id,
		task_id=task_id,
		step_number=step_number,
		url=url,
		title=title,
		text=text,
	)


def _supporting() -> list[EvidenceNode]:
	return [
		_node('evidence-high', HIGH_STAR_TEXT, url='https://github.com/browser-use/browser-use', title='GitHub', step_number=1)
	]


def _contradicting() -> list[EvidenceNode]:
	return [
		_node('evidence-high', HIGH_STAR_TEXT, url='https://github.com/browser-use/browser-use', title='GitHub', step_number=1),
		_node('evidence-low', LOW_STAR_TEXT, url='https://blog.example.com/old', title='Old post', step_number=2),
	]


def _pipeline(model: PipelineChatModel | None = None, **overrides: Any) -> tuple[WebEvidencePipeline, PipelineChatModel]:
	"""Build the pipeline from the real Phase 3-7 components, with only the model faked."""
	model = model or PipelineChatModel()
	components: dict[str, Any] = {
		'claim_extractor': ClaimExtractor(model),
		'aligner': EvidenceAligner(top_k=5),
		'reranker': SemanticEvidenceReranker(model),
		'verifier': ClaimVerifier(model),
		'organizer': EvidenceOrganizer(),
		'report_builder': EvidenceReportBuilder(),
		'markdown_renderer': MarkdownReportRenderer(),
	}
	components.update(overrides)
	return WebEvidencePipeline(**components), model


async def _analyze(
	pipeline: WebEvidencePipeline,
	*,
	task_id: str = 'task-1',
	evidence_nodes: list[EvidenceNode] | None = None,
) -> WebEvidencePipelineResult:
	# ``[]`` has to mean "no evidence", so an absent argument needs a sentinel, not ``or``.
	if evidence_nodes is None:
		evidence_nodes = _supporting()
	return await pipeline.analyze(task_id=task_id, task=TASK, answer=ANSWER, evidence_nodes=evidence_nodes)


class TestHappyPath:
	async def test_a_supporting_page_ends_as_supported(self):  # A
		pipeline, _ = _pipeline()

		result = await _analyze(pipeline)

		assert [section.status for section in result.report.claims] == [VerificationStatus.SUPPORTED]
		assert result.report.summary.supported_claim_count == 1

	async def test_a_refuting_page_ends_as_contradicted(self):  # B
		model = PipelineChatModel(default_relation=EvidenceRelation.CONTRADICTS)
		pipeline, _ = _pipeline(model)

		result = await _analyze(pipeline)

		assert [section.status for section in result.report.claims] == [VerificationStatus.CONTRADICTED]
		assert result.report.claims[0].evidence[0].relation is EvidenceRelation.CONTRADICTS

	async def test_support_and_refutation_end_as_conflicted(self):  # C
		model = PipelineChatModel(
			relations={'evidence-high': EvidenceRelation.SUPPORTS, 'evidence-low': EvidenceRelation.CONTRADICTS}
		)
		pipeline, _ = _pipeline(model)

		result = await _analyze(pipeline, evidence_nodes=_contradicting())

		section = result.report.claims[0]
		assert section.status is VerificationStatus.CONFLICTED
		# The conflict structure survives all the way into the report, in rerank order, with labels.
		assert {item.evidence_id: item.conflicting_evidence_ids for item in section.evidence} == {
			'evidence-high': ['evidence-low'],
			'evidence-low': ['evidence-high'],
		}
		assert 'Conflicts with: [E' in result.markdown

	async def test_no_evidence_ends_as_no_evidence_without_skipping_stages(self):  # D, item 9
		pipeline, model = _pipeline()

		result = await _analyze(pipeline, evidence_nodes=[])

		assert [section.status for section in result.report.claims] == [VerificationStatus.NO_EVIDENCE]
		assert result.evidence_graph.stats.evidence_count == 0
		# Only claim extraction costs a call: nothing was retrieved, so there is nothing to score or verify.
		assert model.stage_names == ['RawClaimExtraction']

	async def test_an_empty_answer_yields_a_valid_report_with_no_claims(self):  # E, item 9
		pipeline, model = _pipeline()

		result = await pipeline.analyze(task_id='task-1', task=TASK, answer='', evidence_nodes=_supporting())

		assert result.claim_set.claims == []
		assert result.report.claims == []
		assert result.report.summary.claim_count == 0
		assert result.report.summary.evidence_coverage_rate == 0.0
		assert '## Claim Verification' in result.markdown
		# Phase 3 short-circuits a blank answer, so an empty answer costs no model call at all.
		assert model.stage_names == []

	async def test_an_answer_without_factual_claims_still_completes_the_chain(self):  # E
		pipeline, model = _pipeline(PipelineChatModel(claim_texts=()))

		result = await _analyze(pipeline)

		assert result.claim_set.claims == []
		assert result.report.summary.claim_count == 0
		assert model.stage_names == ['RawClaimExtraction']

	async def test_every_intermediate_object_is_returned(self):  # F
		pipeline, _ = _pipeline()

		result = await _analyze(pipeline, evidence_nodes=_contradicting())

		assert isinstance(result.claim_set, ClaimSet)
		assert isinstance(result.alignment_result, AlignmentResult)
		assert isinstance(result.reranking_result, RerankingResult)
		assert isinstance(result.verification_result, VerificationResult)
		assert isinstance(result.evidence_graph, EvidenceGraph)
		assert isinstance(result.report, EvidenceGroundedReport)
		assert isinstance(result.markdown, str)
		# Each stage output still describes the same task, which is what makes them one chain.
		assert [result.claim_set.task_id, result.evidence_graph.task_id, result.report.task_id] == ['task-1'] * 3

	async def test_evidence_counts_input_nodes_not_the_used_subset(self):  # G
		pipeline, _ = _pipeline()
		nodes = [
			*_supporting(),
			_node(
				'evidence-uncited',
				'A page about the weather.',
				url='https://weather.example.com/today',
				title='Weather',
				step_number=2,
			),
		]

		result = await _analyze(pipeline, evidence_nodes=nodes)

		assert result.evidence_count == 2
		assert result.evidence_graph.stats.evidence_count == 1
		assert result.report.summary.evidence_count == 1

	async def test_markdown_is_non_empty_and_carries_the_status(self):  # H, I
		pipeline, _ = _pipeline()

		result = await _analyze(pipeline)

		assert result.markdown.strip()
		assert '### Claim 1: SUPPORTED' in result.markdown
		assert STARS_CLAIM in result.markdown


class TestStageContract:
	async def test_stages_run_in_order_once_each(self):  # J, item 6
		pipeline, model = _pipeline(PipelineChatModel(claim_texts=(STARS_CLAIM, LANGUAGE_CLAIM)))

		await _analyze(pipeline)

		# One extraction, then the rerank of every claim, then the verification of every claim: each
		# stage finishes before the next begins, which is what keeps the chain auditable.
		assert model.stage_names == [
			'RawClaimExtraction',
			'RawSemanticReranking',
			'RawSemanticReranking',
			'RawClaimEvidenceAssessment',
			'RawClaimEvidenceAssessment',
		]

	def test_the_pipeline_holds_no_model_browser_or_store_of_its_own(self):
		"""Item 5 and 8: components come in, and nothing here can reach a provider or a file."""
		pipeline, _ = _pipeline()

		assert set(vars(pipeline)) == {
			'claim_extractor',
			'aligner',
			'reranker',
			'verifier',
			'organizer',
			'report_builder',
			'markdown_renderer',
		}
		assert _PIPELINE_IMPORTS == {
			'collections.abc',
			'contextlib',
			'enum',
			'pydantic',
			'browser_use.evidence.alignment',
			'browser_use.evidence.claim_extractor',
			'browser_use.evidence.claims',
			'browser_use.evidence.models',
			'browser_use.evidence.organization',
			'browser_use.evidence.reporting',
			'browser_use.evidence.reranking',
			'browser_use.evidence.verification',
		}, sorted(_PIPELINE_IMPORTS)

	def test_a_fresh_pipeline_never_touches_a_model(self):
		model = PipelineChatModel(fail_on=('RawClaimExtraction', 'RawSemanticReranking', 'RawClaimEvidenceAssessment'))

		pipeline, _ = _pipeline(model)

		assert model.calls == []
		assert pipeline.reranker.llm is model
		assert pipeline.verifier.llm is model


class TestFailureWrapping:
	"""Item 10: a failed stage is reported as a failed stage, with its cause attached and nothing invented."""

	async def test_claim_extraction_failure_names_its_stage(self):  # K
		pipeline, _ = _pipeline(PipelineChatModel(fail_on=('RawClaimExtraction',)))

		with pytest.raises(WebEvidencePipelineError) as excinfo:
			await _analyze(pipeline)

		assert excinfo.value.stage is PipelineStage.CLAIM_EXTRACTION
		assert 'CLAIM_EXTRACTION' in str(excinfo.value)

	async def test_alignment_failure_names_its_stage(self):  # K, pure stage
		pipeline, _ = _pipeline()
		duplicated = [*(_supporting()[0],)] + [deepcopy(_supporting()[0])]

		with pytest.raises(WebEvidencePipelineError) as excinfo:
			await _analyze(pipeline, evidence_nodes=duplicated)

		assert excinfo.value.stage is PipelineStage.LEXICAL_ALIGNMENT

	async def test_reranking_failure_names_its_stage(self):  # L
		pipeline, _ = _pipeline(PipelineChatModel(fail_on=('RawSemanticReranking',)))

		with pytest.raises(WebEvidencePipelineError) as excinfo:
			await _analyze(pipeline)

		assert excinfo.value.stage is PipelineStage.SEMANTIC_RERANKING

	async def test_verification_failure_names_its_stage(self):  # M
		pipeline, _ = _pipeline(PipelineChatModel(fail_on=('RawClaimEvidenceAssessment',)))

		with pytest.raises(WebEvidencePipelineError) as excinfo:
			await _analyze(pipeline)

		assert excinfo.value.stage is PipelineStage.VERIFICATION

	async def test_organization_failure_names_its_stage(self):  # N
		class _BrokenOrganizer:
			def organize(self, **_kwargs):
				raise RuntimeError('graph boom')

		pipeline, _ = _pipeline(organizer=_BrokenOrganizer())

		with pytest.raises(WebEvidencePipelineError) as excinfo:
			await _analyze(pipeline)

		assert excinfo.value.stage is PipelineStage.ORGANIZATION

	async def test_reporting_failure_names_its_stage(self):  # O
		class _BrokenReportBuilder:
			def build(self, **_kwargs):
				raise RuntimeError('report boom')

		pipeline, _ = _pipeline(report_builder=_BrokenReportBuilder())

		with pytest.raises(WebEvidencePipelineError) as excinfo:
			await _analyze(pipeline)

		assert excinfo.value.stage is PipelineStage.REPORTING

	async def test_rendering_failure_names_the_reporting_stage(self):
		class _BrokenRenderer:
			def render(self, _report):
				raise RuntimeError('markdown boom')

		pipeline, _ = _pipeline(markdown_renderer=_BrokenRenderer())

		with pytest.raises(WebEvidencePipelineError) as excinfo:
			await _analyze(pipeline)

		assert excinfo.value.stage is PipelineStage.REPORTING

	async def test_provider_secrets_stay_out_of_the_pipeline_message(self):  # P
		pipeline, _ = _pipeline(PipelineChatModel(fail_on=('RawSemanticReranking',)))

		with pytest.raises(WebEvidencePipelineError) as excinfo:
			await _analyze(pipeline)

		message = str(excinfo.value)
		assert SECRET not in message
		assert 'prompt' not in message.lower()
		assert HIGH_STAR_TEXT not in message
		# The stage's own error type is named, and it is the type the component raised, not the provider's.
		assert 'EvidenceRerankingError' in message
		assert _module_imports('browser_use.evidence.pipeline') == _PIPELINE_IMPORTS

	async def test_the_original_exception_is_kept_as_the_cause(self):  # Q
		pipeline, _ = _pipeline(PipelineChatModel(fail_on=('RawClaimEvidenceAssessment',)))

		with pytest.raises(WebEvidencePipelineError) as excinfo:
			await _analyze(pipeline)

		# The chain survives whole: pipeline error -> stage error -> the provider exception underneath.
		stage_error = excinfo.value.__cause__
		assert type(stage_error).__name__ == 'ClaimVerificationError'
		assert isinstance(stage_error.__cause__, RuntimeError)
		assert SECRET in str(stage_error.__cause__)

	async def test_a_failing_stage_produces_no_partial_result(self):
		"""Item 10: no degraded stand-in for a verification that never happened."""
		pipeline, _ = _pipeline(PipelineChatModel(fail_on=('RawClaimEvidenceAssessment',)))

		with pytest.raises(WebEvidencePipelineError):
			await _analyze(pipeline)

		# Nothing is returned to inspect, so there is no half-populated report to be fooled by.
		result = WebEvidencePipelineResult(
			task_id='task-1',
			task=TASK,
			answer=ANSWER,
			evidence_count=0,
			claim_set=ClaimSet(task_id='task-1', task=TASK, answer=ANSWER),
			alignment_result=AlignmentResult(task_id='task-1'),
			reranking_result=RerankingResult(task_id='task-1'),
			verification_result=VerificationResult(task_id='task-1'),
			evidence_graph=EvidenceGraph(task_id='task-1'),
			report=EvidenceGroundedReport(task_id='task-1', task=TASK),
			markdown='',
		)
		assert result.report.claims == []

	async def test_a_stage_that_forgets_the_task_id_is_refused(self):  # item 12
		"""The inner stages cross-check too, so this stub isolates the pipeline's own last guard."""

		class _ForgetfulReportBuilder:
			def build(self, **kwargs):
				report = EvidenceReportBuilder().build(**kwargs)
				return report.model_copy(update={'task_id': 'task-someone-elses'})

		pipeline, _ = _pipeline(report_builder=_ForgetfulReportBuilder())

		with pytest.raises(WebEvidencePipelineError, match="expected 'task-1'") as excinfo:
			await _analyze(pipeline)

		assert excinfo.value.stage is PipelineStage.REPORTING

	async def test_a_stage_that_loses_the_task_id_never_reaches_a_report(self):  # item 12
		class _ForgetfulOrganizer:
			def organize(self, **kwargs):
				graph = EvidenceOrganizer().organize(**kwargs)
				return graph.model_copy(update={'task_id': 'task-someone-elses'})

		pipeline, _ = _pipeline(organizer=_ForgetfulOrganizer())

		with pytest.raises(WebEvidencePipelineError) as excinfo:
			await _analyze(pipeline)

		assert excinfo.value.stage is PipelineStage.REPORTING


class TestDeterminismAndPurity:
	async def test_input_evidence_is_not_mutated(self):  # R
		pipeline, _ = _pipeline()
		nodes = _contradicting()
		before = deepcopy(nodes)

		await _analyze(pipeline, evidence_nodes=nodes)

		assert nodes == before

	async def test_repeated_runs_agree_on_the_artifacts(self):  # S
		first_pipeline, _ = _pipeline(PipelineChatModel(claim_texts=(STARS_CLAIM, LANGUAGE_CLAIM)))
		second_pipeline, _ = _pipeline(PipelineChatModel(claim_texts=(STARS_CLAIM, LANGUAGE_CLAIM)))
		nodes = _contradicting()

		first = await _analyze(first_pipeline, evidence_nodes=nodes)
		second = await _analyze(second_pipeline, evidence_nodes=nodes)

		# The user-visible artifacts are identical; only the generated identifiers differ per run.
		assert first.markdown == second.markdown
		assert first.report.summary == second.report.summary
		assert [section.status for section in first.report.claims] == [section.status for section in second.report.claims]
		assert [section.claim_id for section in first.report.claims] != [section.claim_id for section in second.report.claims]

	def test_the_result_carries_no_timestamp_or_run_metadata(self):
		assert set(WebEvidencePipelineResult.model_fields) == {
			'task_id',
			'task',
			'answer',
			'evidence_count',
			'claim_set',
			'alignment_result',
			'reranking_result',
			'verification_result',
			'evidence_graph',
			'report',
			'markdown',
		}

	async def test_result_round_trips_through_json(self):
		pipeline, _ = _pipeline()

		result = await _analyze(pipeline, evidence_nodes=_contradicting())

		assert WebEvidencePipelineResult.model_validate_json(result.model_dump_json()) == result


class TestStageEnum:
	def test_stages_are_the_six_phases_in_order(self):  # item 2
		assert [stage.value for stage in PipelineStage] == [
			'CLAIM_EXTRACTION',
			'LEXICAL_ALIGNMENT',
			'SEMANTIC_RERANKING',
			'VERIFICATION',
			'ORGANIZATION',
			'REPORTING',
		]

	def test_the_error_always_reports_which_stage_failed(self):
		for stage in PipelineStage:
			error = WebEvidencePipelineError(f'WebEvidence pipeline failed at {stage.value}: RuntimeError', stage=stage)
			assert error.stage is stage
			assert isinstance(error, RuntimeError)
