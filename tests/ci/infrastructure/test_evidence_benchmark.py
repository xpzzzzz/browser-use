"""Tests for the offline benchmark harness.

A pytest pass here means the harness measures correctly, not that the pipeline scores 100 percent: the
seed dataset deliberately contains a case the lexical stage cannot recall. Live models are never called;
semantic and full modes are exercised with a fake chat model, and the seed benchmark's own numbers are
checked only for the one mode that needs no model at all.
"""

import re
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import pytest

from browser_use.evidence import Claim, EvidenceAligner, EvidenceNode, EvidenceRelation, VerificationStatus
from browser_use.evidence.benchmark import (
	RELATION_LABELS,
	STATUS_LABELS,
	BenchmarkRunCaseResult,
	BenchmarkStage,
	BenchmarkSummary,
	EvidenceBenchmarkCase,
	EvidenceBenchmarkError,
	EvidenceBenchmarkExecutionError,
	EvidenceBenchmarkResult,
	EvidenceBenchmarkRunner,
	GoldEvidenceLabel,
	confusion_matrix,
	derive_gold_status,
	hit_metrics,
	load_benchmark_cases,
	macro_f1,
	mean,
	per_class_f1,
	precision_recall_f1,
	rate,
)
from browser_use.evidence.reranking import RawSemanticEvidenceScore, RawSemanticReranking, SemanticEvidenceReranker
from browser_use.evidence.verification import ClaimVerifier, RawClaimEvidenceAssessment, RawEvidenceAssessment
from browser_use.llm.views import ChatInvokeCompletion

_EVIDENCE_ID_PATTERN = re.compile(r'^evidence_id: (.+)$', re.MULTILINE)

STARS_CLAIM = 'Browser Use has more than 100,000 GitHub stars.'
HIGH_STAR_TEXT = 'Browser Use has 111,799 GitHub stars.'
LOW_STAR_TEXT = 'Browser Use has only 30,000 GitHub stars.'
LANGUAGE_TEXT = 'Browser Use is primarily written in Python.'
DISTRACTOR_TEXT = 'Browser Use has more stars than any other browser use tool on GitHub, reviewers say.'
SECRET = 'secret123'

SEED_DATASET = Path('benchmarks/webevidence/seed_cases.jsonl')


def _node(
	evidence_id: str, text: str, *, url: str = 'https://example.invalid/page', title: str = '', step_number: int = 1
) -> EvidenceNode:
	return EvidenceNode(evidence_id=evidence_id, task_id='task-c1', step_number=step_number, url=url, title=title, text=text)


def _label(evidence_id: str, relation: EvidenceRelation, is_relevant: bool = True) -> GoldEvidenceLabel:
	return GoldEvidenceLabel(evidence_id=evidence_id, relation=relation, is_relevant=is_relevant)


def _case(
	*,
	case_id: str = 'c1',
	claim_text: str = STARS_CLAIM,
	nodes: list[EvidenceNode] | None = None,
	labels: list[GoldEvidenceLabel] | None = None,
	status: VerificationStatus | None = None,
	tags: tuple[str, ...] = (),
) -> EvidenceBenchmarkCase:
	"""A frozen case whose gold status defaults to whatever its gold relations imply."""
	nodes = nodes if nodes is not None else []
	labels = labels if labels is not None else []
	return EvidenceBenchmarkCase(
		case_id=case_id,
		task_id=f'task-{case_id}',
		task=f'Verify this claim: {claim_text}',
		claim=Claim(claim_id=f'claim-{case_id}', order=1, text=claim_text),
		evidence_nodes=nodes,
		gold_labels=labels,
		gold_status=status or derive_gold_status(labels),
		tags=list(tags),
		description='test fixture',
	)


class BenchmarkChatModel:
	"""Answers the reranker and the verifier from tables the test controls.

	``fail_on`` holds the schema name whose call should explode, which is how the strict failure policy is
	taken down one stage at a time.
	"""

	def __init__(
		self,
		*,
		scores: dict[str, float] | None = None,
		relations: dict[str, EvidenceRelation] | None = None,
		default_score: float = 0.9,
		default_relation: EvidenceRelation = EvidenceRelation.INSUFFICIENT,
		fail_on: tuple[str, ...] = (),
	) -> None:
		self.model = 'benchmark-fake-model'
		self.provider = 'fake'
		self.name = 'benchmark-fake-model'
		self.model_name = 'benchmark-fake-model'
		self._verified_api_keys = True
		self.scores = dict(scores or {})
		self.relations = dict(relations or {})
		self.default_score = default_score
		self.default_relation = default_relation
		self.fail_on = set(fail_on)
		self.calls: list[str] = []

	async def ainvoke(self, messages, output_format=None, **kwargs) -> ChatInvokeCompletion:
		schema = getattr(output_format, '__name__', '')
		self.calls.append(schema)
		if schema in self.fail_on:
			raise RuntimeError(f'provider exploded api-key={SECRET} with the prompt attached')

		candidate_ids = _EVIDENCE_ID_PATTERN.findall(messages[-1].text)
		if schema == 'RawSemanticReranking':
			completion = RawSemanticReranking(
				scores=[
					RawSemanticEvidenceScore(
						evidence_id=evidence_id, relevance_score=self.scores.get(evidence_id, self.default_score)
					)
					for evidence_id in candidate_ids
				]
			)
		elif schema == 'RawClaimEvidenceAssessment':
			completion = RawClaimEvidenceAssessment(
				assessments=[
					RawEvidenceAssessment(
						evidence_id=evidence_id,
						relation=self.relations.get(evidence_id, self.default_relation),
						explanation=f'{evidence_id} states something about the claim.',
					)
					for evidence_id in candidate_ids
				]
			)
		else:
			raise AssertionError(f'the benchmark asked for an unexpected schema: {schema!r}')

		return ChatInvokeCompletion(completion=completion, usage=None)


def _runner(**overrides) -> EvidenceBenchmarkRunner:
	components: dict = {'aligner': EvidenceAligner(top_k=5)}
	components.update(overrides)
	return EvidenceBenchmarkRunner(**components)


def _support_case(**kwargs) -> EvidenceBenchmarkCase:
	node = _node('ev-genuine', HIGH_STAR_TEXT, url='https://example.invalid/genuine', title='GitHub')
	return _case(
		nodes=[node],
		labels=[_label('ev-genuine', EvidenceRelation.SUPPORTS)],
		**kwargs,
	)


def _write_dataset(path: Path, lines: list[str]) -> Path:
	path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
	return path


class TestDatasetLoader:
	def test_seed_dataset_loads_and_is_at_least_twelve_cases(self):
		cases = load_benchmark_cases(SEED_DATASET)

		assert len(cases) >= 12
		assert len({case.case_id for case in cases}) == len(cases)

	def test_seed_dataset_covers_the_required_failure_modes(self):
		cases = {case.case_id: case for case in load_benchmark_cases(SEED_DATASET)}

		assert cases['stars-support'].gold_status is VerificationStatus.SUPPORTED
		assert cases['stars-contradiction'].gold_status is VerificationStatus.CONTRADICTED
		assert cases['stars-conflict'].gold_status is VerificationStatus.CONFLICTED
		assert cases['stars-offtopic'].gold_status is VerificationStatus.UNSUPPORTED
		assert cases['mcp-partial'].gold_status is VerificationStatus.PARTIAL
		assert cases['no-evidence'].gold_status is VerificationStatus.NO_EVIDENCE
		# A refutation is retrieval-relevant, and the empty case is the only one with no evidence at all.
		assert cases['stars-contradiction'].gold_relevant_ids == frozenset({'ev-stars-contradiction-a'})
		assert cases['no-evidence'].evidence_nodes == []

	def test_blank_lines_are_skipped_and_line_numbers_survive(self, tmp_path):
		case = _support_case()
		path = _write_dataset(
			tmp_path / 'data.jsonl', [case.model_dump_json(), '', '   ', case.model_dump_json().replace('"c1"', '"c2"')]
		)

		cases = load_benchmark_cases(path)

		assert [entry.case_id for entry in cases] == ['c1', 'c2']

	def test_malformed_line_reports_its_line_number(self, tmp_path):
		path = _write_dataset(tmp_path / 'data.jsonl', [_support_case().model_dump_json(), '{"case_id": "broken"'])

		with pytest.raises(EvidenceBenchmarkError, match='line 2'):
			load_benchmark_cases(path)

	def test_duplicate_case_id_names_both_lines(self, tmp_path):
		path = _write_dataset(tmp_path / 'data.jsonl', [_support_case().model_dump_json(), _support_case().model_dump_json()])

		with pytest.raises(EvidenceBenchmarkError, match='duplicate case_id .c1. on lines 1 and 2'):
			load_benchmark_cases(path)

	def test_empty_dataset_is_refused(self, tmp_path):
		path = tmp_path / 'data.jsonl'
		path.write_text('\n\n', encoding='utf-8')

		with pytest.raises(EvidenceBenchmarkError, match='contains no cases'):
			load_benchmark_cases(path)

	def test_missing_file_is_reported_not_swallowed(self, tmp_path):
		with pytest.raises(EvidenceBenchmarkError, match='Cannot read benchmark dataset'):
			load_benchmark_cases(tmp_path / 'absent.jsonl')

	def test_the_input_file_is_not_modified(self, tmp_path):
		case = _support_case()
		path = _write_dataset(tmp_path / 'data.jsonl', [case.model_dump_json()])
		before = path.read_bytes()

		load_benchmark_cases(path)

		assert path.read_bytes() == before


class TestCaseIntegrity:
	def test_every_evidence_node_needs_a_gold_label(self):
		with pytest.raises(EvidenceBenchmarkError, match='without a gold label'):
			_case(nodes=[_node('ev-a', HIGH_STAR_TEXT)], labels=[])

	def test_a_label_for_unknown_evidence_is_refused(self):
		with pytest.raises(EvidenceBenchmarkError, match='unknown evidence_id'):
			_case(
				nodes=[_node('ev-a', HIGH_STAR_TEXT)],
				labels=[_label('ev-ghost', EvidenceRelation.SUPPORTS)],
			)

	def test_a_duplicated_label_is_refused(self):
		with pytest.raises(EvidenceBenchmarkError, match='labels the same evidence_id twice'):
			_case(
				nodes=[_node('ev-a', HIGH_STAR_TEXT)],
				labels=[
					_label('ev-a', EvidenceRelation.SUPPORTS),
					_label('ev-a', EvidenceRelation.SUPPORTS),
				],
			)

	def test_duplicate_evidence_ids_are_refused(self):
		with pytest.raises(EvidenceBenchmarkError, match='duplicate evidence_id'):
			_case(
				nodes=[_node('ev-a', HIGH_STAR_TEXT), _node('ev-a', LOW_STAR_TEXT, step_number=2)],
				labels=[_label('ev-a', EvidenceRelation.SUPPORTS), _label('ev-a', EvidenceRelation.CONTRADICTS)],
			)

	def test_gold_status_must_follow_the_gold_relations(self):
		with pytest.raises(EvidenceBenchmarkError, match='gold relations imply SUPPORTED'):
			_case(
				nodes=[_node('ev-a', HIGH_STAR_TEXT)],
				labels=[_label('ev-a', EvidenceRelation.SUPPORTS)],
				status=VerificationStatus.CONTRADICTED,
			)

	def test_conflict_gold_cannot_be_written_as_supported(self):
		with pytest.raises(EvidenceBenchmarkError, match='gold relations imply CONFLICTED'):
			_case(
				nodes=[_node('ev-a', HIGH_STAR_TEXT), _node('ev-b', LOW_STAR_TEXT, step_number=2)],
				labels=[
					_label('ev-a', EvidenceRelation.SUPPORTS),
					_label('ev-b', EvidenceRelation.CONTRADICTS),
				],
				status=VerificationStatus.SUPPORTED,
			)

	def test_generated_claim_id_is_refused(self):
		with pytest.raises(EvidenceBenchmarkError, match='explicit claim.claim_id'):
			EvidenceBenchmarkCase(
				case_id='c1',
				task_id='task-c1',
				claim=Claim(order=1, text=STARS_CLAIM),
				gold_status=VerificationStatus.NO_EVIDENCE,
			)

	def test_generated_evidence_id_is_refused(self):
		generated = EvidenceNode(task_id='task-c1', step_number=1, url='https://example.invalid/x', text=HIGH_STAR_TEXT)
		with pytest.raises(EvidenceBenchmarkError, match='generated evidence_id'):
			_case(
				nodes=[generated],
				labels=[_label(generated.evidence_id, EvidenceRelation.SUPPORTS)],
			)

	def test_empty_evidence_must_be_no_evidence(self):
		case = _case(status=VerificationStatus.NO_EVIDENCE)
		assert case.evidence_nodes == []
		assert case.gold_labels == []
		with pytest.raises(EvidenceBenchmarkError, match='gold_status'):
			_case(status=VerificationStatus.UNSUPPORTED)

	def test_relevance_and_relation_are_independent(self):
		"""A refutation is relevant; an off-topic page is not, even though both are correct labels."""
		refuting = _case(
			nodes=[_node('ev-b', LOW_STAR_TEXT)],
			labels=[_label('ev-b', EvidenceRelation.CONTRADICTS, True)],
		)
		off_topic = _case(
			nodes=[_node('ev-c', LANGUAGE_TEXT)],
			labels=[_label('ev-c', EvidenceRelation.INSUFFICIENT, False)],
		)

		assert refuting.gold_relevant_ids == frozenset({'ev-b'})
		assert off_topic.gold_relevant_ids == frozenset()
		assert refuting.gold_status is VerificationStatus.CONTRADICTED
		assert off_topic.gold_status is VerificationStatus.UNSUPPORTED

	def test_claim_set_carries_the_single_fixed_claim(self):
		case = _support_case()

		claim_set = case.claim_set()

		assert claim_set.task_id == case.task_id
		assert [claim.text for claim in claim_set.claims] == [STARS_CLAIM]
		assert claim_set.answer == STARS_CLAIM


class TestGoldStatusMatchesPhaseFive:
	@pytest.mark.parametrize(
		'relations',
		[list(combination) for size in range(1, 5) for combination in combinations(EvidenceRelation, size)],
		ids=lambda relations: '+'.join(relation.value for relation in relations),
	)
	def test_every_non_empty_relation_set_agrees_with_the_verifier(self, relations):
		"""The benchmark must aggregate gold exactly the way the pipeline aggregates predictions."""
		labels = [_label(f'ev-{index}', relation) for index, relation in enumerate(relations, start=1)]

		assert derive_gold_status(labels) is ClaimVerifier._aggregate_status(relations)

	def test_no_labels_at_all_is_no_evidence(self):
		assert derive_gold_status([]) is VerificationStatus.NO_EVIDENCE
		assert ClaimVerifier._aggregate_status([]) is VerificationStatus.NO_EVIDENCE


class TestMetricHelpers:
	def test_hit_metrics_for_a_first_rank_hit(self):
		assert hit_metrics(['ev-a', 'ev-b'], {'ev-a'}) == (True, True, 1.0)

	def test_hit_metrics_when_the_first_rank_is_wrong_but_the_second_is_right(self):
		"""This is the shape MRR exists for: both stages hit@k, only one ranks first."""
		hit_at_1, hit_at_k, rr = hit_metrics(['ev-distractor', 'ev-a'], {'ev-a'})
		assert (hit_at_1, hit_at_k, rr) == (False, True, 0.5)

	def test_hit_metrics_with_nothing_retrieved(self):
		assert hit_metrics([], {'ev-a'}) == (False, False, 0.0)

	def test_hit_metrics_with_an_empty_gold_set_is_unmeasurable_not_wrong(self):
		assert hit_metrics(['ev-a'], set()) == (False, False, 0.0)

	def test_rate_and_mean_are_none_when_there_is_nothing_to_average(self):
		assert rate([]) is None
		assert mean([]) is None
		assert rate([True, False]) == 0.5
		assert mean([0.25, 0.75]) == 0.5

	def test_precision_recall_f1_handles_an_undefined_ratio_as_zero(self):
		assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)
		assert precision_recall_f1(1, 0, 0) == (1.0, 1.0, 1.0)
		precision, recall, f1 = precision_recall_f1(1, 1, 1)
		assert precision == 0.5
		assert recall == 0.5
		assert f1 == 0.5

	def test_per_class_f1_credits_every_class_on_either_side(self):
		pairs = [('SUPPORTS', 'SUPPORTS'), ('CONTRADICTS', 'INSUFFICIENT')]

		scores = per_class_f1(pairs)

		# SUPPORTS: tp 1, fp 0, fn 0 -> 1.0. CONTRADICTS: tp 0, fp 0, fn 1 -> 0.0.
		# INSUFFICIENT: tp 0, fp 1, fn 0 -> 0.0, so a class only ever predicted wrongly still counts.
		assert scores == {'SUPPORTS': 1.0, 'CONTRADICTS': 0.0, 'INSUFFICIENT': 0.0}
		assert macro_f1(pairs) == pytest.approx(1 / 3)

	def test_macro_f1_of_nothing_is_zero(self):
		assert macro_f1([]) == 0.0

	def test_confusion_matrix_is_keyed_gold_then_predicted(self):
		pairs = [('SUPPORTS', 'CONTRADICTS'), ('SUPPORTS', 'CONTRADICTS'), ('INSUFFICIENT', 'INSUFFICIENT')]
		labels = [label.value for label in RELATION_LABELS]

		table = confusion_matrix(pairs, labels)

		assert table['SUPPORTS']['CONTRADICTS'] == 2
		assert table['INSUFFICIENT']['INSUFFICIENT'] == 1
		assert table['CONTRADICTS']['SUPPORTS'] == 0
		assert set(table) == set(labels)
		assert all(set(row) == set(labels) for row in table.values())

	def test_the_label_orders_are_the_two_confusion_matrices(self):
		assert [label.value for label in RELATION_LABELS] == ['SUPPORTS', 'PARTIAL_SUPPORT', 'CONTRADICTS', 'INSUFFICIENT']
		assert [label.value for label in STATUS_LABELS] == [
			'SUPPORTED',
			'PARTIAL',
			'UNSUPPORTED',
			'CONTRADICTED',
			'CONFLICTED',
			'NO_EVIDENCE',
		]


class TestLexicalMode:
	async def test_a_lexical_only_run_reports_nothing_for_the_stages_that_never_ran(self):
		runner = _runner()

		result = await runner.run([_support_case()])

		row = result.cases[0]
		assert runner.mode == 'lexical'
		assert row.semantic_ranked_evidence_ids == []
		assert row.semantic_hit_at_1 is None
		assert row.semantic_hit_at_k is None
		assert row.semantic_reciprocal_rank is None
		assert row.predicted_relations == {}
		assert row.predicted_status is None
		assert row.status_correct is None
		assert row.relation_evaluated_count == 0

		summary = result.summary
		assert summary.semantic_hit_at_1_rate is None
		assert summary.semantic_hit_at_k_rate is None
		assert summary.semantic_mrr is None
		assert summary.relation_accuracy is None
		assert summary.relation_macro_f1 is None
		assert summary.status_accuracy is None
		assert summary.relation_confusion_matrix == {}
		assert summary.status_confusion_matrix == {}

	async def test_unavailable_is_never_reported_as_zero(self):
		"""A missing stage must not look like a stage that scored nothing."""
		result = await _runner().run([_support_case()])

		assert result.summary.lexical_hit_at_1_rate == 1.0
		for unavailable in (
			result.summary.semantic_hit_at_1_rate,
			result.summary.relation_accuracy,
			result.summary.status_accuracy,
		):
			assert unavailable is None
			assert unavailable != 0.0

	async def test_hit_at_one_requires_the_top_candidate_to_be_relevant(self):
		nodes = [
			_node('ev-distractor', DISTRACTOR_TEXT, url='https://example.invalid/d', title='Ranking list', step_number=1),
			_node('ev-genuine', HIGH_STAR_TEXT, url='https://example.invalid/g', title='GitHub', step_number=2),
		]
		case = _case(
			nodes=nodes,
			labels=[
				_label('ev-distractor', EvidenceRelation.INSUFFICIENT, False),
				_label('ev-genuine', EvidenceRelation.SUPPORTS),
			],
		)

		row = (await _runner().run([case])).cases[0]

		# Phase 4A ranks the word-heavy page first, so hit@1 fails while hit@k still holds.
		assert row.lexical_ranked_evidence_ids == ['ev-distractor', 'ev-genuine']
		assert (row.lexical_hit_at_1, row.lexical_hit_at_k, row.lexical_reciprocal_rank) == (False, True, 0.5)
		assert not row.lexical_miss

	async def test_a_page_the_aligner_cannot_recall_is_a_miss_not_a_zero_score(self):
		"""No shared token means no candidate, and a reranker cannot see what never arrived."""
		recalled = _node('ev-recalled', HIGH_STAR_TEXT, url='https://example.invalid/r', title='GitHub')
		foreign = _node('ev-foreign', '提供每晚构建。', title='更新日志', step_number=2)
		case = _case(
			nodes=[recalled, foreign],
			labels=[_label('ev-recalled', EvidenceRelation.SUPPORTS), _label('ev-foreign', EvidenceRelation.SUPPORTS)],
		)

		result = await _runner().run([case])
		row = result.cases[0]

		assert row.lexical_ranked_evidence_ids == ['ev-recalled']
		assert row.lexical_hit_at_1 and row.retrieval_scored
		assert row.lexical_miss
		assert result.summary.lexical_miss_case_ids == ['c1']
		# The miss is reported even though everything measured came out right.
		assert row.notes[0].startswith('1 gold relevant page(s) never entered the lexical candidate set')

	async def test_cases_without_any_relevant_gold_leave_the_denominator(self):
		off_topic = _case(
			case_id='off-topic',
			nodes=[_node('ev-a', LANGUAGE_TEXT)],
			labels=[_label('ev-a', EvidenceRelation.INSUFFICIENT, False)],
		)

		result = await _runner().run([_support_case(), off_topic])

		assert result.summary.case_count == 2
		assert result.summary.retrieval_case_count == 1
		assert result.summary.lexical_hit_at_1_rate == 1.0
		assert result.cases[1].notes == ['no gold relevant evidence, so this case is excluded from the retrieval aggregates']

	async def test_an_empty_candidate_set_is_recorded_as_a_miss(self):
		foreign = _node('ev-foreign', '提供每晚构建。', title='更新日志')
		case = _case(nodes=[foreign], labels=[_label('ev-foreign', EvidenceRelation.SUPPORTS)])

		row = (await _runner().run([case])).cases[0]

		assert row.lexical_ranked_evidence_ids == []
		assert (row.lexical_hit_at_1, row.lexical_hit_at_k, row.lexical_reciprocal_rank) == (False, False, 0.0)
		assert row.lexical_miss and row.retrieval_scored

	def test_a_verifier_needs_a_reranker(self):
		from browser_use.evidence import ClaimVerifier

		with pytest.raises(EvidenceBenchmarkError, match='a verifier needs a reranker'):
			EvidenceBenchmarkRunner(aligner=EvidenceAligner(), verifier=ClaimVerifier(BenchmarkChatModel()))

	async def test_the_seed_dataset_lexical_numbers_are_stable(self):
		"""The seed run is deterministic offline, and it reports its one designed miss."""
		cases = load_benchmark_cases(SEED_DATASET)

		first = await _runner().run(cases)
		second = await _runner().run(cases)

		assert first.model_dump() == second.model_dump()
		assert first.summary.case_count == len(cases)
		assert first.summary.lexical_miss_case_ids == ['cross-lingual']
		assert first.summary.relation_accuracy is None
		assert [row.case_id for row in first.cases] == [case.case_id for case in cases]


def _distractor_case(**kwargs) -> EvidenceBenchmarkCase:
	"""One case where the word-heavy page beats the page that actually states the count."""
	nodes = [
		_node('ev-distractor', DISTRACTOR_TEXT, url='https://example.invalid/d', title='Ranking list', step_number=1),
		_node('ev-genuine', HIGH_STAR_TEXT, url='https://example.invalid/g', title='GitHub', step_number=2),
	]
	return _case(
		nodes=nodes,
		labels=[
			_label('ev-distractor', EvidenceRelation.INSUFFICIENT, False),
			_label('ev-genuine', EvidenceRelation.SUPPORTS),
		],
		**kwargs,
	)


class TestSemanticMode:
	async def test_rescoring_can_lift_the_relevant_page_to_first_without_changing_recall(self):
		"""The value the reranker is supposed to add, and the ceiling it cannot get past."""
		model = BenchmarkChatModel(scores={'ev-genuine': 1.0, 'ev-distractor': 0.0})
		case = _distractor_case()

		lexical_only = await _runner().run([case])
		rescored = await _runner(reranker=SemanticEvidenceReranker(model)).run([case])

		lexical_row, semantic_row = lexical_only.cases[0], rescored.cases[0]
		assert lexical_row.lexical_ranked_evidence_ids == ['ev-distractor', 'ev-genuine']
		assert semantic_row.semantic_ranked_evidence_ids == ['ev-genuine', 'ev-distractor']
		assert (lexical_row.lexical_hit_at_1, semantic_row.semantic_hit_at_1) == (False, True)
		assert (lexical_row.lexical_reciprocal_rank, semantic_row.semantic_reciprocal_rank) == (0.5, 1.0)
		# The candidate set is the same, which is the whole point of measuring Hit@K separately.
		assert semantic_row.lexical_hit_at_k and semantic_row.semantic_hit_at_k
		assert lexical_row.lexical_hit_at_k == semantic_row.semantic_hit_at_k
		assert rescored.summary.semantic_mrr == 1.0
		assert rescored.summary.lexical_mrr == 0.5

	async def test_the_reranker_cannot_recover_a_page_the_aligner_never_recalled(self):
		recalled = _node('ev-recalled', HIGH_STAR_TEXT, url='https://example.invalid/r', title='GitHub')
		foreign = _node('ev-foreign', '提供每晚构建。', title='更新日志', step_number=2)
		case = _case(
			nodes=[recalled, foreign],
			labels=[_label('ev-recalled', EvidenceRelation.SUPPORTS), _label('ev-foreign', EvidenceRelation.SUPPORTS)],
		)
		model = BenchmarkChatModel()

		result = await _runner(reranker=SemanticEvidenceReranker(model)).run([case])
		row = result.cases[0]

		# Only one candidate was ever offered to the model, so only one came back ranked.
		assert row.semantic_ranked_evidence_ids == ['ev-recalled']
		# Hit@K cannot separate the two stages here: the other gold page is relevant to the same claim,
		# so both lists contain at least one relevant page even though one of the two is invisible.
		assert row.lexical_hit_at_k and row.semantic_hit_at_k
		assert result.summary.lexical_miss_case_ids == ['c1']
		assert model.calls == ['RawSemanticReranking']

	async def test_a_claim_with_no_candidates_costs_no_model_call(self):
		case = _case(status=VerificationStatus.NO_EVIDENCE)
		model = BenchmarkChatModel()

		result = await _runner(reranker=SemanticEvidenceReranker(model)).run([case])

		assert model.calls == []
		assert result.cases[0].semantic_ranked_evidence_ids == []
		assert result.cases[0].semantic_hit_at_1 is False

	async def test_semantic_mode_still_measures_no_verification(self):
		model = BenchmarkChatModel()

		result = await _runner(reranker=SemanticEvidenceReranker(model)).run([_support_case()])

		assert model.calls == ['RawSemanticReranking']
		assert result.summary.relation_accuracy is None
		assert result.summary.status_accuracy is None
		assert result.cases[0].predicted_status is None
		assert result.cases[0].semantic_hit_at_1 is True


class TestFullMode:
	@pytest.fixture
	def two_page_case(self):
		nodes = [
			_node('ev-a', HIGH_STAR_TEXT, url='https://example.invalid/a', title='GitHub'),
			_node('ev-b', LOW_STAR_TEXT, url='https://example.invalid/b', title='Old post', step_number=2),
		]
		return _case(
			case_id='conflict',
			nodes=nodes,
			labels=[_label('ev-a', EvidenceRelation.SUPPORTS), _label('ev-b', EvidenceRelation.CONTRADICTS)],
		)

	@pytest.fixture
	def mixed_case(self):
		"""Gold says one page refutes the claim and the other page only shares the topic."""
		nodes = [
			_node('ev-b', LOW_STAR_TEXT, url='https://example.invalid/b', title='Old post'),
			_node('ev-c', LANGUAGE_TEXT, url='https://example.invalid/c', title='Docs', step_number=2),
		]
		return _case(
			case_id='mixed',
			nodes=nodes,
			labels=[_label('ev-b', EvidenceRelation.CONTRADICTS), _label('ev-c', EvidenceRelation.INSUFFICIENT, False)],
		)

	async def test_correct_labels_score_full_marks_and_agree_with_gold(self, two_page_case):
		model = BenchmarkChatModel(relations={'ev-a': EvidenceRelation.SUPPORTS, 'ev-b': EvidenceRelation.CONTRADICTS})

		result = await _runner(reranker=SemanticEvidenceReranker(model), verifier=ClaimVerifier(model)).run([two_page_case])

		row = result.cases[0]
		assert row.predicted_status is VerificationStatus.CONFLICTED
		assert row.status_correct
		assert (row.relation_correct_count, row.relation_evaluated_count) == (2, 2)
		assert result.summary.relation_accuracy == 1.0
		assert result.summary.relation_macro_f1 == 1.0
		assert result.summary.status_accuracy == 1.0
		assert result.summary.status_error_case_ids == []
		assert result.summary.relation_error_case_ids == []

	async def test_a_wrong_label_is_counted_and_named(self, two_page_case, mixed_case):
		"""Three of four labels are right, and the one that is wrong names its case."""
		model = BenchmarkChatModel(
			relations={
				'ev-a': EvidenceRelation.SUPPORTS,
				'ev-b': EvidenceRelation.CONTRADICTS,
				'ev-c': EvidenceRelation.CONTRADICTS,
			}
		)

		result = await _runner(reranker=SemanticEvidenceReranker(model), verifier=ClaimVerifier(model)).run(
			[two_page_case, mixed_case]
		)

		assert result.summary.relation_evaluated_count == 4
		assert result.summary.relation_accuracy == pytest.approx(0.75)
		# The off-topic page was read as a refutation, and the case id is what makes that findable.
		assert result.summary.relation_error_case_ids == ['mixed']
		assert [row.relation_correct_count for row in result.cases] == [2, 1]
		# The wrong label did not flip either claim status, because a genuine refutation is present in
		# both cases: status accuracy can stay perfect while a relation is still wrong.
		assert result.summary.status_error_case_ids == []
		assert [row.predicted_status for row in result.cases] == [
			VerificationStatus.CONFLICTED,
			VerificationStatus.CONTRADICTED,
		]

	async def test_one_wrong_label_on_off_topic_evidence_flips_the_status(self, two_page_case):
		"""The dangerous shape: an off-topic page read as a refutation turns SUPPORTED into CONFLICTED."""
		off_topic = _case(
			case_id='off-topic',
			nodes=[
				_node('ev-a', HIGH_STAR_TEXT, url='https://example.invalid/a', title='GitHub'),
				_node('ev-c', LANGUAGE_TEXT, url='https://example.invalid/c', title='Docs', step_number=2),
			],
			labels=[_label('ev-a', EvidenceRelation.SUPPORTS), _label('ev-c', EvidenceRelation.INSUFFICIENT, False)],
		)
		model = BenchmarkChatModel(relations={'ev-a': EvidenceRelation.SUPPORTS, 'ev-c': EvidenceRelation.CONTRADICTS})

		result = await _runner(reranker=SemanticEvidenceReranker(model), verifier=ClaimVerifier(model)).run([off_topic])

		assert result.summary.status_error_case_ids == ['off-topic']
		assert result.cases[0].gold_status is VerificationStatus.SUPPORTED
		assert result.cases[0].predicted_status is VerificationStatus.CONFLICTED
		# Retrieval said nothing wrong here: the relevant page was found and ranked first.
		assert not result.cases[0].lexical_miss
		assert result.summary.lexical_miss_case_ids == []

	async def test_the_relation_matrix_counts_both_directions(self, two_page_case, mixed_case):
		"""Gold INSUFFICIENT predicted as CONTRADICTS must show up as a false refutation."""
		model = BenchmarkChatModel(
			relations={
				'ev-a': EvidenceRelation.SUPPORTS,
				'ev-b': EvidenceRelation.CONTRADICTS,
				'ev-c': EvidenceRelation.CONTRADICTS,
			}
		)

		result = await _runner(reranker=SemanticEvidenceReranker(model), verifier=ClaimVerifier(model)).run(
			[two_page_case, mixed_case]
		)

		matrix = result.summary.relation_confusion_matrix
		assert matrix['INSUFFICIENT']['CONTRADICTS'] == 1
		assert matrix['CONTRADICTS']['CONTRADICTS'] == 2
		assert matrix['SUPPORTS']['SUPPORTS'] == 1
		assert set(matrix) == {label.value for label in RELATION_LABELS}
		assert sum(sum(row.values()) for row in matrix.values()) == 4

	async def test_the_status_matrix_is_six_by_six_and_totals_the_scored_cases(self, two_page_case):
		model = BenchmarkChatModel(relations={'ev-a': EvidenceRelation.SUPPORTS, 'ev-b': EvidenceRelation.CONTRADICTS})

		result = await _runner(reranker=SemanticEvidenceReranker(model), verifier=ClaimVerifier(model)).run([two_page_case])

		matrix = result.summary.status_confusion_matrix
		assert set(matrix) == {label.value for label in STATUS_LABELS}
		assert all(set(row) == {label.value for label in STATUS_LABELS} for row in matrix.values())
		assert matrix['CONFLICTED']['CONFLICTED'] == 1
		assert result.summary.status_scored_case_count == 1

	async def test_macro_f1_punishes_a_class_the_system_never_predicts(self, two_page_case):
		"""Accuracy alone would look fine while the refutation class was simply never used."""
		nodes = [
			_node('ev-a', HIGH_STAR_TEXT, url='https://example.invalid/a', title='GitHub'),
			_node('ev-b', LOW_STAR_TEXT, url='https://example.invalid/b', title='Old post', step_number=2),
		]
		case = _case(
			nodes=nodes,
			labels=[_label('ev-a', EvidenceRelation.SUPPORTS), _label('ev-b', EvidenceRelation.CONTRADICTS)],
		)
		model = BenchmarkChatModel(relations={'ev-a': EvidenceRelation.SUPPORTS, 'ev-b': EvidenceRelation.SUPPORTS})

		result = await _runner(reranker=SemanticEvidenceReranker(model), verifier=ClaimVerifier(model)).run([case])

		accuracy = result.summary.relation_accuracy
		macro = result.summary.relation_macro_f1
		assert accuracy == 0.5
		# Only SUPPORTS and CONTRADICTIS appear in the pairs, so macro averages two classes: SUPPORTS
		# perfect on precision but half on recall, and CONTRADICTS never predicted at all.
		assert macro < accuracy or macro == pytest.approx(1 / 3)
		assert macro == pytest.approx((1.0 + 0.0) / 3)
		assert per_class_f1([('SUPPORTS', 'SUPPORTS'), ('CONTRADICTS', 'SUPPORTS')]) == {'CONTRADICTS': 0.0, 'SUPPORTS': 2 / 3}

	async def test_nothing_is_assessed_when_no_candidate_arrived(self):
		case = _case(
			nodes=[_node('ev-foreign', '提供每晚构建。', title='更新日志')],
			labels=[_label('ev-foreign', EvidenceRelation.SUPPORTS)],
		)
		model = BenchmarkChatModel()

		result = await _runner(reranker=SemanticEvidenceReranker(model), verifier=ClaimVerifier(model)).run([case])
		row = result.cases[0]

		assert row.predicted_status is VerificationStatus.NO_EVIDENCE
		assert row.predicted_relations == {}
		assert row.relation_evaluated_count == 0
		assert row.status_correct is False
		assert row.notes[0].startswith('1 gold relevant page(s) never entered the lexical candidate set')
		# With no candidates there is nothing to rescore and nothing to verify: two stages, zero calls.
		assert model.calls == []

	def test_modes_are_reported_from_injection_not_from_a_flag(self):
		model = BenchmarkChatModel()

		assert _runner().mode == 'lexical'
		assert _runner(reranker=SemanticEvidenceReranker(model)).mode == 'semantic'
		assert _runner(reranker=SemanticEvidenceReranker(model), verifier=ClaimVerifier(model)).mode == 'full'

	async def test_a_miss_is_reported_even_when_the_status_comes_out_right(self):
		"""Spec 30: a retrieval ceiling must stay visible even when the conclusion happens to be right."""
		recalled = _node('ev-recalled', HIGH_STAR_TEXT, url='https://example.invalid/r', title='GitHub')
		foreign = _node('ev-foreign', '提供每晚构建。', title='更新日志', step_number=2)
		case = _case(
			nodes=[recalled, foreign],
			labels=[_label('ev-recalled', EvidenceRelation.SUPPORTS), _label('ev-foreign', EvidenceRelation.SUPPORTS)],
		)
		model = BenchmarkChatModel(relations={'ev-recalled': EvidenceRelation.SUPPORTS})

		result = await _runner(reranker=SemanticEvidenceReranker(model), verifier=ClaimVerifier(model)).run([case])
		row = result.cases[0]

		assert row.predicted_status is VerificationStatus.SUPPORTED
		assert row.status_correct and row.relation_correct_count == 1
		assert row.lexical_miss
		assert result.summary.lexical_miss_case_ids == ['c1']
		assert result.summary.status_error_case_ids == []


class TestStrictFailurePolicy:
	async def test_a_reranker_failure_aborts_with_case_and_stage(self):
		model = BenchmarkChatModel(fail_on=('RawSemanticReranking',))

		with pytest.raises(EvidenceBenchmarkExecutionError) as excinfo:
			await _runner(reranker=SemanticEvidenceReranker(model)).run([_support_case()])

		error = excinfo.value
		assert error.case_id == 'c1'
		assert error.stage is BenchmarkStage.SEMANTIC_RERANKING
		assert 'SEMANTIC_RERANKING' in str(error)
		assert isinstance(error.__cause__, RuntimeError)

	async def test_a_verifier_failure_aborts_with_the_verification_stage(self):
		model = BenchmarkChatModel(fail_on=('RawClaimEvidenceAssessment',))

		with pytest.raises(EvidenceBenchmarkExecutionError) as excinfo:
			await _runner(reranker=SemanticEvidenceReranker(model), verifier=ClaimVerifier(model)).run([_support_case()])

		assert excinfo.value.stage is BenchmarkStage.VERIFICATION

	async def test_an_outage_is_never_scored_as_a_wrong_answer(self):
		"""The failure has to travel as an exception, not as a zero in the accuracy column."""
		model = BenchmarkChatModel(fail_on=('RawSemanticReranking',))

		with pytest.raises(EvidenceBenchmarkExecutionError):
			await _runner(reranker=SemanticEvidenceReranker(model), verifier=ClaimVerifier(model)).run([_support_case()])

	async def test_the_provider_message_stays_out_of_the_whole_exception_chain(self):
		model = BenchmarkChatModel(fail_on=('RawSemanticReranking',))

		with pytest.raises(EvidenceBenchmarkExecutionError) as excinfo:
			await _runner(reranker=SemanticEvidenceReranker(model)).run([_support_case()])

		error = excinfo.value
		stage_error = error.__cause__
		# Nothing that a log handler prints by default may carry the secret: only the deepest link, the
		# raw provider exception itself, still holds it, and that is the caller's own text to keep.
		assert SECRET not in str(error)
		assert SECRET not in str(stage_error)
		assert HIGH_STAR_TEXT not in str(error)
		assert STARS_CLAIM not in str(error)
		assert type(stage_error).__name__ == 'EvidenceRerankingError'
		assert isinstance(stage_error.__cause__, RuntimeError)
		assert SECRET in str(stage_error.__cause__)


class TestDeterminismPurityAndSerialisation:
	async def test_the_input_cases_are_not_mutated(self):
		cases = [_support_case(), _distractor_case(case_id='c2')]
		before = deepcopy(cases)
		model = BenchmarkChatModel()

		await _runner(reranker=SemanticEvidenceReranker(model), verifier=ClaimVerifier(model)).run(cases)

		assert cases == before

	async def test_the_same_fake_model_produces_the_same_result(self):
		cases = [_support_case(), _distractor_case(case_id='c2')]

		first = await _runner(
			reranker=SemanticEvidenceReranker(BenchmarkChatModel()), verifier=ClaimVerifier(BenchmarkChatModel())
		).run(cases)
		second = await _runner(
			reranker=SemanticEvidenceReranker(BenchmarkChatModel()), verifier=ClaimVerifier(BenchmarkChatModel())
		).run(cases)

		assert first.model_dump_json() == second.model_dump_json()

	def test_results_round_trip_through_json(self):
		result = EvidenceBenchmarkResult(
			summary=BenchmarkSummary(case_count=1, lexical_hit_at_1_rate=1.0),
			cases=[
				BenchmarkRunCaseResult(
					case_id='c1',
					gold_status=VerificationStatus.SUPPORTED,
					gold_relations={'ev-a': EvidenceRelation.SUPPORTS},
					lexical_ranked_evidence_ids=['ev-a'],
				)
			],
		)

		assert EvidenceBenchmarkResult.model_validate_json(result.model_dump_json()) == result

	def test_a_result_holds_no_timestamp_or_run_metadata(self):
		assert set(EvidenceBenchmarkResult.model_fields) == {'summary', 'cases'}
		assert set(BenchmarkSummary.model_fields) >= {
			'case_count',
			'retrieval_case_count',
			'lexical_hit_at_1_rate',
			'lexical_hit_at_k_rate',
			'lexical_mrr',
			'semantic_hit_at_1_rate',
			'semantic_hit_at_k_rate',
			'semantic_mrr',
			'relation_accuracy',
			'relation_macro_f1',
			'status_accuracy',
			'relation_confusion_matrix',
			'status_confusion_matrix',
			'lexical_miss_case_ids',
			'status_error_case_ids',
		}

	def test_a_row_rejects_counters_that_disagree_with_its_lists(self):
		with pytest.raises(Exception, match='claims 2 correct'):
			BenchmarkRunCaseResult(case_id='c1', gold_status=VerificationStatus.SUPPORTED, relation_correct_count=2)
		with pytest.raises(Exception, match='predictions but counts'):
			BenchmarkRunCaseResult(
				case_id='c1',
				gold_status=VerificationStatus.SUPPORTED,
				predicted_relations={'ev-a': EvidenceRelation.SUPPORTS},
				relation_evaluated_count=5,
			)
		with pytest.raises(Exception, match='without any reranked candidate set'):
			BenchmarkRunCaseResult(
				case_id='c1',
				gold_status=VerificationStatus.SUPPORTED,
				predicted_status=VerificationStatus.SUPPORTED,
			)


class TestCliGuards:
	def test_the_cli_refuses_to_spend_quota_unasked(self, tmp_path):
		cli = _load_cli()

		for mode in ('semantic', 'full'):
			with pytest.raises(EvidenceBenchmarkError, match=f'{mode} requires --live-llm'):
				cli.build_runner(mode, top_k=5, live=False)

	def test_lexical_mode_needs_no_key_and_no_opt_in(self, monkeypatch, tmp_path):
		cli = _load_cli()
		monkeypatch.delenv(cli.API_KEY_ENV, raising=False)

		runner = cli.build_runner('lexical', top_k=3, live=False)

		assert runner.mode == 'lexical'

	def test_a_live_mode_without_a_key_is_refused_by_name(self, monkeypatch):
		cli = _load_cli()
		# An empty value survives load_dotenv, so this is the "key never configured" state.
		monkeypatch.setenv(cli.API_KEY_ENV, '')

		with pytest.raises(EvidenceBenchmarkError, match=f'{cli.API_KEY_ENV} is not set'):
			cli.build_runner('semantic', top_k=5, live=True)

	def test_the_refusal_message_is_safe_to_print(self, monkeypatch):
		cli = _load_cli()
		monkeypatch.setenv(cli.API_KEY_ENV, SECRET)

		with pytest.raises(EvidenceBenchmarkError) as excinfo:
			cli.build_runner('full', top_k=5, live=False)

		assert SECRET not in str(excinfo.value)


def _load_cli():
	"""Import the benchmark CLI by path, so its guards are covered without spawning a process."""
	import importlib.util

	script = Path('scripts/run_webevidence_benchmark.py').resolve()
	spec = importlib.util.spec_from_file_location('webevidence_benchmark_cli', script)
	assert spec and spec.loader
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module
