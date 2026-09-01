"""Offline, frozen-evidence benchmark for retrieval and verification quality.

This harness answers narrow questions against hand written gold labels: did Phase 4A recall the evidence
that matters, did Phase 4B rank it higher, did Phase 5 label each page correctly, and did the Python
aggregation reach the right claim status. Every case pins one atomic claim and a fixed set of
``EvidenceNode`` objects, so a score that moves means the pipeline moved, not that a web page changed or
that a model split the claim differently this time.

One architectural fact shapes the design. ``SemanticEvidenceReranker`` rescores only what
``EvidenceAligner`` already recalled, so a page that scored zero lexically never becomes a candidate and
no semantic score can recover it. The harness keeps that visible through ``lexical_miss_case_ids``, even
when the final status happens to come out right, and nothing here widens the reranker input or retunes
the aligner to flatter a number: a miss is the result, not a bug in this file.

Read live scores as observations rather than constants. With a real model, the same case can come back
``SUPPORTED`` on one run and ``PARTIAL`` on the next even at ``temperature=0.0``, because server-side
batching, logit rounding and sampling are not guaranteed reproducible. Everything here is deterministic
given its inputs, so the fake-model unit benchmark is exactly repeatable while a live run is best
reported as a range over repeats.
"""

from collections.abc import Iterable, Sequence
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator

from browser_use.evidence.alignment import AlignmentResult, EvidenceAligner
from browser_use.evidence.claims import Claim, ClaimSet, NonBlankString
from browser_use.evidence.models import EvidenceNode
from browser_use.evidence.reranking import RerankingResult, SemanticEvidenceReranker
from browser_use.evidence.verification import ClaimVerifier, EvidenceRelation, VerificationResult, VerificationStatus

# Fixed label orders for the confusion matrices, so two reports are comparable cell by cell.
RELATION_LABELS: tuple[EvidenceRelation, ...] = (
	EvidenceRelation.SUPPORTS,
	EvidenceRelation.PARTIAL_SUPPORT,
	EvidenceRelation.CONTRADICTS,
	EvidenceRelation.INSUFFICIENT,
)
STATUS_LABELS: tuple[VerificationStatus, ...] = (
	VerificationStatus.SUPPORTED,
	VerificationStatus.PARTIAL,
	VerificationStatus.UNSUPPORTED,
	VerificationStatus.CONTRADICTED,
	VerificationStatus.CONFLICTED,
	VerificationStatus.NO_EVIDENCE,
)


class EvidenceBenchmarkError(RuntimeError):
	"""Raised when a dataset or a case cannot be trusted.

	A benchmark number is worth only as much as the gold labels behind it, so a case whose labels
	contradict one another, or whose identifiers do not line up, is refused rather than quietly scored.
	"""


class BenchmarkStage(str, Enum):
	"""Which measured stage was running when a case stopped producing a result."""

	LEXICAL_ALIGNMENT = 'LEXICAL_ALIGNMENT'
	SEMANTIC_RERANKING = 'SEMANTIC_RERANKING'
	VERIFICATION = 'VERIFICATION'


class EvidenceBenchmarkExecutionError(EvidenceBenchmarkError):
	"""Raised when a stage fails on a case, which is not the same thing as a wrong prediction.

	``case_id`` and ``stage`` say which measurement was lost, and ``__cause__`` keeps the original
	exception. The message carries the exception type only, because a provider error can echo its own
	request and that request contains scraped page content. Phase 9A stays strict on purpose: scoring an
	outage as a wrong answer would blend a service failure into an accuracy figure.
	"""

	def __init__(self, message: str, *, case_id: str, stage: BenchmarkStage) -> None:
		super().__init__(message)
		self.case_id = case_id
		self.stage = stage


class GoldEvidenceLabel(BaseModel):
	"""The human answer for one evidence node of one claim.

	``is_relevant`` is a retrieval question: does this page speak to the fact the claim asserts?
	``relation`` is a verification question: what does it say about that fact? The two are separate because
	a page that refutes the claim is exactly the evidence a good retriever should surface.
	"""

	evidence_id: str = Field(description='EvidenceNode.evidence_id this label belongs to')
	relation: EvidenceRelation = Field(description='Gold verification relation for this evidence')
	is_relevant: bool = Field(description='Whether retrieval should surface this evidence for the claim')


class EvidenceBenchmarkCase(BaseModel):
	"""One frozen measurement: a single atomic claim, fixed evidence, and gold labels for all of it."""

	case_id: NonBlankString = Field(description='Stable identifier, unique within a dataset')
	task_id: NonBlankString = Field(description='Task id the fixture pretends the evidence came from')
	task: str = Field(default='', description='Original task text, used only to build the ClaimSet')
	claim: Claim = Field(description='The fixed atomic claim; its id is never regenerated here')
	evidence_nodes: list[EvidenceNode] = Field(default_factory=list, description='Frozen evidence, in capture order')
	gold_labels: list[GoldEvidenceLabel] = Field(default_factory=list, description='Exactly one gold label per evidence node')
	gold_status: VerificationStatus = Field(description='Claim level gold, which must agree with the evidence gold')
	tags: list[str] = Field(default_factory=list, description='Free-form grouping, e.g. paraphrase or numeric')
	description: str = Field(default='', description='What this case is meant to probe')

	@model_validator(mode='after')
	def _check_fixture(self) -> 'EvidenceBenchmarkCase':
		"""Refuse a fixture that could not be reproduced: a regenerated id, a stray label, a missing label."""
		if 'claim_id' not in self.claim.model_fields_set:
			raise EvidenceBenchmarkError(
				f'Benchmark case {self.case_id!r} needs an explicit claim.claim_id; generated ids make scores unreproducible'
			)

		evidence_ids = [node.evidence_id for node in self.evidence_nodes]
		if len(set(evidence_ids)) != len(evidence_ids):
			raise EvidenceBenchmarkError(f'Benchmark case {self.case_id!r} contains duplicate evidence_id')
		generated = [node.evidence_id for node in self.evidence_nodes if 'evidence_id' not in node.model_fields_set]
		if generated:
			raise EvidenceBenchmarkError(
				f'Benchmark case {self.case_id!r} has {len(generated)} evidence node(s) with a generated evidence_id, first: {generated[0]!r}'
			)

		label_ids = [label.evidence_id for label in self.gold_labels]
		if len(set(label_ids)) != len(label_ids):
			raise EvidenceBenchmarkError(f'Benchmark case {self.case_id!r} labels the same evidence_id twice')
		unknown = sorted(set(label_ids) - set(evidence_ids))
		if unknown:
			raise EvidenceBenchmarkError(
				f'Benchmark case {self.case_id!r} labels {len(unknown)} unknown evidence_id(s), first: {unknown[0]!r}'
			)
		missing = sorted(set(evidence_ids) - set(label_ids))
		if missing:
			raise EvidenceBenchmarkError(
				f'Benchmark case {self.case_id!r} has {len(missing)} evidence node(s) without a gold label, first: {missing[0]!r}'
			)

		derived = derive_gold_status(self.gold_labels)
		if derived is not self.gold_status:
			raise EvidenceBenchmarkError(
				f'Benchmark case {self.case_id!r} gold_status is {self.gold_status.value} but its gold relations imply {derived.value}'
			)
		return self

	@property
	def gold_relevant_ids(self) -> frozenset[str]:
		"""Every page a good retriever should have surfaced, refutations included."""
		return frozenset(label.evidence_id for label in self.gold_labels if label.is_relevant)

	@property
	def gold_relation_by_id(self) -> dict[str, EvidenceRelation]:
		return {label.evidence_id: label.relation for label in self.gold_labels}

	def claim_set(self) -> ClaimSet:
		"""The one-claim ``ClaimSet`` the measured stages expect, built from the frozen fixture."""
		return ClaimSet(task_id=self.task_id, task=self.task, answer=self.claim.text, claims=[self.claim])


def derive_gold_status(gold_labels: Iterable[GoldEvidenceLabel]) -> VerificationStatus:
	"""Aggregate gold relations with the same deterministic rule Phase 5 applies in Python.

	The order matters and is deliberately identical to ``ClaimVerifier``: a refutation next to any support
	is a conflict, support outranks insufficiency, partial support needs no refutation to count, and a set
	of nothing but insufficiency is unsupported rather than false. A test binds the two implementations
	together over every combination of relations, so neither can drift from the other unnoticed.
	"""
	relations = {label.relation for label in gold_labels}
	if not relations:
		return VerificationStatus.NO_EVIDENCE

	supports = EvidenceRelation.SUPPORTS in relations
	partials = EvidenceRelation.PARTIAL_SUPPORT in relations
	contradicts = EvidenceRelation.CONTRADICTS in relations
	if contradicts and (supports or partials):
		return VerificationStatus.CONFLICTED
	if supports:
		return VerificationStatus.SUPPORTED
	if partials:
		return VerificationStatus.PARTIAL
	if contradicts:
		return VerificationStatus.CONTRADICTED
	return VerificationStatus.UNSUPPORTED


def load_benchmark_cases(path: Path | str) -> list[EvidenceBenchmarkCase]:
	"""Read a JSONL dataset of benchmark cases, one case per line, in file order.

	Blank lines are skipped, since a hand edited file usually ends with one. Anything else fails the load
	with its line number: a case quietly dropped would change every denominator, and a benchmark whose
	denominators are uncertain measures nothing.

	Raises:
		EvidenceBenchmarkError: unreadable file, a line that is not JSON or not a valid case, a duplicate
			``case_id``, or a case whose own gold labels disagree with each other.
	"""
	dataset_path = Path(path)
	try:
		text = dataset_path.read_text(encoding='utf-8')
	except OSError as e:
		raise EvidenceBenchmarkError(f'Cannot read benchmark dataset {dataset_path}: {type(e).__name__}') from e

	cases: list[EvidenceBenchmarkCase] = []
	line_of_id: dict[str, int] = {}
	for line_number, line in enumerate(text.splitlines(), start=1):
		if not line.strip():
			continue
		try:
			case = EvidenceBenchmarkCase.model_validate_json(line)
		except ValidationError as e:
			raise EvidenceBenchmarkError(f'Benchmark dataset line {line_number} is not a valid case: {type(e).__name__}') from e

		if case.case_id in line_of_id:
			raise EvidenceBenchmarkError(
				f'Benchmark dataset has duplicate case_id {case.case_id!r} on lines {line_of_id[case.case_id]} and {line_number}'
			)
		line_of_id[case.case_id] = line_number
		cases.append(case)

	if not cases:
		raise EvidenceBenchmarkError(f'Benchmark dataset {dataset_path} contains no cases')
	return cases


def hit_metrics(ranked_ids: Sequence[str], gold_relevant: Iterable[str]) -> tuple[bool, bool, float]:
	"""``(hit@1, hit@k, reciprocal rank)`` for one ranked list.

	``k`` is the length of the list handed in, which for Phase 4A is already the aligner's Top-K output and
	for Phase 4B is that same candidate set rescored. So ``hit@k`` reads as "a gold relevant page reached
	the candidate set at all", and both stages share that ceiling by construction.

	An empty gold set makes a case unmeasurable rather than wrong, so every value is falsy and the summary
	leaves the case out of the retrieval denominators. Counting it as a miss would punish the pipeline for
	failing to retrieve a page nobody labelled as worth finding.
	"""
	relevant = set(gold_relevant)
	if not relevant:
		return False, False, 0.0

	hit_at_1 = bool(ranked_ids) and ranked_ids[0] in relevant
	hit_at_k = any(evidence_id in relevant for evidence_id in ranked_ids)
	reciprocal_rank = 0.0
	for position, evidence_id in enumerate(ranked_ids, start=1):
		if evidence_id in relevant:
			reciprocal_rank = 1.0 / position
			break
	return hit_at_1, hit_at_k, reciprocal_rank


def rate(values: Sequence[bool]) -> float | None:
	"""Share of ``True`` values, or ``None`` when there is nothing to average."""
	if not values:
		return None
	return sum(1 for value in values if value) / len(values)


def mean(values: Sequence[float]) -> float | None:
	"""Arithmetic mean, or ``None`` for an empty list, which the summary reports as unavailable."""
	if not values:
		return None
	return sum(values) / len(values)


def precision_recall_f1(true_positive: int, false_positive: int, false_negative: int) -> tuple[float, float, float]:
	"""Standard-library precision, recall and F1 for one class.

	An undefined ratio is 0.0 rather than skipped. That is the convention that keeps macro-F1 honest about
	a class the system never predicted, so a dataset dominated by easy labels cannot look good by quietly
	ignoring the rest.
	"""
	precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
	recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
	f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
	return precision, recall, f1


def per_class_f1(pairs: Sequence[tuple[str, str]]) -> dict[str, float]:
	"""Per-class F1 over ``(gold, predicted)`` pairs, for every class appearing on either side."""
	scores: dict[str, float] = {}
	for label in sorted({gold for gold, _ in pairs} | {predicted for _, predicted in pairs}):
		true_positive = sum(1 for gold, predicted in pairs if gold == label and predicted == label)
		false_positive = sum(1 for gold, predicted in pairs if gold != label and predicted == label)
		false_negative = sum(1 for gold, predicted in pairs if gold == label and predicted != label)
		scores[label] = precision_recall_f1(true_positive, false_positive, false_negative)[2]
	return scores


def macro_f1(pairs: Sequence[tuple[str, str]]) -> float:
	"""Unweighted mean of the per-class F1 values, or 0.0 when nothing was measured."""
	scores = per_class_f1(pairs)
	return sum(scores.values()) / len(scores) if scores else 0.0


def confusion_matrix(pairs: Sequence[tuple[str, str]], labels: Sequence[str]) -> dict[str, dict[str, int]]:
	"""``matrix[gold][predicted]`` over a fixed label list, so an unused cell still reports 0."""
	table = {gold: {predicted: 0 for predicted in labels} for gold in labels}
	for gold, predicted in pairs:
		table[gold][predicted] += 1
	return table


class BenchmarkRunCaseResult(BaseModel):
	"""One case measured: the ranked lists, the labels compared, and the numbers behind the aggregates.

	Nothing model-authored is stored beyond the relation labels and the status: no reasoning, no
	confidence, no prompt. The ranked id lists and the gold relations stay because they are what makes a
	headline number debuggable, showing whether a wrong status came from a retrieval miss or from a
	mislabelled page.
	"""

	case_id: str = Field(description='EvidenceBenchmarkCase.case_id')
	gold_status: VerificationStatus = Field(description='Gold claim status, kept so a report needs no dataset lookup')
	gold_relations: dict[str, EvidenceRelation] = Field(
		default_factory=dict, description='Gold relation per evidence id, copied from the case'
	)
	tags: list[str] = Field(default_factory=list, description='Case tags, for grouping in reports')
	lexical_ranked_evidence_ids: list[str] = Field(default_factory=list, description='Phase 4A candidate order')
	semantic_ranked_evidence_ids: list[str] = Field(default_factory=list, description='Phase 4B order; empty if no reranker ran')
	predicted_relations: dict[str, EvidenceRelation] = Field(
		default_factory=dict, description='Phase 5 relation per evidence id; empty when no verifier ran'
	)
	predicted_status: VerificationStatus | None = Field(default=None, description='Phase 5 status, or None if not run')
	retrieval_scored: bool = Field(
		default=True, description='False when the case has no gold relevant evidence, so retrieval metrics do not apply'
	)
	lexical_hit_at_1: bool = Field(default=False, description='Top lexical candidate is gold relevant')
	lexical_hit_at_k: bool = Field(default=False, description='A gold relevant page reached the lexical candidate set')
	lexical_reciprocal_rank: float = Field(default=0.0, ge=0.0, le=1.0, description='1 / rank of the first gold relevant page')
	semantic_hit_at_1: bool | None = Field(default=None, description='None when no reranker ran')
	semantic_hit_at_k: bool | None = Field(default=None, description='None when no reranker ran')
	semantic_reciprocal_rank: float | None = Field(default=None, ge=0.0, le=1.0, description='None when no reranker ran')
	lexical_miss: bool = Field(default=False, description='Gold relevant evidence that never entered the candidate set')
	relation_correct_count: int = Field(default=0, ge=0, description='Predicted relations equal to gold')
	relation_evaluated_count: int = Field(default=0, ge=0, description='Relations the verifier was asked about')
	status_correct: bool | None = Field(default=None, description='None when no verifier ran')
	notes: list[str] = Field(default_factory=list, description='Short machine-written observations about this case')

	@model_validator(mode='after')
	def _check_consistency(self) -> 'BenchmarkRunCaseResult':
		"""Keep the counters honest against the lists they summarize."""
		if self.relation_correct_count > self.relation_evaluated_count:
			raise ValueError(
				f'case {self.case_id!r} claims {self.relation_correct_count} correct of {self.relation_evaluated_count}'
			)
		if len(self.predicted_relations) != self.relation_evaluated_count:
			raise ValueError(
				f'case {self.case_id!r} holds {len(self.predicted_relations)} predictions but counts {self.relation_evaluated_count} evaluated'
			)
		if self.predicted_status is None and self.status_correct is not None:
			raise ValueError(f'case {self.case_id!r} has no predicted status but reports status_correct={self.status_correct}')
		if self.predicted_status is not None and self.semantic_hit_at_1 is None:
			raise ValueError(f'case {self.case_id!r} verified a status without any reranked candidate set to verify')
		return self


class BenchmarkSummary(BaseModel):
	"""Aggregate scores. A ``None`` rate means that stage never ran, never that it scored zero."""

	case_count: int = Field(default=0, ge=0, description='Cases measured')
	retrieval_case_count: int = Field(
		default=0, ge=0, description='Cases with at least one gold relevant page, i.e. the retrieval denominator'
	)
	lexical_hit_at_1_rate: float | None = Field(default=None, ge=0.0, le=1.0)
	lexical_hit_at_k_rate: float | None = Field(default=None, ge=0.0, le=1.0)
	lexical_mrr: float | None = Field(default=None, ge=0.0, le=1.0)
	semantic_hit_at_1_rate: float | None = Field(default=None, ge=0.0, le=1.0)
	semantic_hit_at_k_rate: float | None = Field(default=None, ge=0.0, le=1.0)
	semantic_mrr: float | None = Field(default=None, ge=0.0, le=1.0)
	relation_evaluated_count: int = Field(default=0, ge=0, description='Evidence relations predicted across all cases')
	relation_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
	relation_macro_f1: float | None = Field(default=None, ge=0.0, le=1.0)
	status_scored_case_count: int = Field(default=0, ge=0, description='Cases where a status was predicted and compared')
	status_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
	relation_confusion_matrix: dict[str, dict[str, int]] = Field(
		default_factory=dict, description='relation_confusion_matrix[gold][predicted]; empty when no verifier ran'
	)
	status_confusion_matrix: dict[str, dict[str, int]] = Field(
		default_factory=dict, description='status_confusion_matrix[gold][predicted]; empty when no verifier ran'
	)
	lexical_miss_case_ids: list[str] = Field(
		default_factory=list, description='Cases where gold relevant evidence never reached the candidate set'
	)
	status_error_case_ids: list[str] = Field(default_factory=list, description='Cases whose predicted status differs from gold')
	relation_error_case_ids: list[str] = Field(
		default_factory=list, description='Cases with at least one wrong relation label among those measured'
	)


class EvidenceBenchmarkResult(BaseModel):
	"""A whole run: aggregates plus the per-case detail that produced them, in dataset order."""

	summary: BenchmarkSummary = Field(default_factory=BenchmarkSummary)
	cases: list[BenchmarkRunCaseResult] = Field(default_factory=list, description='One entry per case, input order kept')


class EvidenceBenchmarkRunner:
	"""Measure retrieval and verification over frozen cases, with the participating stages chosen by injection.

	One code path serves every ablation mode. An aligner alone measures lexical recall; adding a reranker
	measures what rescoring contributed; adding a verifier measures the final claim status. A stage that is
	absent reports its metrics as ``None`` rather than as a zero, so a lexical-only run can never be read as
	"the reranker scored nothing".

	The runner never calls ``ClaimExtractor``, never runs the organizer or the report, and never offers the
	reranker a page the aligner did not recall. Each case carries one fixed claim, so a low score points at
	retrieval or verification rather than at a claim that was split differently this time.
	"""

	def __init__(
		self,
		*,
		aligner: EvidenceAligner,
		reranker: SemanticEvidenceReranker | None = None,
		verifier: ClaimVerifier | None = None,
	) -> None:
		if verifier is not None and reranker is None:
			# Verification consumes a RerankingResult, so this combination would have to invent one.
			raise EvidenceBenchmarkError('a verifier needs a reranker; use the lexical mode to measure retrieval alone')

		self.aligner = aligner
		self.reranker = reranker
		self.verifier = verifier

	@property
	def mode(self) -> str:
		"""Which of the three ablation modes this runner implements, for reporting rather than branching."""
		if self.verifier is not None:
			return 'full'
		return 'semantic' if self.reranker is not None else 'lexical'

	async def run(self, cases: Sequence[EvidenceBenchmarkCase]) -> EvidenceBenchmarkResult:
		"""Score every case in order.

		Raises:
			EvidenceBenchmarkExecutionError: as soon as a stage fails on a case, naming case and stage.
		"""
		results = [await self.run_case(case) for case in cases]
		return EvidenceBenchmarkResult(summary=self.summarize(results), cases=results)

	async def run_case(self, case: EvidenceBenchmarkCase) -> BenchmarkRunCaseResult:
		"""Measure one case through whichever stages were injected."""
		claim_set = case.claim_set()
		nodes = list(case.evidence_nodes)

		alignment = self._align(case, claim_set, nodes)
		lexical_ids = [match.evidence_id for entry in alignment.alignments for match in entry.matches]

		reranking = await self._rerank(case, claim_set, alignment, nodes) if self.reranker is not None else None
		verification = (
			await self._verify(case, claim_set, reranking, nodes) if self.verifier is not None and reranking is not None else None
		)

		gold_relevant = case.gold_relevant_ids
		lexical_hit_1, lexical_hit_k, lexical_rr = hit_metrics(lexical_ids, gold_relevant)
		result = BenchmarkRunCaseResult(
			case_id=case.case_id,
			gold_status=case.gold_status,
			gold_relations=case.gold_relation_by_id,
			tags=list(case.tags),
			lexical_ranked_evidence_ids=lexical_ids,
			semantic_ranked_evidence_ids=_ranked_ids(reranking),
			retrieval_scored=bool(gold_relevant),
			lexical_hit_at_1=lexical_hit_1,
			lexical_hit_at_k=lexical_hit_k,
			lexical_reciprocal_rank=lexical_rr,
			lexical_miss=bool(gold_relevant - set(lexical_ids)),
		)

		if reranking is not None:
			result.semantic_hit_at_1, result.semantic_hit_at_k, result.semantic_reciprocal_rank = hit_metrics(
				result.semantic_ranked_evidence_ids, gold_relevant
			)
		if verification is not None:
			_apply_verification(result, case, verification)

		result.notes.extend(_notes_for(result, gold_relevant))
		return result

	def _align(self, case: EvidenceBenchmarkCase, claim_set: ClaimSet, nodes: list[EvidenceNode]) -> AlignmentResult:
		"""Phase 4A, with any failure reported as a lost measurement rather than a wrong answer."""
		try:
			return self.aligner.align(claim_set=claim_set, evidence_nodes=nodes)
		except Exception as e:
			raise _execution_error(case, BenchmarkStage.LEXICAL_ALIGNMENT, e) from e

	async def _rerank(
		self, case: EvidenceBenchmarkCase, claim_set: ClaimSet, alignment: AlignmentResult, nodes: list[EvidenceNode]
	) -> RerankingResult:
		assert self.reranker is not None
		try:
			return await self.reranker.rerank(claim_set=claim_set, alignment_result=alignment, evidence_nodes=nodes)
		except Exception as e:
			raise _execution_error(case, BenchmarkStage.SEMANTIC_RERANKING, e) from e

	async def _verify(
		self, case: EvidenceBenchmarkCase, claim_set: ClaimSet, reranking: RerankingResult, nodes: list[EvidenceNode]
	) -> VerificationResult:
		assert self.verifier is not None
		try:
			return await self.verifier.verify(claim_set=claim_set, reranking_result=reranking, evidence_nodes=nodes)
		except Exception as e:
			raise _execution_error(case, BenchmarkStage.VERIFICATION, e) from e

	@staticmethod
	def summarize(results: Sequence[BenchmarkRunCaseResult]) -> BenchmarkSummary:
		"""Aggregate per-case rows. A rate stays ``None`` when its stage never ran."""
		retrieval_rows = [row for row in results if row.retrieval_scored]
		relation_pairs = _relation_pairs(results)
		status_pairs = [
			(row.gold_status.value, row.predicted_status.value) for row in results if row.predicted_status is not None
		]
		verified = any(row.predicted_status is not None for row in results)

		return BenchmarkSummary(
			case_count=len(results),
			retrieval_case_count=len(retrieval_rows),
			lexical_hit_at_1_rate=rate([row.lexical_hit_at_1 for row in retrieval_rows]),
			lexical_hit_at_k_rate=rate([row.lexical_hit_at_k for row in retrieval_rows]),
			lexical_mrr=mean([row.lexical_reciprocal_rank for row in retrieval_rows]),
			semantic_hit_at_1_rate=rate([row.semantic_hit_at_1 for row in retrieval_rows if row.semantic_hit_at_1 is not None]),
			semantic_hit_at_k_rate=rate([row.semantic_hit_at_k for row in retrieval_rows if row.semantic_hit_at_k is not None]),
			semantic_mrr=mean(
				[row.semantic_reciprocal_rank for row in retrieval_rows if row.semantic_reciprocal_rank is not None]
			),
			relation_evaluated_count=sum(row.relation_evaluated_count for row in results),
			relation_accuracy=rate([gold == predicted for gold, predicted in relation_pairs]) if relation_pairs else None,
			relation_macro_f1=macro_f1(relation_pairs) if relation_pairs else None,
			status_scored_case_count=len(status_pairs),
			status_accuracy=rate([gold == predicted for gold, predicted in status_pairs]) if status_pairs else None,
			relation_confusion_matrix=confusion_matrix(relation_pairs, [label.value for label in RELATION_LABELS])
			if verified
			else {},
			status_confusion_matrix=confusion_matrix(status_pairs, [label.value for label in STATUS_LABELS]) if verified else {},
			lexical_miss_case_ids=[row.case_id for row in results if row.lexical_miss],
			status_error_case_ids=[row.case_id for row in results if row.status_correct is False],
			relation_error_case_ids=[
				row.case_id
				for row in results
				if row.relation_evaluated_count and row.relation_correct_count < row.relation_evaluated_count
			],
		)


def _ranked_ids(reranking: RerankingResult | None) -> list[str]:
	"""The reranked candidate order, or an empty list when the reranker did not run."""
	if reranking is None:
		return []
	return [match.evidence_id for entry in reranking.rerankings for match in entry.matches]


def _execution_error(case: EvidenceBenchmarkCase, stage: BenchmarkStage, cause: Exception) -> EvidenceBenchmarkExecutionError:
	return EvidenceBenchmarkExecutionError(
		f'Benchmark case {case.case_id} failed at {stage.value}: {type(cause).__name__}',
		case_id=case.case_id,
		stage=stage,
	)


def _apply_verification(result: BenchmarkRunCaseResult, case: EvidenceBenchmarkCase, verification: VerificationResult) -> None:
	"""Copy one claim's predictions onto the row and score them against gold."""
	if len(verification.verifications) != 1:
		raise EvidenceBenchmarkExecutionError(
			f'Benchmark case {case.case_id} failed at {BenchmarkStage.VERIFICATION.value}: ExpectedOneVerification',
			case_id=case.case_id,
			stage=BenchmarkStage.VERIFICATION,
		)

	claim_verification = verification.verifications[0]
	gold_relations = case.gold_relation_by_id
	predicted = {assessment.evidence_id: assessment.relation for assessment in claim_verification.assessments}

	result.predicted_relations = predicted
	result.relation_evaluated_count = len(predicted)
	result.relation_correct_count = sum(
		1 for evidence_id, relation in predicted.items() if gold_relations.get(evidence_id) == relation
	)
	result.predicted_status = claim_verification.status
	result.status_correct = claim_verification.status is case.gold_status


def _notes_for(result: BenchmarkRunCaseResult, gold_relevant: frozenset[str]) -> list[str]:
	"""Short mechanical observations, so a report explains a number without re-deriving it elsewhere."""
	notes: list[str] = []
	if not gold_relevant:
		notes.append('no gold relevant evidence, so this case is excluded from the retrieval aggregates')
	if result.lexical_miss:
		missing = sorted(gold_relevant - set(result.lexical_ranked_evidence_ids))
		notes.append(f'{len(missing)} gold relevant page(s) never entered the lexical candidate set, first: {missing[0]!r}')
	unassessed = sorted(gold_relevant - set(result.predicted_relations))
	if result.predicted_relations and unassessed:
		notes.append(f'{len(unassessed)} gold relevant page(s) were never assessed, first: {unassessed[0]!r}')
	return notes


def _relation_pairs(results: Sequence[BenchmarkRunCaseResult]) -> list[tuple[str, str]]:
	"""Flatten each measured prediction into ``(gold, predicted)`` pairs, in case then id order."""
	pairs: list[tuple[str, str]] = []
	for row in results:
		for evidence_id, predicted in row.predicted_relations.items():
			gold = row.gold_relations.get(evidence_id)
			if gold is not None:
				pairs.append((gold.value, predicted.value))
	return pairs
