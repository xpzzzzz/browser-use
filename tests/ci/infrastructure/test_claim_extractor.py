"""Unit tests for atomic claim extraction; a fake chat model stands in for the LLM."""

import traceback

import pytest
from pydantic import ValidationError

from browser_use.evidence import Claim, ClaimExtractionError, ClaimExtractor, ClaimSet, RawClaim, RawClaimExtraction
from browser_use.llm.messages import SystemMessage, UserMessage
from browser_use.llm.views import ChatInvokeCompletion


class FakeChatModel:
	"""Minimal chat model that records calls and replays a canned completion."""

	def __init__(self, completion: RawClaimExtraction | object | None = None, error: Exception | None = None) -> None:
		self.model = 'fake-claim-model'
		self.provider = 'fake'
		self.name = 'fake-claim-model'
		self.model_name = 'fake-claim-model'
		self._verified_api_keys = True
		self.calls: list[dict] = []
		self._completion = completion
		self._error = error

	async def ainvoke(self, messages, output_format=None, **kwargs) -> ChatInvokeCompletion:
		self.calls.append({'messages': messages, 'output_format': output_format, 'kwargs': kwargs})
		if self._error is not None:
			raise self._error
		return ChatInvokeCompletion(completion=self._completion, usage=None)


def _extraction(*texts: str) -> RawClaimExtraction:
	return RawClaimExtraction(claims=[RawClaim(text=text) for text in texts])


TASK = 'Research what Browser Use is built with.'
ANSWER = 'Browser Use is open source and is primarily written in Python.'


class TestClaim:
	def test_claim_has_generated_id_and_keeps_text_and_order(self):
		claim = Claim(text='Browser Use is open source.', order=1)

		assert claim.claim_id
		assert claim.text == 'Browser Use is open source.'
		assert claim.order == 1

	def test_claim_rejects_blank_text(self):
		for blank in ('', '   ', '\n\t  '):
			with pytest.raises(ValidationError):
				Claim(text=blank, order=1)

	def test_claim_strips_surrounding_whitespace(self):
		assert Claim(text='  Browser Use is open source.  ', order=1).text == 'Browser Use is open source.'

	def test_claim_order_must_be_one_based(self):
		for invalid in (0, -1):
			with pytest.raises(ValidationError):
				Claim(text='Some claim.', order=invalid)

	def test_claim_ids_are_unique_across_calls(self):
		ids = {Claim(text='Same text.', order=1).claim_id for _ in range(25)}

		assert len(ids) == 25


class TestClaimExtractor:
	async def test_three_raw_claims_become_three_claims(self):
		llm = FakeChatModel(completion=_extraction('Claim one.', 'Claim two.', 'Claim three.'))

		result = await ClaimExtractor(llm).extract(task_id='task-1', task=TASK, answer='a. b. c.')

		assert isinstance(result, ClaimSet)
		assert len(result.claims) == 3

	async def test_orders_are_one_based_and_sequential(self):
		llm = FakeChatModel(completion=_extraction('Claim one.', 'Claim two.', 'Claim three.'))

		result = await ClaimExtractor(llm).extract(task_id='task-1', task=TASK, answer='a. b. c.')

		assert [claim.order for claim in result.claims] == [1, 2, 3]

	async def test_claim_ids_are_non_empty_and_unique(self):
		llm = FakeChatModel(completion=_extraction('Claim one.', 'Claim two.', 'Claim three.'))

		result = await ClaimExtractor(llm).extract(task_id='task-1', task=TASK, answer='a. b. c.')

		claim_ids = [claim.claim_id for claim in result.claims]
		assert all(claim_ids)
		assert len(set(claim_ids)) == 3

	async def test_task_id_task_and_answer_are_preserved(self):
		llm = FakeChatModel(completion=_extraction('Claim one.'))

		result = await ClaimExtractor(llm).extract(task_id='task-42', task=TASK, answer=ANSWER)

		assert result.task_id == 'task-42'
		assert result.task == TASK
		assert result.answer == ANSWER

	async def test_raw_claim_text_is_stripped(self):
		llm = FakeChatModel(completion=_extraction('   Browser Use is open source.   ', '\nBrowser Use uses Python.\t'))

		result = await ClaimExtractor(llm).extract(task_id='task-1', task=TASK, answer=ANSWER)

		assert [claim.text for claim in result.claims] == ['Browser Use is open source.', 'Browser Use uses Python.']

	async def test_blank_raw_claims_are_dropped_and_ordering_stays_compact(self):
		llm = FakeChatModel(completion=_extraction('Claim one.', '', '   ', '\n\t', 'Claim two.'))

		result = await ClaimExtractor(llm).extract(task_id='task-1', task=TASK, answer=ANSWER)

		assert [claim.text for claim in result.claims] == ['Claim one.', 'Claim two.']
		assert [claim.order for claim in result.claims] == [1, 2]

	async def test_extraction_keeps_the_model_claim_order(self):
		llm = FakeChatModel(completion=_extraction('First stated fact.', 'Second stated fact.', 'Third stated fact.'))

		result = await ClaimExtractor(llm).extract(task_id='task-1', task=TASK, answer=ANSWER)

		assert [claim.text for claim in result.claims] == ['First stated fact.', 'Second stated fact.', 'Third stated fact.']

	async def test_split_claims_stay_split_with_distinct_ids(self):
		# The pipeline must not re-merge what the model already split into atomic claims.
		llm = FakeChatModel(completion=_extraction('Browser Use is open source.', 'Browser Use is primarily written in Python.'))

		result = await ClaimExtractor(llm).extract(task_id='task-1', task=TASK, answer=ANSWER)

		assert [claim.text for claim in result.claims] == [
			'Browser Use is open source.',
			'Browser Use is primarily written in Python.',
		]
		assert [claim.order for claim in result.claims] == [1, 2]
		assert result.claims[0].claim_id != result.claims[1].claim_id

	async def test_empty_answer_returns_empty_claim_set_without_calling_llm(self):
		llm = FakeChatModel(completion=_extraction('Should never be used.'))

		for blank_answer in ('', '   ', '\n\t '):
			result = await ClaimExtractor(llm).extract(task_id='task-1', task=TASK, answer=blank_answer)

			assert result.claims == []
			assert result.answer == blank_answer
			assert result.task == TASK
			assert result.task_id == 'task-1'

		assert llm.calls == []

	async def test_llm_failure_raises_instead_of_empty_claim_set(self):
		llm = FakeChatModel(error=RuntimeError('provider exploded'))

		with pytest.raises(ClaimExtractionError) as excinfo:
			await ClaimExtractor(llm).extract(task_id='task-1', task=TASK, answer=ANSWER)

		# the original failure stays reachable through the cause, never through our own message
		assert str(excinfo.value) == 'Claim extraction failed: RuntimeError'
		assert 'provider exploded' in str(excinfo.value.__cause__)
		assert isinstance(excinfo.value.__cause__, RuntimeError)

	async def test_failure_message_does_not_leak_prompt_or_answer(self):
		task = 'Task: USER_PRIVATE_TASK find the repository stars'
		answer = 'ANSWER-SENTINEL the repository has 30k stars'
		llm = FakeChatModel(error=RuntimeError('upstream timeout while sending messages'))

		with pytest.raises(ClaimExtractionError) as excinfo:
			await ClaimExtractor(llm).extract(task_id='task-1', task=task, answer=answer)

		message = str(excinfo.value)
		for secret in ('USER_PRIVATE_TASK', 'ANSWER-SENTINEL', '30k', 'stars', 'repository'):
			assert secret not in message
		assert isinstance(excinfo.value.__cause__, RuntimeError)

	async def test_failure_message_does_not_leak_original_exception_text(self):
		llm = FakeChatModel(error=RuntimeError('secret-api-key=abc123 USER_PRIVATE_ANSWER'))

		with pytest.raises(ClaimExtractionError) as excinfo:
			await ClaimExtractor(llm).extract(task_id='task-1', task=TASK, answer=ANSWER)

		message = str(excinfo.value)
		assert 'abc123' not in message
		assert 'secret-api-key' not in message
		assert 'USER_PRIVATE_ANSWER' not in message
		# ... but the wrapped exception keeps the detail for whoever logs the cause or traceback
		assert str(excinfo.value.__cause__) == 'secret-api-key=abc123 USER_PRIVATE_ANSWER'
		assert 'abc123' in ''.join(traceback.format_exception(excinfo.value))

	async def test_non_structured_completion_is_rejected(self):
		llm = FakeChatModel(completion='just a string, not a claim set')

		with pytest.raises(ClaimExtractionError) as excinfo:
			await ClaimExtractor(llm).extract(task_id='task-1', task=TASK, answer=ANSWER)

		assert 'just a string' not in str(excinfo.value)

	async def test_no_claims_returned_yields_empty_claim_set(self):
		llm = FakeChatModel(completion=RawClaimExtraction(claims=[]))

		result = await ClaimExtractor(llm).extract(task_id='task-1', task=TASK, answer='In my opinion, it looks nice.')

		assert result.claims == []

	async def test_request_asks_for_the_structured_output_model(self):
		llm = FakeChatModel(completion=_extraction('Claim one.'))

		await ClaimExtractor(llm).extract(task_id='task-1', task=TASK, answer=ANSWER)

		call = llm.calls[0]
		assert call['output_format'] is RawClaimExtraction

	async def test_prompt_contains_task_and_answer(self):
		llm = FakeChatModel(completion=_extraction('Claim one.'))

		await ClaimExtractor(llm).extract(task_id='task-1', task=TASK, answer=ANSWER)

		messages = llm.calls[0]['messages']
		assert [type(message) for message in messages] == [SystemMessage, UserMessage]
		user_content = messages[-1].text
		assert TASK in user_content
		assert ANSWER in user_content

	async def test_system_prompt_defines_atomic_verifiable_claims(self):
		llm = FakeChatModel(completion=_extraction('Claim one.'))

		await ClaimExtractor(llm).extract(task_id='task-1', task=TASK, answer=ANSWER)

		system_content = llm.calls[0]['messages'][0].text
		for instruction in ('one independently verifiable fact', 'open source', '100k stars', 'empty claims list', 'order'):
			assert instruction in system_content

	async def test_consecutive_extractions_do_not_reuse_claim_ids(self):
		extractor = ClaimExtractor(FakeChatModel(completion=_extraction('Shared claim text.')))

		first = await extractor.extract(task_id='task-1', task=TASK, answer=ANSWER)
		second = await extractor.extract(task_id='task-2', task=TASK, answer=ANSWER)

		assert first.claims[0].claim_id != second.claims[0].claim_id
