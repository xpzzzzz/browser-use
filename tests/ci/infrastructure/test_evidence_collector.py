"""Unit tests for the WebEvidence collector; no browser, no LLM, no network."""

import base64
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from browser_use.agent.views import AgentOutput
from browser_use.browser.views import BrowserStateSummary
from browser_use.evidence import EvidenceCollector, JsonlEvidenceStore
from browser_use.evidence import collector as collector_module
from browser_use.tools.service import Tools

PNG_BYTES = b'\x89PNG\r\n\x1a\n fake but valid png payload \r\n'


class _StubDomState:
	"""Stand-in for SerializedDOMState that only exposes what the collector uses."""

	def __init__(self, text: str = '', error: Exception | None = None) -> None:
		self.selector_map: dict = {}
		self._text = text
		self._error = error

	def llm_representation(self, include_attributes: list[str] | None = None) -> str:
		if self._error is not None:
			raise self._error
		return self._text


class _ExplodingStore:
	"""Store double whose append always fails, to prove the collector stays quiet."""

	def append(self, node) -> None:
		raise RuntimeError('disk on fire')


class _StubAction:
	"""Action double that dumps whatever the real ActionModel would have reported."""

	def __init__(self, payload: dict) -> None:
		self._payload = payload

	def model_dump(self, **kwargs) -> dict:
		return self._payload


def _browser_state(**overrides) -> BrowserStateSummary:
	data: dict = {
		'dom_state': _StubDomState('- page: example\n- heading "Example Page"'),
		'url': 'https://example.com/page',
		'title': 'Example Page',
		'tabs': [],
		'screenshot': None,
	}
	data.update(overrides)
	return BrowserStateSummary(**data)


@pytest.fixture(scope='module')
def action_types() -> tuple[type, type[AgentOutput]]:
	"""Real AgentOutput/ActionModel classes built from the default Tools registry."""
	action_model = Tools().registry.create_action_model()
	return action_model, AgentOutput.type_with_custom_actions(action_model)


def _model_output(action_types, *actions: dict) -> AgentOutput:
	action_model, output_type = action_types
	return output_type(memory='looking at the page', action=[action_model(**action) for action in actions])


def _collector(tmp_path: Path, **overrides) -> EvidenceCollector:
	kwargs: dict = {
		'task_id': 'task-abc',
		'store': JsonlEvidenceStore(tmp_path / 'evidence.jsonl'),
		'screenshot_dir': tmp_path / 'shots',
	}
	kwargs.update(overrides)
	return EvidenceCollector(**kwargs)


@pytest.fixture
def evidence_warnings(monkeypatch) -> list[str]:
	"""Spy on the collector logger; browser-use keeps its loggers off the root handler, so caplog stays empty."""
	messages: list[str] = []
	monkeypatch.setattr(
		collector_module, 'logger', SimpleNamespace(warning=lambda message, *args, **kwargs: messages.append(str(message)))
	)
	return messages


class TestEvidenceCollector:
	async def test_collect_step_appends_evidence_node(self, tmp_path, action_types):
		collector = _collector(tmp_path)

		await collector.collect_step(
			_browser_state(screenshot=base64.b64encode(PNG_BYTES).decode()),
			_model_output(action_types, {'done': {'text': 'ok'}}),
			1,
		)

		assert len(collector.store.load_all()) == 1

	async def test_task_id_step_number_url_and_title_are_recorded(self, tmp_path, action_types):
		collector = _collector(tmp_path, task_id='task-42')

		await collector.collect_step(_browser_state(), _model_output(action_types, {'done': {'text': 'ok'}}), 7)

		node = collector.store.load_all()[0]
		assert node.task_id == 'task-42'
		assert node.step_number == 7
		assert node.url == 'https://example.com/page'
		assert node.title == 'Example Page'

	async def test_dom_text_becomes_evidence_text(self, tmp_path, action_types):
		collector = _collector(tmp_path)
		state = _browser_state(dom_state=_StubDomState('- heading "Quotes to scrape"'))

		await collector.collect_step(state, _model_output(action_types, {'done': {'text': 'ok'}}), 1)

		assert collector.store.load_all()[0].text == '- heading "Quotes to scrape"'

	async def test_action_names_keep_model_output_order(self, tmp_path, action_types):
		collector = _collector(tmp_path)
		output = _model_output(
			action_types,
			{'search': {'query': 'web evidence'}},
			{'click': {'index': 3}},
			{'done': {'text': 'ok', 'success': True}},
		)

		await collector.collect_step(_browser_state(), output, 1)

		assert collector.store.load_all()[0].action_names == ['search', 'click', 'done']

	def test_action_entries_without_a_set_name_are_skipped(self):
		# An ActionModel that dumps {} carries no action name, so it must not become an empty entry.
		class _Output:
			action = [_StubAction({}), _StubAction({'click': {'index': 2}}), _StubAction({'search': {'query': 'q'}})]

		assert EvidenceCollector._extract_action_names(_Output()) == ['click', 'search']

	def test_action_entries_that_fail_to_dump_are_skipped(self):
		class _BrokenAction:
			def model_dump(self, **kwargs) -> dict:
				raise RuntimeError('cannot serialize')

		class _Output:
			action = [_BrokenAction(), _StubAction({'done': {'text': 'ok'}})]

		assert EvidenceCollector._extract_action_names(_Output()) == ['done']

	def test_model_output_without_action_list_yields_no_names(self):
		assert EvidenceCollector._extract_action_names(SimpleNamespace(action=None)) == []
		assert EvidenceCollector._extract_action_names(SimpleNamespace()) == []

	async def test_empty_step_output_yields_no_action_names(self, tmp_path, action_types):
		collector = _collector(tmp_path)
		_, output_type = action_types

		await collector.collect_step(_browser_state(), output_type(memory='m', action=[]), 1)

		assert collector.store.load_all()[0].action_names == []

	async def test_missing_model_output_yields_no_action_names(self, tmp_path):
		collector = _collector(tmp_path)

		await collector.collect_step(_browser_state(), None, 1)

		assert collector.store.load_all()[0].action_names == []

	async def test_screenshot_is_saved_and_matches_original_bytes(self, tmp_path, action_types):
		collector = _collector(tmp_path, screenshot_dir=tmp_path / 'deep' / 'shots')

		await collector.collect_step(
			_browser_state(screenshot=base64.b64encode(PNG_BYTES).decode()),
			_model_output(action_types, {'done': {'text': 'ok'}}),
			2,
		)

		node = collector.store.load_all()[0]
		screenshot_path = Path(node.screenshot_path)
		assert screenshot_path.name == 'step_0002.png'
		assert screenshot_path.read_bytes() == PNG_BYTES

	async def test_missing_screenshot_leaves_path_none(self, tmp_path, action_types):
		collector = _collector(tmp_path)

		await collector.collect_step(_browser_state(screenshot=None), _model_output(action_types, {'done': {'text': 'ok'}}), 1)

		assert collector.store.load_all()[0].screenshot_path is None
		assert not collector.screenshot_dir.exists()

	async def test_unwritable_screenshot_dir_still_saves_evidence(self, tmp_path, action_types, evidence_warnings):
		# a regular file where the screenshot directory should be makes mkdir fail
		blocked_dir = tmp_path / 'blocked'
		blocked_dir.write_text('not a directory', encoding='utf-8')
		collector = _collector(tmp_path, screenshot_dir=blocked_dir)

		await collector.collect_step(
			_browser_state(screenshot=base64.b64encode(PNG_BYTES).decode()),
			_model_output(action_types, {'done': {'text': 'ok'}}),
			6,
		)

		nodes = collector.store.load_all()
		assert len(nodes) == 1
		assert nodes[0].screenshot_path is None
		assert nodes[0].text
		assert any('Failed to write evidence screenshot' in message for message in evidence_warnings)

	async def test_dom_extraction_failure_still_saves_evidence(self, tmp_path, action_types, evidence_warnings):
		collector = _collector(tmp_path)
		state = _browser_state(
			dom_state=_StubDomState(error=RuntimeError('serializer blew up')),
			screenshot=base64.b64encode(PNG_BYTES).decode(),
		)

		await collector.collect_step(state, _model_output(action_types, {'done': {'text': 'ok'}}), 3)

		nodes = collector.store.load_all()
		assert len(nodes) == 1
		assert nodes[0].text == ''
		assert nodes[0].screenshot_path is not None
		assert any('Failed to extract DOM text' in message for message in evidence_warnings)

	async def test_missing_dom_state_yields_empty_text(self, tmp_path, action_types):
		collector = _collector(tmp_path)

		await collector.collect_step(_browser_state(dom_state=None), _model_output(action_types, {'done': {'text': 'ok'}}), 1)

		assert collector.store.load_all()[0].text == ''

	async def test_undecodable_screenshot_still_saves_evidence(self, tmp_path, action_types, evidence_warnings):
		collector = _collector(tmp_path)

		await collector.collect_step(_browser_state(screenshot='AA'), _model_output(action_types, {'done': {'text': 'ok'}}), 4)

		nodes = collector.store.load_all()
		assert len(nodes) == 1
		assert nodes[0].screenshot_path is None
		assert nodes[0].text
		assert any('Undecodable screenshot' in message for message in evidence_warnings)

	async def test_store_append_failure_does_not_raise(self, tmp_path, action_types, evidence_warnings):
		collector = _collector(tmp_path, store=_ExplodingStore())

		await collector.collect_step(_browser_state(), _model_output(action_types, {'done': {'text': 'ok'}}), 5)

		assert any('Failed to persist evidence' in message for message in evidence_warnings)

	async def test_invalid_step_number_is_skipped_without_raising(self, tmp_path, action_types, evidence_warnings):
		collector = _collector(tmp_path)

		await collector.collect_step(_browser_state(), _model_output(action_types, {'done': {'text': 'ok'}}), 0)

		assert collector.store.load_all() == []
		assert any('Failed to build evidence node' in message for message in evidence_warnings)

	async def test_metadata_marks_the_pre_action_phase(self, tmp_path, action_types):
		collector = _collector(tmp_path)

		await collector.collect_step(_browser_state(), _model_output(action_types, {'done': {'text': 'ok'}}), 1)

		assert collector.store.load_all()[0].metadata == {'observation_phase': 'pre_action'}

	async def test_consecutive_steps_are_stored_in_order(self, tmp_path, action_types):
		collector = _collector(tmp_path)

		for step_number in (1, 2, 3):
			await collector.collect_step(
				_browser_state(url=f'https://example.com/{step_number}', screenshot=base64.b64encode(PNG_BYTES).decode()),
				_model_output(action_types, {'click': {'index': step_number}}),
				step_number,
			)

		nodes = collector.store.load_all()
		assert [node.step_number for node in nodes] == [1, 2, 3]
		assert [node.url for node in nodes] == ['https://example.com/1', 'https://example.com/2', 'https://example.com/3']
		assert [node.action_names for node in nodes] == [['click'], ['click'], ['click']]
		assert [Path(node.screenshot_path).name for node in nodes] == ['step_0001.png', 'step_0002.png', 'step_0003.png']

	async def test_include_attributes_is_forwarded_to_dom_extraction(self, tmp_path, action_types):
		recorded: list[list[str] | None] = []

		class _RecordingDomState(_StubDomState):
			def llm_representation(self, include_attributes: list[str] | None = None) -> str:
				recorded.append(include_attributes)
				return 'text'

		collector = _collector(tmp_path, include_attributes=['href', 'aria-label'])

		await collector.collect_step(
			_browser_state(dom_state=_RecordingDomState()), _model_output(action_types, {'done': {'text': 'ok'}}), 1
		)

		assert recorded == [['href', 'aria-label']]

	def test_collect_step_matches_the_step_callback_contract(self, tmp_path):
		"""The agent checks for a coroutine function and calls the callback with three positional args."""
		collector = _collector(tmp_path)

		assert inspect.iscoroutinefunction(collector.collect_step)
		assert list(inspect.signature(collector.collect_step).parameters) == [
			'browser_state_summary',
			'model_output',
			'step_number',
		]
