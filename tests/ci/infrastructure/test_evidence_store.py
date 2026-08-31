"""Unit tests for the WebEvidence evidence data layer."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from browser_use.evidence import EvidenceNode, JsonlEvidenceStore


def _make_node(**overrides) -> EvidenceNode:
	"""Build a valid evidence node, with any field replaced by ``overrides``."""
	data: dict = {
		'task_id': 'task-abc',
		'step_number': 1,
		'url': 'https://example.com/page',
		'title': 'Example Page',
		'text': 'Some extracted page text',
	}
	data.update(overrides)
	return EvidenceNode(**data)


class TestEvidenceNode:
	def test_required_fields_must_be_provided(self):
		with pytest.raises(ValidationError):
			EvidenceNode()  # type: ignore[call-arg]

	def test_default_field_values(self):
		node = _make_node()

		assert node.screenshot_path is None
		assert node.action_names == []
		assert node.metadata == {}
		assert node.evidence_id
		assert isinstance(node.created_at, datetime)
		assert node.created_at.tzinfo is not None

	def test_evidence_id_is_generated_uniquely(self):
		assert _make_node().evidence_id != _make_node().evidence_id

	def test_default_containers_are_not_shared_between_instances(self):
		first = _make_node()
		second = _make_node()

		first.action_names.append('click')
		first.metadata['key'] = 'value'

		assert second.action_names == []
		assert second.metadata == {}

	def test_step_number_must_be_one_based(self):
		with pytest.raises(ValidationError):
			_make_node(step_number=0)

	def test_json_round_trip_preserves_all_fields(self):
		node = _make_node(
			screenshot_path='shots/step-1.png',
			action_names=['click', 'scroll'],
			metadata={'tab': 0, 'score': 1.5, 'tags': ['a', 'b']},
			created_at=datetime(2026, 8, 31, 10, 30, tzinfo=timezone.utc),
		)

		restored = EvidenceNode.model_validate_json(node.model_dump_json())

		assert restored == node
		assert restored.created_at == node.created_at
		assert restored.metadata == node.metadata
		assert restored.action_names == ['click', 'scroll']


class TestJsonlEvidenceStore:
	def test_append_then_load_single_node(self, tmp_path):
		store = JsonlEvidenceStore(tmp_path / 'evidence.jsonl')
		node = _make_node()

		store.append(node)

		assert store.load_all() == [node]

	def test_append_writes_one_json_object_per_line(self, tmp_path):
		path = tmp_path / 'evidence.jsonl'
		store = JsonlEvidenceStore(path)

		store.append(_make_node(step_number=1))
		store.append(_make_node(step_number=2))

		lines = path.read_text(encoding='utf-8').strip().splitlines()
		assert len(lines) == 2
		assert all(line.lstrip().startswith('{') and line.rstrip().endswith('}') for line in lines)

	def test_append_multiple_nodes_keeps_count_and_order(self, tmp_path):
		store = JsonlEvidenceStore(tmp_path / 'evidence.jsonl')
		nodes = [_make_node(step_number=step, url=f'https://example.com/{step}') for step in range(1, 4)]

		for node in nodes:
			store.append(node)

		loaded = store.load_all()
		assert len(loaded) == 3
		assert [node.step_number for node in loaded] == [1, 2, 3]
		assert [node.url for node in loaded] == ['https://example.com/1', 'https://example.com/2', 'https://example.com/3']
		assert loaded == nodes

	def test_load_all_on_empty_file_returns_empty_list(self, tmp_path):
		path = tmp_path / 'evidence.jsonl'
		path.write_text('', encoding='utf-8')

		assert JsonlEvidenceStore(path).load_all() == []

	def test_load_all_before_first_append_returns_empty_list(self, tmp_path):
		assert JsonlEvidenceStore(tmp_path / 'missing' / 'evidence.jsonl').load_all() == []

	def test_new_store_instance_appends_to_existing_file(self, tmp_path):
		path = tmp_path / 'evidence.jsonl'

		JsonlEvidenceStore(path).append(_make_node(step_number=1))
		JsonlEvidenceStore(path).append(_make_node(step_number=2))

		assert [node.step_number for node in JsonlEvidenceStore(path).load_all()] == [1, 2]

	def test_clear_empties_the_store(self, tmp_path):
		store = JsonlEvidenceStore(tmp_path / 'evidence.jsonl')
		store.append(_make_node())
		store.append(_make_node(step_number=2))
		assert store.load_all()

		store.clear()

		assert store.load_all() == []

	def test_clear_is_safe_when_nothing_is_stored(self, tmp_path):
		store = JsonlEvidenceStore(tmp_path / 'evidence.jsonl')

		store.clear()

		assert store.load_all() == []

	def test_creates_missing_parent_directories(self, tmp_path):
		path = tmp_path / 'deep' / 'nested' / 'dir' / 'evidence.jsonl'
		assert not path.parent.exists()

		JsonlEvidenceStore(path).append(_make_node())

		assert path.parent.is_dir()
		assert path.exists()

	def test_accepts_str_path(self, tmp_path):
		store = JsonlEvidenceStore(str(tmp_path / 'from-str' / 'evidence.jsonl'))

		store.append(_make_node())

		assert [node.url for node in store.load_all()] == ['https://example.com/page']

	def test_unicode_content_survives_round_trip(self, tmp_path):
		store = JsonlEvidenceStore(tmp_path / 'evidence.jsonl')
		node = _make_node(title='证据标题', text='中文内容与符号 ✓')

		store.append(node)

		assert store.load_all() == [node]
