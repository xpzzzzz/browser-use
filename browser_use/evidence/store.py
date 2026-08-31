"""Local JSON Lines persistence for evidence nodes."""

from pathlib import Path

from browser_use.evidence.models import EvidenceNode


class JsonlEvidenceStore:
	"""Store evidence nodes as JSON Lines, one serialized ``EvidenceNode`` per line.

	Intentionally minimal: append-only writes, full scan on read, no database and
	no index. Enough for a single research task, easy to inspect by hand.
	"""

	def __init__(self, path: Path | str) -> None:
		self.path = Path(path)
		self._ensure_parent_dir()

	def append(self, node: EvidenceNode) -> None:
		"""Persist a single evidence node as a new line."""
		self._ensure_parent_dir()
		with self.path.open('a', encoding='utf-8') as f:
			f.write(node.model_dump_json() + '\n')

	def load_all(self) -> list[EvidenceNode]:
		"""Return every stored evidence node in append order."""
		if not self.path.exists():
			return []

		nodes: list[EvidenceNode] = []
		with self.path.open('r', encoding='utf-8') as f:
			for line in f:
				if line.strip():
					nodes.append(EvidenceNode.model_validate_json(line))
		return nodes

	def clear(self) -> None:
		"""Drop all stored evidence nodes."""
		if self.path.exists():
			self.path.write_text('', encoding='utf-8')

	def _ensure_parent_dir(self) -> None:
		self.path.parent.mkdir(parents=True, exist_ok=True)
