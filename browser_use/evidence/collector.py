"""Collect pre-action page observations from an agent step into evidence nodes."""

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from browser_use.evidence.models import EvidenceNode
from browser_use.evidence.store import JsonlEvidenceStore

if TYPE_CHECKING:
	from browser_use.agent.views import AgentOutput
	from browser_use.browser.views import BrowserStateSummary

logger = logging.getLogger(__name__)


class EvidenceCollector:
	"""Turn each agent step into an ``EvidenceNode`` and persist it.

	Semantics: this captures the **pre-action** page state, i.e. exactly what the LLM saw
	when it decided the actions of that step. Browser Use passes the ``BrowserStateSummary``
	captured by ``_prepare_context`` to ``register_new_step_callback`` *before*
	``_execute_actions`` runs, so the page produced by a click shows up in the evidence of
	the next step, not of this one.

	Failures are deliberately isolated: evidence collection is auxiliary and must never
	break the agent loop, so every extraction degrades to an empty value plus a warning.
	"""

	def __init__(
		self,
		task_id: str,
		store: JsonlEvidenceStore,
		screenshot_dir: str | Path,
		include_attributes: list[str] | None = None,
	) -> None:
		"""Args:
		task_id: the owning ``Agent`` task id, injected because the step callback does not receive the agent.
		store: where evidence nodes are appended.
		screenshot_dir: directory for ``step_XXXX.png`` screenshots, created on demand.
		include_attributes: forwarded to ``dom_state.llm_representation`` so the stored text matches
			the agent's own page representation. ``None`` keeps Browser Use's default attributes.
		"""
		self.task_id = task_id
		self.store = store
		self.screenshot_dir = Path(screenshot_dir)
		self.include_attributes = include_attributes

	async def collect_step(
		self,
		browser_state_summary: 'BrowserStateSummary',
		model_output: 'AgentOutput',
		step_number: int,
	) -> None:
		"""Record one step; usable directly as ``register_new_step_callback``."""
		try:
			node = EvidenceNode(
				task_id=self.task_id,
				step_number=step_number,
				url=browser_state_summary.url or '',
				title=browser_state_summary.title or '',
				text=self._extract_dom_text(browser_state_summary, step_number),
				screenshot_path=self._save_screenshot(browser_state_summary, step_number),
				action_names=self._extract_action_names(model_output),
				metadata={'observation_phase': 'pre_action'},
			)
		except Exception as e:
			logger.warning(f'Failed to build evidence node for step {step_number}: {type(e).__name__}: {e}. Skipping evidence.')
			return

		try:
			self.store.append(node)
		except Exception as e:
			logger.warning(f'Failed to persist evidence {node.evidence_id} for step {step_number}: {type(e).__name__}: {e}.')

	def _extract_dom_text(self, browser_state_summary: 'BrowserStateSummary', step_number: int) -> str:
		"""Get the LLM-facing DOM text, or an empty string when it is unavailable."""
		dom_state = getattr(browser_state_summary, 'dom_state', None)
		if not dom_state:
			return ''

		try:
			return dom_state.llm_representation(self.include_attributes)
		except Exception as e:
			logger.warning(f'Failed to extract DOM text for evidence at step {step_number}: {e}')
			return ''

	@staticmethod
	def _extract_action_names(model_output: 'AgentOutput') -> list[str]:
		"""Read the action names the model chose this step, in output order, without assuming any action set."""
		action_names: list[str] = []
		for action in getattr(model_output, 'action', None) or []:
			try:
				action_data = action.model_dump(exclude_unset=True)
			except Exception as e:
				logger.warning(f'Failed to read an action for evidence: {e}')
				continue

			# an unset ActionModel dumps to {}, which carries no name to record
			action_name = next(iter(action_data), None)
			if action_name:
				action_names.append(action_name)
		return action_names

	def _save_screenshot(self, browser_state_summary: 'BrowserStateSummary', step_number: int) -> str | None:
		"""Decode and persist the observation screenshot, returning its path or ``None``."""
		screenshot = getattr(browser_state_summary, 'screenshot', None)
		if not screenshot:
			return None

		try:
			screenshot_bytes = base64.b64decode(screenshot)
		except Exception as e:
			logger.warning(f'Undecodable screenshot at step {step_number}: {type(e).__name__}: {e}. Saving evidence without it.')
			return None

		try:
			self.screenshot_dir.mkdir(parents=True, exist_ok=True)
			screenshot_path = self.screenshot_dir / f'step_{step_number:04d}.png'
			screenshot_path.write_bytes(screenshot_bytes)
		except Exception as e:
			logger.warning(f'Failed to write evidence screenshot for step {step_number}: {e}. Saving evidence without it.')
			return None

		return str(screenshot_path)
