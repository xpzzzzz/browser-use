"""Evidence data layer for the WebEvidence research agent."""

from browser_use.evidence.collector import EvidenceCollector
from browser_use.evidence.models import EvidenceNode
from browser_use.evidence.store import JsonlEvidenceStore

__all__ = ['EvidenceCollector', 'EvidenceNode', 'JsonlEvidenceStore']
