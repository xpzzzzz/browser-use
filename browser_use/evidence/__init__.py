"""Evidence data layer for the WebEvidence research agent."""

from browser_use.evidence.alignment import AlignmentResult, ClaimAlignment, EvidenceAligner, EvidenceMatch
from browser_use.evidence.claim_extractor import (
	ClaimExtractionError,
	ClaimExtractor,
	RawClaim,
	RawClaimExtraction,
)
from browser_use.evidence.claims import Claim, ClaimSet
from browser_use.evidence.collector import EvidenceCollector
from browser_use.evidence.models import EvidenceNode
from browser_use.evidence.store import JsonlEvidenceStore

__all__ = [
	'AlignmentResult',
	'Claim',
	'ClaimAlignment',
	'ClaimExtractor',
	'ClaimExtractionError',
	'ClaimSet',
	'EvidenceAligner',
	'EvidenceCollector',
	'EvidenceMatch',
	'EvidenceNode',
	'JsonlEvidenceStore',
	'RawClaim',
	'RawClaimExtraction',
]
