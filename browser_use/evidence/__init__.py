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
from browser_use.evidence.organization import (
	ClaimEvidenceEdge,
	ClaimGraphNode,
	EvidenceEdgeType,
	EvidenceEvidenceEdge,
	EvidenceGraph,
	EvidenceGraphNode,
	EvidenceGraphStats,
	EvidenceOrganizationError,
	EvidenceOrganizer,
)
from browser_use.evidence.pipeline import (
	PipelineStage,
	WebEvidencePipeline,
	WebEvidencePipelineError,
	WebEvidencePipelineResult,
)
from browser_use.evidence.reporting import (
	ClaimReportSection,
	EvidenceGroundedReport,
	EvidenceReportBuilder,
	EvidenceReportError,
	MarkdownReportRenderer,
	ReportClaimEvidence,
	ReportEvidenceSource,
	ReportSummary,
)
from browser_use.evidence.reranking import (
	ClaimReranking,
	EvidenceRerankingError,
	RawSemanticEvidenceScore,
	RawSemanticReranking,
	RerankedEvidenceMatch,
	RerankingResult,
	SemanticEvidenceReranker,
)
from browser_use.evidence.store import JsonlEvidenceStore
from browser_use.evidence.verification import (
	ClaimVerification,
	ClaimVerificationError,
	ClaimVerifier,
	EvidenceAssessment,
	EvidenceRelation,
	VerificationResult,
	VerificationStatus,
)

__all__ = [
	'AlignmentResult',
	'Claim',
	'ClaimAlignment',
	'ClaimExtractor',
	'ClaimExtractionError',
	'ClaimEvidenceEdge',
	'ClaimGraphNode',
	'ClaimReranking',
	'ClaimSet',
	'ClaimVerification',
	'ClaimVerificationError',
	'ClaimVerifier',
	'EvidenceAligner',
	'EvidenceAssessment',
	'EvidenceCollector',
	'EvidenceMatch',
	'EvidenceNode',
	'EvidenceRerankingError',
	'EvidenceRelation',
	'EvidenceEdgeType',
	'EvidenceEvidenceEdge',
	'EvidenceGraph',
	'EvidenceGraphNode',
	'EvidenceGraphStats',
	'EvidenceGroundedReport',
	'EvidenceOrganizationError',
	'EvidenceOrganizer',
	'EvidenceReportBuilder',
	'EvidenceReportError',
	'PipelineStage',
	'JsonlEvidenceStore',
	'MarkdownReportRenderer',
	'RawSemanticEvidenceScore',
	'RawSemanticReranking',
	'RawClaim',
	'RawClaimExtraction',
	'RerankedEvidenceMatch',
	'RerankingResult',
	'ReportClaimEvidence',
	'ReportEvidenceSource',
	'ReportSummary',
	'SemanticEvidenceReranker',
	'ClaimReportSection',
	'VerificationResult',
	'VerificationStatus',
	'WebEvidencePipeline',
	'WebEvidencePipelineError',
	'WebEvidencePipelineResult',
]
