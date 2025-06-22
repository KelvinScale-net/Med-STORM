# ---------------------------------------------------------------------------
# Evidence data models                                                (v2 API)
# ---------------------------------------------------------------------------
# NOTE: This file has been migrated to Pydantic v2. It should **not** rely on
# legacy v1 compatibility features such as `@validator` or class-based Config
# objects. Instead we now use:
#   • `@field_validator` (and `@model_validator` when needed)
#   • `ConfigDict` for model configuration
#   • `model_dump()` / `model_validate()` for (de)serialization helpers
# ---------------------------------------------------------------------------

from enum import Enum
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ConfigDict,
    HttpUrl,  # noqa: F401 – retained for potential future use
)

class EvidenceTier(Enum):
    """Classification of evidence sources by trust level."""
    TIER_1 = 1  # Highest confidence (PubMed, ClinicalTrials.gov, Cochrane)
    TIER_2 = 2  # High confidence (Guidelines, Professional Organizations)
    TIER_3 = 3  # Moderate confidence (Reputable Web Sources)
    TIER_4 = 4  # Lower confidence (Other sources)

class EvidenceSourceType(str, Enum):
    """Type of evidence source."""
    RESEARCH_PAPER = "research_paper"
    CLINICAL_TRIAL = "clinical_trial"
    SYSTEMATIC_REVIEW = "systematic_review"
    CLINICAL_GUIDELINE = "clinical_guideline"
    EXPERT_OPINION = "expert_opinion"
    OTHER = "other"

# ---------------------------------------------------------------------------
# EvidenceSource
# ---------------------------------------------------------------------------

class EvidenceSource(BaseModel):
    """Represents a single piece of evidence, like a research paper or clinical trial."""
    id: Optional[str] = Field(None, description="Unique identifier, typically the PMID")
    title: str = Field(..., description="Title of the evidence source")
    url: str = Field(..., description="URL to access the source")
    summary: str = Field(..., description="Abstract or summary of the evidence")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    journal: Optional[str] = Field(None, description="Journal or publication name")
    publication_year: Optional[int] = Field(None, description="Year of publication")
    pmid: Optional[str] = Field(None, description="PubMed ID")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    source_name: str = Field(..., description="Name of the source connector (e.g., PubMed, Local Corpus)")
    confidence_score: Optional[float] = Field(None, description="Confidence score from retrieval (e.g., Qdrant similarity)")
    tier: 'EvidenceTier' = Field(default=EvidenceTier.TIER_3, description="Trust level tier of the evidence source")
    source_type: 'EvidenceSourceType' = Field(default=EvidenceSourceType.RESEARCH_PAPER, description="Study type of the evidence")
    sample_size: Optional[int] = Field(default=0, description="Sample size for clinical or observational studies")
    published_date: Optional[str] = Field(default=None, description="ISO formatted date when the study was published")
    raw_metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional raw metadata payload (kept for legacy compatibility)")

    # Deprecated: 'metadata' kept for backward compatibility
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Alias to raw_metadata for backward compatibility.")

    # ---------------------------------------------------------------------
    # Pydantic v2 configuration
    # ---------------------------------------------------------------------
    model_config = ConfigDict(
        extra="allow",  # Allow unknown fields for forward compatibility
        str_strip_whitespace=True,
    )

# ---------------------------------------------------------------------
# Validators (v2)
# ---------------------------------------------------------------------

    @field_validator("id", mode="before")
    def set_default_id(cls, v: Optional[str], info):  # noqa: N805 (cls)
        """Automatically assign an ID from PMID, DOI, or fallback to title hash if not provided."""
        if v:
            return v

        data = info.data  # Access to already-parsed/partial values
        pmid = data.get("pmid")
        if pmid:
            return str(pmid)

        doi = data.get("doi")
        if doi:
            return doi

        # Fallback: deterministic hash of title
        title: str = data.get("title", "")
        return str(abs(hash(title)))[:12]  # 12-digit hash fragment

    @field_validator("metadata", mode="before")
    def sync_metadata_alias(cls, v, info):  # noqa: N805
        """Ensure 'metadata' mirrors 'raw_metadata' for backward compatibility."""
        if v:
            # If metadata provided directly, also set raw_metadata if missing
            info.data.setdefault("raw_metadata", v)
            return v

        return info.data.get("raw_metadata", {})

class EvidenceCorpus(BaseModel):
    """
    A collection of evidence related to a specific query or research question.
    
    Attributes:
        query: The original search query
        sources: List of evidence sources
        search_timestamp: When the search was performed (ISO 8601 format)
        total_results: Total number of results found
        filtered_results: Number of results after filtering
    """
    query: str = Field(..., description="The original search query")
    sources: List[EvidenceSource] = Field(
        default_factory=list,
        description="List of evidence sources"
    )
    search_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When the search was performed (ISO 8601)"
    )
    total_results: int = Field(
        default=0,
        description="Total number of results found"
    )
    filtered_results: int = Field(
        default=0,
        description="Number of results after filtering"
    )
    
    def add_source(self, source: EvidenceSource) -> None:
        """Add a source to the corpus."""
        self.sources.append(source)
    
    def get_sources_by_tier(self, tier: EvidenceTier) -> List[EvidenceSource]:
        """Get all sources from a specific evidence tier."""
        return [s for s in self.sources if s.tier == tier]
    
    def get_trusted_sources(self, min_tier: EvidenceTier = EvidenceTier.TIER_3) -> List[EvidenceSource]:
        """
        Get sources that meet or exceed the specified minimum tier.
        
        Args:
            min_tier: Minimum tier to include (default: TIER_3)
            
        Returns:
            List of trusted evidence sources
        """
        return [s for s in self.sources if s.tier.value <= min_tier.value]
    
    def sort_by_confidence(self, reverse: bool = True) -> None:
        """Sort sources by confidence score."""
        self.sources.sort(key=lambda x: x.confidence_score, reverse=reverse)
    
    def filter_by_confidence(self, min_confidence: float = 0.5) -> 'EvidenceCorpus':
        """
        Create a new EvidenceCorpus with sources above a minimum confidence threshold.
        
        Args:
            min_confidence: Minimum confidence score (0.0-1.0)
            
        Returns:
            New EvidenceCorpus with filtered sources
        """
        filtered = EvidenceCorpus(
            query=self.query,
            sources=[s for s in self.sources if s.confidence_score >= min_confidence],
            search_timestamp=self.search_timestamp,
            total_results=self.total_results,
            filtered_results=len([s for s in self.sources if s.confidence_score >= min_confidence])
        )
        return filtered
