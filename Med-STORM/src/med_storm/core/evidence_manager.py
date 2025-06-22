"""
Evidence Manager for Med-STORM

This module handles the classification and management of evidence sources
based on their trustworthiness and relevance.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import re

from med_storm.models.evidence import EvidenceTier

@dataclass
class EvidenceSourceConfig:
    """Configuration for an evidence source."""
    name: str
    tier: EvidenceTier
    domains: List[str]  # List of domain patterns to match
    requires_api_key: bool = False
    priority: int = 0  # Higher number = higher priority within same tier

class EvidenceManager:
    """Manages evidence sources and their classification."""
    
    def __init__(self):
        # Define known evidence sources and their configurations
        self.sources: Dict[str, EvidenceSourceConfig] = self._initialize_known_sources()
        
    def _initialize_known_sources(self) -> Dict[str, EvidenceSourceConfig]:
        """Initialize the list of known evidence sources with their configurations."""
        return {
            # Tier 1: Highest confidence (Peer-reviewed, clinical trials)
            "pubmed": EvidenceSourceConfig(
                name="PubMed",
                tier=EvidenceTier.TIER_1,
                domains=["pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov/pubmed"],
                requires_api_key=True,
                priority=100
            ),
            "clinicaltrials_gov": EvidenceSourceConfig(
                name="ClinicalTrials.gov",
                tier=EvidenceTier.TIER_1,
                domains=["clinicaltrials.gov"],
                priority=90
            ),
            "cochrane": EvidenceSourceConfig(
                name="Cochrane Library",
                tier=EvidenceTier.TIER_1,
                domains=["cochranelibrary.com"],
                priority=80
            ),
            
            # Tier 2: High confidence (Guidelines, Professional Organizations)
            "guideline_gov": EvidenceSourceConfig(
                name="Guideline Central",
                tier=EvidenceTier.TIER_2,
                domains=["guidelinecentral.com", "guideline.gov"],
                priority=70
            ),
            "uptodate": EvidenceSourceConfig(
                name="UpToDate",
                tier=EvidenceTier.TIER_2,
                domains=["uptodate.com"],
                requires_api_key=True,
                priority=60
            ),
            
            # Tier 3: Reputable Web Sources
            "mayoclinic": EvidenceSourceConfig(
                name="Mayo Clinic",
                tier=EvidenceTier.TIER_3,
                domains=["mayoclinic.org"],
                priority=50
            ),
            "webmd": EvidenceSourceConfig(
                name="WebMD",
                tier=EvidenceTier.TIER_3,
                domains=["webmd.com"],
                priority=40
            ),
            
            # Default for unknown sources will be Tier 4
        }
    
    def get_source_config(self, url: str) -> Tuple[EvidenceSourceConfig, bool]:
        """
        Get the configuration for a source based on its URL.
        
        Args:
            url: The URL of the source
            
        Returns:
            Tuple of (EvidenceSourceConfig, is_known) where is_indicators whether the source is known
        """
        if not url:
            return self._get_default_config(), False
            
        # Try to match against known domains
        for source_id, config in self.sources.items():
            for domain in config.domains:
                if domain.lower() in url.lower():
                    return config, True
        
        # Default for unknown sources
        return self._get_default_config(), False
    
    def _get_default_config(self) -> EvidenceSourceConfig:
        """Get the default configuration for unknown sources."""
        return EvidenceSourceConfig(
            name="Unknown Source",
            tier=EvidenceTier.TIER_4,
            domains=[],
            priority=0
        )
    
    def get_source_tier(self, url: str) -> EvidenceTier:
        """Get the evidence tier for a given URL."""
        config, _ = self.get_source_config(url)
        return config.tier
    
    def is_trusted_source(self, url: str, min_tier: EvidenceTier = EvidenceTier.TIER_3) -> bool:
        """
        Check if a source is trusted based on its URL and minimum required tier.
        
        Args:
            url: The URL to check
            min_tier: Minimum tier to be considered trusted (default: TIER_3)
            
        Returns:
            bool: True if the source is trusted, False otherwise
        """
        config, _ = self.get_source_config(url)
        return config.tier.value <= min_tier.value
    
    def get_source_name(self, url: str) -> str:
        """Get the display name for a source based on its URL."""
        config, is_known = self.get_source_config(url)
        if is_known:
            return config.name
        
        # For unknown sources, try to extract a meaningful name from the URL
        try:
            domain = url.split('//')[-1].split('/')[0]
            return domain
        except (IndexError, AttributeError):
            return "Unknown Source"

    # ------------------------------------------------------------------
    # Backward-compatibility helpers (used by legacy tests)
    # ------------------------------------------------------------------

    _evidence_store: List['EvidenceSource'] = []

    def add_evidence(self, evidence: 'EvidenceSource') -> None:  # noqa: F821
        """Add an EvidenceSource to the internal store (legacy helper)."""
        self._evidence_store.append(evidence)

    def list_evidence(self) -> List['EvidenceSource']:
        """Return all stored EvidenceSource objects (legacy helper)."""
        return list(self._evidence_store)

    # Legacy alias methods expected by certain test suites
    def get_all_evidence(self) -> List['EvidenceSource']:
        """Alias of list_evidence for backward compatibility."""
        return self.list_evidence()

    def get_evidence_by_id(self, evidence_id: str):
        """Retrieve a stored EvidenceSource by its ID, if present."""
        for ev in self._evidence_store:
            if getattr(ev, "id", None) == evidence_id:
                return ev
        return None

# Singleton instance
evidence_manager = EvidenceManager()
