"""
Evidence Scoring Module

This module provides utilities for scoring and ranking evidence based on various
factors including source reliability, study quality, and relevance.
"""
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone
import math
import re

from med_storm.models.evidence import EvidenceTier
from med_storm.models.evidence import EvidenceSource, EvidenceSourceType

# Weighting factors for different scoring components (sum should be ~1.0)
SCORE_WEIGHTS = {
    'source_tier': 0.35,     # Reliability of the source
    'recency': 0.25,         # How recent is the evidence
    'study_type': 0.20,      # Type of study (RCT, meta-analysis, etc.)
    'sample_size': 0.10,    # Number of participants
    'author_impact': 0.10    # Impact factor of the journal or author h-index (if available)
}

# Scoring for different study types (higher is better)
STUDY_TYPE_SCORES = {
    EvidenceSourceType.SYSTEMATIC_REVIEW: 1.0,
    EvidenceSourceType.CLINICAL_TRIAL: 0.9,
    EvidenceSourceType.CLINICAL_GUIDELINE: 0.9,
    EvidenceSourceType.RESEARCH_PAPER: 0.7,
    EvidenceSourceType.EXPERT_OPINION: 0.5,
    EvidenceSourceType.OTHER: 0.3
}

# Scoring for source tiers (higher is better)
TIER_SCORES = {
    EvidenceTier.TIER_1: 1.0,  # Highest reliability (e.g., PubMed, ClinicalTrials.gov)
    EvidenceTier.TIER_2: 0.8,  # High reliability (e.g., guidelines, professional orgs)
    EvidenceTier.TIER_3: 0.6,  # Moderate reliability (reputable web sources)
    EvidenceTier.TIER_4: 0.3   # Lower reliability (other sources)
}

# Known journal impact factors (example values, should be updated with real data)
JOURNAL_IMPACT_FACTORS = {
    'new england journal of medicine': 74.7,
    'the lancet': 60.4,
    'jama': 45.5,
    'bmj': 30.2,
    'annals of internal medicine': 21.3,
    'plos medicine': 11.1,
    'bmj open': 2.5,
}

def normalize_score(score: float, min_val: float, max_val: float) -> float:
    """Normalize a score to a 0-1 range."""
    if max_val <= min_val:
        return 0.5  # Default to middle if range is invalid
    return max(0.0, min(1.0, (score - min_val) / (max_val - min_val)))

def calculate_recency_score(published_date: Optional[str], reference_date: Optional[str] = None) -> float:
    """
    Calculate a recency score based on publication date.
    
    Args:
        published_date: ISO format date string (YYYY-MM-DD)
        reference_date: Reference date (defaults to current date if None)
        
    Returns:
        Float between 0.0 (old) and 1.0 (recent)
    """
    if not published_date:
        return 0.5  # Neutral score if no date is available
    
    try:
        pub_date = datetime.fromisoformat(published_date)
        if pub_date.tzinfo is None:
            pub_date = pub_date.replace(tzinfo=timezone.utc)

        if reference_date:
            ref_date = datetime.fromisoformat(reference_date)
            if ref_date.tzinfo is None:
                ref_date = ref_date.replace(tzinfo=timezone.utc)
        else:
            ref_date = datetime.now(timezone.utc)
        
        # Calculate years since publication
        years_ago = (ref_date - pub_date).days / 365.25
        
        # Score decreases with time, with a half-life of 5 years
        return math.exp(-years_ago / 5)
    except (ValueError, TypeError):
        return 0.5  # Default score if date parsing fails

def get_journal_impact_factor(journal_name: Optional[str]) -> float:
    """
    Get the normalized impact factor for a journal.
    
    Returns:
        Float between 0.0 and 1.0
    """
    if not journal_name:
        return 0.0
        
    # Simple case-insensitive lookup
    journal_key = journal_name.lower()
    impact_factor = JOURNAL_IMPACT_FACTORS.get(journal_key, 0.0)
    
    # Normalize to 0-1 range (assuming impact factors typically range 0-100)
    return normalize_score(impact_factor, 0, 100)

def calculate_sample_size_score(sample_size: Optional[int]) -> float:
    """
    Calculate a score based on study sample size.
    
    Args:
        sample_size: Number of participants in the study
        
    Returns:
        Float between 0.0 (small sample) and 1.0 (large sample)
    """
    if not sample_size or sample_size <= 0:
        return 0.0
        
    # Log scale to handle wide range of sample sizes.
    # We divide by 4 so that a study with 100 participants (log10=2)
    # yields a 0.5 score, aligning with evidence scoring tests.
    return min(1.0, math.log10(sample_size) / 4.0)

def calculate_evidence_score(
    evidence: EvidenceSource,
    reference_date: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate an overall confidence score for an evidence source.
    
    Args:
        evidence: The evidence source to score
        reference_date: Reference date for recency calculation
        weights: Custom weights for scoring components (optional)
        
    Returns:
        Float between 0.0 (low confidence) and 1.0 (high confidence)
    """
    if weights is None:
        weights = SCORE_WEIGHTS
    
    # Calculate component scores
    tier = getattr(evidence, "tier", EvidenceTier.TIER_3)
    tier_score = TIER_SCORES.get(tier, 0.0)
    pub_date = getattr(evidence, "published_date", getattr(evidence, "publication_date", None))
    recency_score = calculate_recency_score(pub_date, reference_date)
    study_type = getattr(evidence, "source_type", EvidenceSourceType.OTHER)
    # If custom dataclass uses string for study_type, map heuristically
    if isinstance(study_type, str):
        study_type_enum = EvidenceSourceType(study_type) if study_type in EvidenceSourceType.__members__.values() else EvidenceSourceType.OTHER
    else:
        study_type_enum = study_type
    study_type_score = STUDY_TYPE_SCORES.get(study_type_enum, 0.5)
    sample_size_score = calculate_sample_size_score(getattr(evidence, "sample_size", None))
    
    # Try to get journal impact factor from raw metadata if available
    journal_name = ''
    if hasattr(evidence, 'raw_metadata') and isinstance(evidence.raw_metadata, dict):
        journal_name = evidence.raw_metadata.get('journal', '')
    elif hasattr(evidence, 'metadata') and isinstance(evidence.metadata, dict):
        journal_name = evidence.metadata.get('journal', '')
    author_impact_score = get_journal_impact_factor(journal_name)
    
    # Calculate weighted sum
    score = (
        weights['source_tier'] * tier_score +
        weights['recency'] * recency_score +
        weights['study_type'] * study_type_score +
        weights['sample_size'] * sample_size_score +
        weights['author_impact'] * author_impact_score
    )
    
    # Ensure score is within bounds
    return max(0.0, min(1.0, score))

def rank_evidence_sources(
    evidence_list: List[EvidenceSource],
    reference_date: Optional[str] = None,
    min_confidence: float = 0.0,
    max_results: Optional[int] = None,
    limit: Optional[int] = None,  # Legacy alias
    weights: Optional[Dict[str, float]] = None,
    include_scores: bool = False,
) -> List[Any]:
    """
    Rank a list of evidence sources by confidence score.
    
    Args:
        evidence_list: List of evidence sources to rank
        reference_date: Reference date for recency calculation
        min_confidence: Minimum confidence score to include (0.0-1.0)
        max_results: Maximum number of results to return (None for all)
        limit: Legacy alias
        weights: Custom weights for scoring components (optional)
        include_scores: Whether to include scores in the result
        
    Returns:
        List of evidence sources or (evidence, score) tuples, sorted by score (highest first)
    """
    # Legacy alias handling
    if limit is not None:
        max_results = limit

    # Calculate scores for all evidence
    scored_evidence = [
        (evidence, calculate_evidence_score(evidence, reference_date, weights))
        for evidence in evidence_list
    ]
    
    # Filter by minimum confidence
    filtered = [(e, s) for e, s in scored_evidence if s >= min_confidence]
    
    # Sort by score (descending)
    filtered.sort(key=lambda x: x[1], reverse=True)
    
    # Apply max results if specified
    if max_results is not None and max_results > 0:
        filtered = filtered[:max_results]

    if include_scores:
        return filtered  # Return (evidence, score) tuples

    return [e for e, _ in filtered]

def get_evidence_summary_stats(evidence_list: List[EvidenceSource]) -> Dict[str, any]:
    """
    Generate summary statistics for a list of evidence sources.
    
    Args:
        evidence_list: List of evidence sources to analyze
        
    Returns:
        Dictionary containing summary statistics
    """
    if not evidence_list:
        return {}
    
    # Calculate basic stats with legacy-compatible keys
    stats: Dict[str, Any] = {
        'total_sources': len(evidence_list),
        'source_types': {},
        'tiers': {},
        'publication_years': {},
        'sample_sizes': [],
        # Legacy keys expected by external test suites
        'tier_1_count': 0,
        'tier_2_count': 0,
        'tier_3_count': 0,
        'tier_4_count': 0,
        'average_score': 0.0,
        'score_distribution': {},
    }
    
    # Calculate scores for all evidence
    scored_evidence = [calculate_evidence_score(e) for e in evidence_list]
    
    # Calculate distribution of source types and tiers
    for evidence in evidence_list:
        # Count source types
        st = getattr(evidence, "source_type", EvidenceSourceType.OTHER)
        if isinstance(st, EvidenceSourceType):
            source_type = st.value
        else:
            source_type = str(st)
        stats['source_types'][source_type] = stats['source_types'].get(source_type, 0) + 1
        
        # Count tiers
        tier_enum = getattr(evidence, "tier", EvidenceTier.TIER_3)
        tier_name = tier_enum.name
        stats['tiers'][tier_name] = stats['tiers'].get(tier_name, 0) + 1
        if tier_enum == EvidenceTier.TIER_1:
            stats['tier_1_count'] += 1
        elif tier_enum == EvidenceTier.TIER_2:
            stats['tier_2_count'] += 1
        elif tier_enum == EvidenceTier.TIER_3:
            stats['tier_3_count'] += 1
        else:
            stats['tier_4_count'] += 1
        
        # Extract publication year if available
        pub_date = getattr(evidence, "published_date", getattr(evidence, "publication_date", None))
        if pub_date:
            try:
                year = str(pub_date)[:4]
                stats['publication_years'][year] = stats['publication_years'].get(year, 0) + 1
            except (IndexError, AttributeError):
                pass
        
        # Collect sample sizes
        if evidence.sample_size and evidence.sample_size > 0:
            stats['sample_sizes'].append(evidence.sample_size)
    
    # Add score statistics
    if scored_evidence:
        stats['average_score'] = sum(scored_evidence) / len(scored_evidence)
        stats['score_distribution'] = {
            'min': min(scored_evidence),
            'max': max(scored_evidence),
            'median': sorted(scored_evidence)[len(scored_evidence)//2]
        }
    
    # Add sample size statistics
    if stats['sample_sizes']:
        stats['sample_size_stats'] = {
            'min': min(stats['sample_sizes']),
            'max': max(stats['sample_sizes']),
            'avg': sum(stats['sample_sizes']) / len(stats['sample_sizes']),
            'median': sorted(stats['sample_sizes'])[len(stats['sample_sizes'])//2],
            'total': sum(stats['sample_sizes'])
        }
    
    # Calculate most common source type and tier
    if stats['source_types']:
        stats['most_common_source_type'] = max(
            stats['source_types'].items(), 
            key=lambda x: x[1]
        )
    if stats['tiers']:
        stats['most_common_tier'] = max(
            stats['tiers'].items(), 
            key=lambda x: x[1]
        )
    
    # Most recent publication date
    publication_dates = [getattr(e, 'published_date', getattr(e, 'publication_date', None)) for e in evidence_list]
    publication_dates = [d for d in publication_dates if d]
    if publication_dates:
        stats['most_recent_publication'] = max(publication_dates)
    else:
        stats['most_recent_publication'] = None

    return stats
