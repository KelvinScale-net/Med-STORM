"""
Utilities for generating executive summaries from evidence sources.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re

from med_storm.models.evidence import EvidenceSource, EvidenceCorpus

class SummarySectionType(Enum):
    """Types of sections in an executive summary."""
    OVERVIEW = "overview"
    KEY_FINDINGS = "key_findings"
    METHODOLOGY = "methodology"
    CLINICAL_IMPLICATIONS = "clinical_implications"
    LIMITATIONS = "limitations"
    RECOMMENDATIONS = "recommendations"

@dataclass
class SummarySection:
    """A section in an executive summary."""
    type: SummarySectionType
    title: str
    content: str
    source_ids: List[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "title": self.title,
            "content": self.content,
            "source_ids": self.source_ids or [],
            "metadata": self.metadata or {}
        }

class ExecutiveSummaryGenerator:
    """Generates executive summaries from evidence sources."""
    
    def __init__(self, llm_provider=None):
        """Initialize with an optional LLM provider for advanced summarization."""
        self.llm = llm_provider
    
    def generate_summary(
        self, 
        corpus: EvidenceCorpus,
        query: str = None,
        target_length: int = 1000,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Generate an executive summary from evidence sources.
        
        Args:
            corpus: EvidenceCorpus containing sources to summarize
            query: Original query that generated the evidence
            target_length: Approximate target length in words
            language: Language for the summary
            
        Returns:
            Dictionary with summary sections and metadata
        """
        if not corpus or not corpus.sources:
            return {
                "sections": [
                    SummarySection(
                        type=SummarySectionType.OVERVIEW,
                        title="No Evidence Found",
                        content="No relevant evidence was found for the query."
                    ).to_dict()
                ],
                "metadata": {
                    "query": query,
                    "sources_used": [],
                    "total_sources": 0,
                    "language": language
                }
            }
        
        # Group sources by study type and recency
        sources_by_type = {}
        for source in corpus.sources:
            study_type = source.metadata.get("study_type", "Other")
            if study_type not in sources_by_type:
                sources_by_type[study_type] = []
            sources_by_type[study_type].append(source)
        
        # Generate sections
        sections = []
        
        # 1. Overview Section
        overview = self._generate_overview(corpus, query, language)
        sections.append(overview)
        
        # 2. Key Findings
        key_findings = self._generate_key_findings(corpus, language)
        sections.append(key_findings)
        
        # 3. Methodology
        methodology = self._generate_methodology(sources_by_type, language)
        sections.append(methodology)
        
        # 4. Clinical Implications
        implications = self._generate_clinical_implications(corpus, language)
        sections.append(implications)
        
        # 5. Limitations
        limitations = self._generate_limitations(corpus, language)
        sections.append(limitations)
        
        # 6. Recommendations
        recommendations = self._generate_recommendations(corpus, language)
        sections.append(recommendations)
        
        # Prepare final summary
        return {
            "sections": [s.to_dict() for s in sections],
            "metadata": {
                "query": query,
                "sources_used": [s.id for s in corpus.sources],
                "total_sources": len(corpus.sources),
                "language": language,
                "generated_at": corpus.timestamp.isoformat() if corpus.timestamp else None
            }
        }
    
    def _generate_overview(
        self, 
        corpus: EvidenceCorpus, 
        query: str,
        language: str
    ) -> SummarySection:
        """Generate the overview/introduction section."""
        # Count studies by type
        study_counts = {}
        for source in corpus.sources:
            study_type = source.metadata.get("study_type", "Other")
            study_counts[study_type] = study_counts.get(study_type, 0) + 1
        
        # Build overview text
        total_studies = len(corpus.sources)
        study_types = ", ".join([f"{count} {typ}" for typ, count in study_counts.items()])
        
        content = (
            f"This summary is based on {total_studies} studies ({study_types}) "
            f"related to '{query}'. The evidence includes systematic reviews, "
            "randomized controlled trials, and observational studies published "
            f"between {self._get_publication_range(corpus)}."
        )
        
        return SummarySection(
            type=SummarySectionType.OVERVIEW,
            title="Overview",
            content=content,
            source_ids=[s.id for s in corpus.sources]
        )
    
    def _generate_key_findings(
        self, 
        corpus: EvidenceCorpus,
        language: str
    ) -> SummarySection:
        """Generate the key findings section."""
        # Group findings by outcome
        findings = {}
        for source in corpus.sources:
            outcome = source.metadata.get("outcome", "General")
            if outcome not in findings:
                findings[outcome] = []
            findings[outcome].append(source)
        
        # Generate findings text
        content_parts = []
        for outcome, sources in findings.items():
            content_parts.append(f"**{outcome}**:")
            
            # Group by direction of effect if available
            effects = {}
            for src in sources:
                effect = src.metadata.get("effect_direction", "neutral")
                if effect not in effects:
                    effects[effect] = []
                effects[effect].append(src)
            
            # Add findings for each effect direction
            for effect, effect_sources in effects.items():
                count = len(effect_sources)
                if effect == "positive":
                    content_parts.append(f"- {count} study/studies reported positive effects")
                elif effect == "negative":
                    content_parts.append(f"- {count} study/studies reported negative effects")
                else:
                    content_parts.append(f"- {count} study/studies reported no significant effect")
                
                # Add key statistics if available
                stats = []
                for src in effect_sources[:3]:  # Limit to top 3 per category
                    if "statistics" in src.metadata:
                        stats.append(f"{src.metadata.get('statistics')} ({src.metadata.get('sample_size', 'N/A')})")
                
                if stats:
                    content_parts[-1] += f" ({'; '.join(stats)})"
            
            content_parts.append("\n")
        
        return SummarySection(
            type=SummarySectionType.KEY_FINDINGS,
            title="Key Findings",
            content="\n".join(content_parts).strip(),
            source_ids=[s.id for s in corpus.sources]
        )
    
    def _generate_methodology(
        self, 
        sources_by_type: Dict[str, List[EvidenceSource]],
        language: str
    ) -> SummarySection:
        """Generate the methodology section."""
        content_parts = ["The evidence base includes the following types of studies:"]
        
        for study_type, sources in sources_by_type.items():
            count = len(sources)
            sample_sizes = [
                s.metadata.get("sample_size") 
                for s in sources 
                if s.metadata and "sample_size" in s.metadata
            ]
            
            sample_info = ""
            if sample_sizes:
                avg_sample = sum(sample_sizes) / len(sample_sizes)
                sample_info = f" (average sample size: {avg_sample:.0f})"
            
            content_parts.append(f"- {count} {study_type} studies{sample_info}")
        
        return SummarySection(
            type=SummarySectionType.METHODOLOGY,
            title="Methodology",
            content="\n".join(content_parts),
            source_ids=[s.id for sources in sources_by_type.values() for s in sources]
        )
    
    def _generate_clinical_implications(
        self, 
        corpus: EvidenceCorpus,
        language: str
    ) -> SummarySection:
        """Generate clinical implications section."""
        # Extract key clinical implications from sources
        implications = []
        
        for source in corpus.sources:
            if not source.metadata:
                continue
                
            if "clinical_implications" in source.metadata:
                implications.append({
                    "text": source.metadata["clinical_implications"],
                    "source_id": source.id,
                    "study_type": source.metadata.get("study_type", "Study")
                })
        
        # Group similar implications
        grouped = {}
        for imp in implications:
            key = imp["text"][:100]  # First 100 chars as key
            if key not in grouped:
                grouped[key] = {
                    "text": imp["text"],
                    "sources": [],
                    "study_types": set()
                }
            grouped[key]["sources"].append(imp["source_id"])
            grouped[key]["study_types"].add(imp["study_type"])
        
        # Generate content
        content_parts = []
        for imp in grouped.values():
            study_types = ", ".join(sorted(imp["study_types"]))
            content_parts.append(f"- {imp['text']} (Based on {len(imp['sources'])} {study_types} studies)")
        
        if not content_parts:
            content_parts.append("No specific clinical implications were extracted from the available evidence.")
        
        return SummarySection(
            type=SummarySectionType.CLINICAL_IMPLICATIONS,
            title="Clinical Implications",
            content="\n".join(content_parts),
            source_ids=[s.id for s in corpus.sources]
        )
    
    def _generate_limitations(
        self, 
        corpus: EvidenceCorpus,
        language: str
    ) -> SummarySection:
        """Generate limitations section."""
        limitations = []
        
        # Common limitations to check for
        common_limitations = [
            "small sample size",
            "retrospective design",
            "lack of control group",
            "short follow-up",
            "heterogeneous population",
            "risk of bias",
            "confounding factors"
        ]
        
        # Count limitations across studies
        limitation_counts = {lim: 0 for lim in common_limitations}
        
        for source in corpus.sources:
            if not source.metadata:
                continue
                
            # Check study design limitations
            study_type = source.metadata.get("study_type", "").lower()
            if "retrospective" in study_type:
                limitation_counts["retrospective design"] += 1
            
            # Check sample size
            sample_size = source.metadata.get("sample_size")
            if sample_size and sample_size < 50:
                limitation_counts["small sample size"] += 1
            
            # Check for other limitations in metadata
            if "limitations" in source.metadata:
                for lim in common_limitations:
                    if lim in source.metadata["limitations"].lower():
                        limitation_counts[lim] += 1
        
        # Generate limitations text
        content_parts = []
        for lim, count in limitation_counts.items():
            if count > 0:
                content_parts.append(f"- {count} studies had {lim}")
        
        if not content_parts:
            content_parts = ["No major limitations were identified across the included studies."]
        
        return SummarySection(
            type=SummarySectionType.LIMITATIONS,
            title="Limitations",
            content="\n".join(content_parts),
            source_ids=[s.id for s in corpus.sources]
        )
    
    def _generate_recommendations(
        self, 
        corpus: EvidenceCorpus,
        language: str
    ) -> SummarySection:
        """Generate recommendations section."""
        # Extract recommendations from sources
        recommendations = []
        
        for source in corpus.sources:
            if not source.metadata:
                continue
                
            if "recommendations" in source.metadata:
                recs = source.metadata["recommendations"]
                if isinstance(recs, str):
                    recs = [recs]
                
                for rec in recs:
                    recommendations.append({
                        "text": rec,
                        "source_id": source.id,
                        "confidence": source.confidence
                    })
        
        # Group similar recommendations
        grouped = {}
        for rec in recommendations:
            key = rec["text"][:100]  # First 100 chars as key
            if key not in grouped:
                grouped[key] = {
                    "text": rec["text"],
                    "sources": [],
                    "confidence_scores": []
                }
            grouped[key]["sources"].append(rec["source_id"])
            grouped[key]["confidence_scores"].append(rec["confidence"])
        
        # Sort by average confidence (descending)
        sorted_recs = sorted(
            grouped.values(),
            key=lambda x: sum(x["confidence_scores"]) / len(x["confidence_scores"]),
            reverse=True
        )
        
        # Generate content
        content_parts = []
        for i, rec in enumerate(sorted_recs[:5], 1):  # Top 5 recommendations
            avg_confidence = sum(rec["confidence_scores"]) / len(rec["confidence_scores"])
            confidence_str = f"(Confidence: {avg_confidence:.1f}/1.0, " \
                           f"based on {len(rec['sources'])} sources)"
            content_parts.append(f"{i}. {rec['text']} {confidence_str}")
        
        if not content_parts:
            content_parts = ["No specific recommendations were extracted from the available evidence."]
        
        return SummarySection(
            type=SummarySectionType.RECOMMENDATIONS,
            title="Recommendations",
            content="\n".join(content_parts),
            source_ids=list(set([s.id for s in corpus.sources]))
        )
    
    def _get_publication_range(self, corpus: EvidenceCorpus) -> str:
        """Get publication date range from sources."""
        if not corpus.sources:
            return "N/A"
            
        years = []
        for source in corpus.sources:
            if source.metadata and "publication_date" in source.metadata:
                try:
                    # Try to extract year from various date formats
                    date_str = str(source.metadata["publication_date"])
                    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
                    if year_match:
                        years.append(int(year_match.group()))
                except (ValueError, AttributeError):
                    continue
        
        if not years:
            return "N/A"
            
        min_year = min(years)
        max_year = max(years)
        
        if min_year == max_year:
            return str(min_year)
        return f"{min_year} to {max_year}"
