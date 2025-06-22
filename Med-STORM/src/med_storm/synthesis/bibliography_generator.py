from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from med_storm.models.evidence import EvidenceCorpus, EvidenceSource

class BibliographyGenerator:
    """Generates a consolidated bibliography from all collected evidence."""

    def generate_bibliography(self, evidence_map: Dict[str, Tuple[str, 'EvidenceCorpus']]) -> str:
        """
        Creates a formatted, deduplicated, and sorted bibliography.
        """
        all_sources: Dict[str, 'EvidenceSource'] = {}
        for _, (_, corpus) in evidence_map.items():
            for source in corpus.sources:
                if source.url not in all_sources:
                    all_sources[source.url] = source

        if not all_sources:
            return "No sources were cited in this report."

        # Sort sources by source name, then by title for consistent ordering
        sorted_sources = sorted(all_sources.values(), key=lambda s: (s.source_name, s.title))

        bibliography = "# Bibliography\n\n"
        for i, source in enumerate(sorted_sources, 1):
            author_str = ", ".join(source.authors) if source.authors else "N/A"
            bibliography += f"[{i}] {source.title}\n"
            bibliography += f"    Authors: {author_str}\n"
            # Categorize confidence score into levels (low, medium, high)
            if source.confidence_score >= 0.8:
                conf_level = 'high'
            elif source.confidence_score >= 0.5:
                conf_level = 'medium'
            else:
                conf_level = 'low'
            bibliography += f"    Source: {source.source_name} (Confidence: {conf_level}, Score: {source.confidence_score:.2f})\n"
            bibliography += f"    URL: {source.url}\n\n"

        return bibliography
