"""
Utilities for generating comparison tables from evidence sources.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from med_storm.models.evidence import EvidenceSource

class ComparisonColumnType(Enum):
    """Types of columns for comparison tables."""
    TEXT = "text"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    CITATION = "citation"

@dataclass
class ComparisonColumn:
    """Definition of a column in a comparison table."""
    key: str
    title: str
    type: ComparisonColumnType = ComparisonColumnType.TEXT
    sortable: bool = True
    filterable: bool = True
    width: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "title": self.title,
            "type": self.type.value,
            "sortable": self.sortable,
            "filterable": self.filterable,
            "width": self.width
        }

def generate_comparison_table(
    sources: List[EvidenceSource], 
    columns: List[ComparisonColumn]
) -> Dict[str, Any]:
    """
    Generate a comparison table from a list of evidence sources.
    
    Args:
        sources: List of evidence sources to include in the table
        columns: List of column definitions
        
    Returns:
        Dictionary with table definition and data
    """
    # Default columns if none provided
    if not columns:
        columns = [
            ComparisonColumn("title", "Title", width="30%"),
            ComparisonColumn("authors", "Authors", width="20%"),
            ComparisonColumn("journal", "Journal", width="15%"),
            ComparisonColumn("publication_date", "Year", 
                          type=ComparisonColumnType.DATE, width="10%"),
            ComparisonColumn("study_type", "Study Type", width="15%"),
            ComparisonColumn("sample_size", "Sample Size", 
                          type=ComparisonColumnType.NUMBER, width="10%")
        ]
    
    # Convert sources to table rows
    rows = []
    for source in sources:
        row = {
            "id": source.id,
            "source_url": source.url,
            "confidence_score": source.confidence_score,
            "metadata": source.metadata or {}
        }
        
        # Add columns from source metadata
        for col in columns:
            if col.key in source.metadata:
                row[col.key] = source.metadata[col.key]
            elif hasattr(source, col.key):
                row[col.key] = getattr(source, col.key, "")
        
        rows.append(row)
    
    # Prepare column definitions
    column_defs = [col.to_dict() for col in columns]
    
    return {
        "columns": column_defs,
        "rows": rows,
        "total": len(rows),
        "sources": [source.id for source in sources]
    }

def generate_evidence_summary_table(sources: List[EvidenceSource]) -> Dict[str, Any]:
    """
    Generate a summary table of evidence sources with key metrics.
    
    Args:
        sources: List of evidence sources
        
    Returns:
        Dictionary with table definition and data
    """
    columns = [
        ComparisonColumn("title", "Study", width="30%"),
        ComparisonColumn("year", "Year", type=ComparisonColumnType.DATE, width="10%"),
        ComparisonColumn("study_type", "Design", width="15%"),
        ComparisonColumn("sample_size", "N", type=ComparisonColumnType.NUMBER, width="10%"),
        ComparisonColumn("intervention", "Intervention", width="20%"),
        ComparisonColumn("outcome", "Primary Outcome", width="25%")
    ]
    
    return generate_comparison_table(sources, columns)

def generate_risk_of_bias_table(sources: List[EvidenceSource]) -> Dict[str, Any]:
    """
    Generate a risk of bias assessment table.
    
    Args:
        sources: List of evidence sources with risk of bias information
        
    Returns:
        Dictionary with table definition and data
    """
    columns = [
        ComparisonColumn("title", "Study", width="25%"),
        ComparisonColumn("year", "Year", type=ComparisonColumnType.DATE, width="10%"),
        ComparisonColumn("randomization", "Randomization", type=ComparisonColumnType.BOOLEAN, width="15%"),
        ComparisonColumn("blinding", "Blinding", type=ComparisonColumnType.BOOLEAN, width="15%"),
        ComparisonColumn("dropout_rate", "Dropout Rate", type=ComparisonColumnType.NUMBER, width="15%"),
        ComparisonColumn("conflict_of_interest", "COI", type=ComparisonColumnType.BOOLEAN, width="10%"),
        ComparisonColumn("funding_source", "Funding", width="20%")
    ]
    
    return generate_comparison_table(sources, columns)
