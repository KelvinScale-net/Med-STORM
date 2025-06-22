"""study_table.py
------------------
Utilities to convert a list of StudyData objects into:
1. Pandas DataFrame (if pandas installed)  
2. Markdown table string  
3. CSV file saved under ``output/`` for further analysis.

The helper is intentionally dependency-light; if *pandas* is missing we
fallback to the built-in ``csv`` module and simple string building.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List
from datetime import datetime, timezone

from med_storm.statistics.advanced_analysis import StudyData

try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – optional
    pd = None  # type: ignore

def to_dataframe(studies: List[StudyData]):  # type: ignore[override]
    """Return pandas DataFrame or list of dicts if pandas unavailable."""
    rows = [s.__dict__ for s in studies]
    if pd is not None:
        return pd.DataFrame(rows)
    return rows  # Fallback

def to_markdown(studies: List[StudyData]) -> str:
    """Generate GitHub-flavoured Markdown table."""
    if not studies:
        return "_No structured studies found_"

    headers = ["ID", "Sample", "Effect", "SE"]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for s in studies:
        lines.append(f"{s.study_id} | {s.sample_size} | {s.effect_size:.3f} | {s.standard_error:.3f}")
    return "\n".join(lines)

def save_csv(studies: List[StudyData], output_dir: Path = Path("output")) -> Path:
    """Save studies to CSV and return path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fn = output_dir / f"study_table_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    with fn.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["study_id", "sample_size", "effect_size", "standard_error"])
        for s in studies:
            writer.writerow([s.study_id, s.sample_size, s.effect_size, s.standard_error])
    return fn 