import re
from pathlib import Path
import pytest


def test_quality_gate_complete() -> None:
    """Fail the CI pipeline if quality-gate checklist is not yet complete.

    The markdown table in *docs/quality_gate/standards_comparison.md* intentionally
    uses the symbols `✗` and the word *Parcial* to mark unmet criteria. Any
    occurrence of these indicators means the quality bar has not been reached
    and therefore the build must fail.
    """
    md_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "quality_gate"
        / "standards_comparison.md"
    )

    assert md_path.exists(), f"Quality-gate file not found at expected path: {md_path}"
    content = md_path.read_text(encoding="utf-8")

    # Detect any ❌ or partially-met markers.
    unmet_pattern = re.compile(r"✗|Parcial", re.IGNORECASE)
    unmet = unmet_pattern.search(content)

    assert (
        unmet is None
    ), "Quality gate checklist is not fully complete – unmet criteria remain." 