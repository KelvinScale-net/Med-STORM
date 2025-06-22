"""grade_evaluator.py
----------------------
Implements a simplified GRADE certainty assessment based on number of RCTs,
effect sizes and risk of bias results.
"""
from __future__ import annotations

from typing import List, Dict, Tuple

from dataclasses import dataclass

from med_storm.statistics.advanced_analysis import StudyData


@dataclass
class GRADEFactors:
    """Quantify downgrade / upgrade elements for transparency."""

    risk_of_bias: int = 0  # 0–2
    inconsistency: int = 0
    indirectness: int = 0
    imprecision: int = 0
    publication_bias: int = 0

    large_effect: int = 0  # 0–2
    dose_response: int = 0
    confounding_reduces_effect: int = 0

    def total_downgrade(self) -> int:
        return sum([self.risk_of_bias, self.inconsistency, self.indirectness, self.imprecision, self.publication_bias])

    def total_upgrade(self) -> int:
        return sum([self.large_effect, self.dose_response, self.confounding_reduces_effect])


GRADE_ORDER = ["High", "Moderate", "Low", "Very low"]


class GradeEvaluator:
    """Compute overall certainty (High / Moderate / Low / Very low)."""

    def evaluate(
        self,
        studies: List[StudyData],
        rob_summaries: List[Dict[str, str]],
        *,
        heterogeneity_i2: float | None = None,
        ci_width: float | None = None,
        publication_bias_detected: bool | None = None,
        large_effect: bool | None = None,
        dose_response: bool | None = None,
    ) -> Tuple[str, GRADEFactors]:
        """Return certainty rating and contributing factor scores.

        Parameters accepted as simple numeric / boolean surrogates to avoid deep stats.
        """

        factors = GRADEFactors()

        # --------------------------------------------------
        # Downgrades
        # --------------------------------------------------
        high_risk = sum(1 for s in rob_summaries if s.get("overall_bias") == "High risk")
        some_concerns = sum(1 for s in rob_summaries if s.get("overall_bias") == "Some concerns")

        if high_risk / max(len(rob_summaries), 1) > 0.3:
            factors.risk_of_bias = 2
        elif some_concerns / max(len(rob_summaries), 1) > 0.3 or high_risk > 0:
            factors.risk_of_bias = 1

        if heterogeneity_i2 is not None:
            if heterogeneity_i2 >= 75:
                factors.inconsistency = 2
            elif heterogeneity_i2 >= 50:
                factors.inconsistency = 1

        if ci_width is not None and ci_width > 0.5:
            factors.imprecision = 1

        if publication_bias_detected:
            factors.publication_bias = 1

        # --------------------------------------------------
        # Upgrades (observational studies scenario). For RCTs, start High.
        # --------------------------------------------------
        if large_effect:
            factors.large_effect = 1
        if dose_response:
            factors.dose_response = 1

        # --------------------------------------------------
        # Compute final rating
        # --------------------------------------------------
        start_index = 0  # assume RCTs → High; in future differentiate study designs
        rating_index = start_index + factors.total_downgrade() - factors.total_upgrade()
        rating_index = min(max(rating_index, 0), len(GRADE_ORDER) - 1)
        rating = GRADE_ORDER[rating_index]

        return rating, factors

    # -----------------------------------------------------------------
    # Convenience helpers for reports
    # -----------------------------------------------------------------

    @staticmethod
    def generate_sof_table(outcomes: Dict[str, Dict[str, str]]) -> str:
        """Build Summary-of-Findings Markdown table.

        The *outcomes* dict maps outcome → {"effect": str, "certainty": str}
        """
        lines = [
            "| Outcome | Effect size / interpretation | Certainty |",
            "|---|---|:---:|",
        ]
        for outcome, data in outcomes.items():
            lines.append(f"| {outcome} | {data.get('effect','—')} | {data.get('certainty','—')} |")
        return "\n".join(lines) 