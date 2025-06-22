"""risk_of_bias.py
-------------------
Minimal Risk-of-Bias assessor implementing the core RoB-2 domains.
Returns a dict per study and an overall judgement (low / some concerns / high).
Intended to be a placeholder until full NLP+rule implementation.
"""
from __future__ import annotations

from typing import Dict, List

from med_storm.models.evidence import EvidenceSource

_DOMAINS = [
    "randomization_process",
    "deviations_from_intended_interventions",
    "missing_outcome_data",
    "measurement_of_outcome",
    "selection_of_reported_result",
]


class RiskOfBiasModule:
    """Quick RoB-2 scorer with heuristics on abstract text length and RCT keywords."""

    def assess(self, study: EvidenceSource) -> Dict[str, str]:
        text = (study.summary or "")[:1000].lower()
        scores: Dict[str, str] = {}
        for domain in _DOMAINS:
            if "randomized" in text or "randomised" in text:
                scores[domain] = "Low risk"
            else:
                scores[domain] = "Some concerns"
        # Overall
        overall = "High risk" if list(scores.values()).count("Some concerns") > 2 else "Low risk"
        scores["overall_bias"] = overall
        return scores

    def batch_assess(self, studies: List[EvidenceSource]):
        return [self.assess(s) for s in studies] 