"""recommender.py
~~~~~~~~~~~~~~~~~~~
Simple rule-based personalized recommender that converts a PatientProfile
plus a list of candidate interventions into ranked `PersonalizedRecommendation`
objects.

This module is intended as a lightweight bridge until a full probabilistic
engine is available.  It focuses on critical pharmacogenomic and clinical
edge-cases documented in guidelines (e.g., CPIC).
"""
from __future__ import annotations

from typing import List
from med_storm.personalized.medicine_engine import (
    PatientProfile,
    PersonalizedRecommendation,
    GeneticVariant,
)


class PersonalizedRecommender:  # pragma: no cover – thin orchestrator
    """Generate personalized therapeutic suggestions."""

    def generate(
        self,
        *,
        patient: PatientProfile,
        medical_condition: str,
        interventions: List[str],
    ) -> List[PersonalizedRecommendation]:
        """Return sorted personalized recommendations."""
        recs: List[PersonalizedRecommendation] = []
        for intervention in interventions:
            recs.append(self._evaluate_intervention(patient, intervention, medical_condition))

        # Rank by composite of efficacy_prediction – safety first
        recs.sort(key=lambda r: (-(r.safety_score), -(r.efficacy_prediction)))
        return recs

    # ------------------------------------------------------------------
    # Internal helpers – heuristic rules only
    # ------------------------------------------------------------------

    def _evaluate_intervention(self, patient: PatientProfile, intervention: str, medical_condition: str) -> PersonalizedRecommendation:
        # Defaults
        dosage: str | None = None
        monitor: List[str] = []
        contraindications: List[str] = []
        efficacy: float = 0.5
        safety: float = 0.8
        rationale_blurbs: List[str] = []

        # Age-based adjustments
        if patient.age > 65 and intervention.lower() in {"metformin", "tramadol", "codeine"}:
            dosage = "Reduce initial dose by 50%"
            monitor.append("Renal function every 3 months")
            rationale_blurbs.append("Elderly patient – reduced clearance expected")

        # Renal function via biomarker
        if intervention.lower() == "metformin" and patient.biomarkers.get("creatinine", 0) > 1.5:
            contraindications.append("Contraindicated – renal impairment")
            safety = 0.2
            rationale_blurbs.append("Creatinine >1.5 mg/dL increases risk of lactic acidosis")

        # Pharmacogenomics CYP2D6
        if intervention.lower() in {"codeine", "tramadol"}:
            if GeneticVariant.CYP2D6_POOR in patient.genetic_variants:
                contraindications.append("Avoid – CYP2D6 poor metabolizer → toxicity")
                safety = 0.1
                efficacy = 0.3
                rationale_blurbs.append("CYP2D6 poor metabolizer – risk of accumulation")
            elif GeneticVariant.CYP2D6_ULTRA in patient.genetic_variants:
                contraindications.append("Avoid – CYP2D6 ultra-rapid → lack of efficacy")
                safety = 0.4
                efficacy = 0.2
                rationale_blurbs.append("Ultra-rapid metabolizer – subtherapeutic effect")

        # Basic efficacy heuristic
        if medical_condition.lower() in intervention.lower():
            efficacy += 0.2  # alignment bump

        return PersonalizedRecommendation(
            intervention=intervention,
            dosage_adjustment=dosage,
            monitoring_requirements=monitor,
            contraindications=contraindications,
            efficacy_prediction=min(efficacy, 1.0),
            safety_score=min(safety, 1.0),
            evidence_level="Rule-based heuristic",
            rationale="; ".join(rationale_blurbs) or "Rule-based assessment",
        ) 