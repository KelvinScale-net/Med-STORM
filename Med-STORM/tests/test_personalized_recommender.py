from med_storm.personalized.recommender import PersonalizedRecommender
from med_storm.personalized.medicine_engine import PatientProfile, EthnicityGroup, GeneticVariant


def _base_profile():
    return PatientProfile(
        age=45,
        sex="male",
        ethnicity=EthnicityGroup.CAUCASIAN,
        weight=80.0,
        height=180.0,
        comorbidities=["hypertension"],
        medications=[],
        genetic_variants=[],
        biomarkers={"creatinine": 1.0},
        lifestyle_factors={},
    )


def test_cyp2d6_poor_avoids_codeine():
    profile = _base_profile()
    profile.genetic_variants.append(GeneticVariant.CYP2D6_POOR)
    recs = PersonalizedRecommender().generate(
        patient=profile,
        medical_condition="pain",
        interventions=["Codeine", "Ibuprofen"],
    )
    codeine_rec = next(r for r in recs if r.intervention.lower() == "codeine")
    assert any("avoid" in c.lower() for c in codeine_rec.contraindications)
    assert codeine_rec.safety_score < 0.5


def test_renal_impairment_metformin_contra():
    profile = _base_profile()
    profile.biomarkers["creatinine"] = 1.8
    recs = PersonalizedRecommender().generate(
        patient=profile,
        medical_condition="diabetes",
        interventions=["Metformin"],
    )
    met_rec = recs[0]
    assert "contraindicated" in ";".join(met_rec.contraindications).lower() 