from med_storm.quality.grade_evaluator import GradeEvaluator
from med_storm.statistics.advanced_analysis import StudyData


def test_grade_evaluator_high_quality():
    studies = [StudyData(study_id=str(i), effect_size=0.8, standard_error=0.1, sample_size=100) for i in range(5)]
    rob = [{"overall_bias": "Low risk"} for _ in range(5)]
    rating, factors = GradeEvaluator().evaluate(studies, rob)
    assert rating == "High"
    assert factors.total_downgrade() == 0


def test_grade_evaluator_downgrade_bias():
    studies = [StudyData(study_id=str(i), effect_size=0.8, standard_error=0.1, sample_size=100) for i in range(3)]
    rob = [
        {"overall_bias": "High risk"},
        {"overall_bias": "High risk"},
        {"overall_bias": "Low risk"},
    ]
    rating, factors = GradeEvaluator().evaluate(studies, rob)
    assert rating in ["Moderate", "Low", "Very low"]  # should downgrade
    assert factors.risk_of_bias >= 1


def test_generate_sof_table():
    outcomes = {
        "Mortality": {"effect": "RR 0.80 (95% CI 0.70-0.92)", "certainty": "High"},
        "Hospital stay": {"effect": "MD -2.1 days", "certainty": "Moderate"},
    }
    md = GradeEvaluator.generate_sof_table(outcomes)
    assert "Mortality" in md and "Hospital stay" in md
    assert md.count("|") > 5 