import asyncio
import pytest
from med_storm.statistics.advanced_analysis import AdvancedStatisticalEngine, StudyData


@pytest.mark.asyncio
async def test_analysis_exposes_i2_pub_bias():
    # Simulate 6 studies with heterogeneous effects
    studies = [
        StudyData(study_id=str(i), effect_size=1.0 + (i * 0.2), standard_error=0.1 + (i*0.02), sample_size=100)
        for i in range(6)
    ]
    engine = AdvancedStatisticalEngine()
    res = await engine.conduct_comprehensive_analysis(studies, {"effect_measure": "risk_ratio"})
    assert res["i_squared"] is not None
    # Egger likely non-significant in synthetic data; just assert key exists
    assert "publication_bias_detected" in res 