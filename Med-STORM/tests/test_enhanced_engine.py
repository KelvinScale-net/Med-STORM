"""
Tests for the enhanced Med-STORM engine.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the class we're testing
from src.med_storm.core.engine_enhanced import EnhancedStormEngine
from src.med_storm.models.evidence import EvidenceSourceType
from src.med_storm.core.evidence_manager import EvidenceTier

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Mock LLMProvider
class MockLLMProvider:
    """Mock LLM provider for testing."""
    async def generate(self, prompt: str, **kwargs) -> str:
        return "Mocked LLM response"
    
    async def generate_structured(self, prompt: str, response_model: Dict, **kwargs) -> Dict:
        return {"result": "mocked structured response"}

# Mock KnowledgeConnector
class MockKnowledgeConnector:
    """Mock KnowledgeConnector for testing."""
    async def search(self, query: str) -> Dict[str, Any]:
        return {
            "query": query,
            "results": [{"title": f"Result for {query}", "snippet": "Sample snippet"}]
        }

# Mock QueryOptimizer
class MockQueryOptimizer:
    """Mock QueryOptimizer for testing."""
    async def optimize(self, query: str) -> str:
        return f"Optimized: {query}"

# Mock SynthesisEngine
class MockSynthesisEngine:
    """Mock SynthesisEngine for testing."""
    async def synthesize_evidence(self, evidence: List[Dict[str, Any]]) -> str:
        return "Synthesized evidence summary"

# Mock ReportGenerator
class MockReportGenerator:
    """Mock ReportGenerator for testing."""
    async def generate_report(self, content: str, **kwargs) -> str:
        return f"Generated report for: {content}"

# Mock ExecutiveSummaryGenerator
class MockExecutiveSummaryGenerator:
    """Mock ExecutiveSummaryGenerator for testing."""
    async def generate_summary(self, content: str, **kwargs) -> str:
        return f"Generated summary for: {content}"

# Mock BibliographyGenerator
class MockBibliographyGenerator:
    """Mock BibliographyGenerator for testing."""
    async def generate_bibliography(self, sources: List[Dict[str, Any]], **kwargs) -> str:
        return "Generated bibliography"

# Define test data classes first to avoid import issues
class MockEvidenceSource:
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.title = kwargs.get('title', '')
        self.abstract = kwargs.get('abstract', '')
        self.url = kwargs.get('url', '')
        self.source_type = kwargs.get('source_type', '')
        self.publication_date = kwargs.get('publication_date', '')
        self.authors = kwargs.get('authors', [])
        self.journal = kwargs.get('journal', '')
        self.doi = kwargs.get('doi', '')
        self.confidence_score = kwargs.get('confidence_score', 0.9)  # Default to high confidence
        self.metadata = kwargs.get('metadata', {})
        self.summary = kwargs.get('summary', '')

class MockEvidenceCorpus:
    def __init__(self, **kwargs):
        self.query = kwargs.get('query', '')
        self.sources = kwargs.get('sources', [])

# Import for type checking only
if TYPE_CHECKING:
    from med_storm.core.engine_enhanced import EnhancedStormEngine
    from med_storm.llm.base import LLMProvider
    from med_storm.connectors.base import KnowledgeConnector
    from med_storm.models.evidence import EvidenceCorpus, EvidenceSource
    from med_storm.core.query_optimizer import QueryOptimizer
    from med_storm.synthesis.engine import SynthesisEngine
    from med_storm.synthesis.report_generator import ReportGenerator
    from med_storm.synthesis.executive_summary_generator import ExecutiveSummaryGenerator
    from med_storm.synthesis.bibliography_generator import BibliographyGenerator

# Import for runtime
try:
    from med_storm.models.evidence import EvidenceSource, EvidenceCorpus
    from med_storm.core.engine_enhanced import EnhancedStormEngine
    from med_storm.llm.base import LLMProvider
    from med_storm.connectors.base import KnowledgeConnector
    from med_storm.core.query_optimizer import QueryOptimizer
    from med_storm.synthesis.engine import SynthesisEngine
    from med_storm.synthesis.report_generator import ReportGenerator
    from med_storm.synthesis.executive_summary_generator import ExecutiveSummaryGenerator
    from med_storm.synthesis.bibliography_generator import BibliographyGenerator
except ImportError:
    # Use mock classes if imports fail during test collection
    EvidenceSource = MockEvidenceSource
    EvidenceCorpus = MockEvidenceCorpus

# Test data
SAMPLE_EVIDENCE = EvidenceSource(
    title="Test Evidence",
    url="http://example.com",
    source_name="Test Journal",
    source_type=EvidenceSourceType.RESEARCH_PAPER,
    tier=1,  # Using the integer value for TIER_1
    summary="This is a test evidence abstract.",
    published_date="2023-01-01",
    authors=["Author 1", "Author 2"],
    doi="10.1234/test.2023.1",
    study_type="randomized_controlled_trial",
    sample_size=100,
    confidence_score=0.9,
    raw_metadata={"study_type": "randomized_controlled_trial"}
)

SAMPLE_CORPUS = EvidenceCorpus(
    query="test query",
    sources=[SAMPLE_EVIDENCE]
)

# Fixtures
@pytest.fixture
def mock_llm_provider():
    provider = MockLLMProvider()
    provider.generate = AsyncMock(return_value="Test response")
    provider.generate_structured = AsyncMock(return_value={"result": "mocked structured response"})
    return provider

@pytest.fixture
def mock_knowledge_connector():
    return MockKnowledgeConnector()

@pytest.fixture
def mock_query_optimizer():
    return MockQueryOptimizer()

@pytest.fixture
def mock_synthesis_engine():
    return MockSynthesisEngine()

@pytest.fixture
def mock_report_generator():
    return MockReportGenerator()

@pytest.fixture
def mock_summary_generator():
    return MockExecutiveSummaryGenerator()

@pytest.fixture
def mock_bibliography_generator():
    return MockBibliographyGenerator()

@pytest.fixture
def enhanced_engine(
    mock_llm_provider,
    mock_knowledge_connector,
    mock_query_optimizer,
    mock_synthesis_engine,
    mock_report_generator,
    mock_summary_generator,
    mock_bibliography_generator
):
    return EnhancedStormEngine(
        llm_provider=mock_llm_provider,
        knowledge_connectors=[mock_knowledge_connector],
        query_optimizer=mock_query_optimizer,
        synthesis_engine=mock_synthesis_engine,
        report_generator=mock_report_generator,
        executive_summary_generator=mock_summary_generator,
        bibliography_generator=mock_bibliography_generator
    )

# Tests
@pytest.mark.asyncio
async def test_generate_research_outline(enhanced_engine):
    """Test generating a research outline."""
    topic = "Test Topic"
    outline = await enhanced_engine.generate_research_outline(topic)
    
    assert isinstance(outline, list)
    enhanced_engine.llm.generate.assert_awaited_once()

@pytest.mark.asyncio
async def test_generate_research_questions(enhanced_engine):
    """Test generating research questions."""
    questions = await enhanced_engine.generate_research_questions("Sub Topic", "Main Topic")
    
    assert isinstance(questions, list)
    enhanced_engine.llm.generate.assert_awaited_once()

@pytest.mark.asyncio
async def test_find_evidence_for_questions(enhanced_engine, mock_knowledge_connector):
    """Test finding evidence for questions."""
    questions = ["Question 1", "Question 2"]
    evidence_map = await enhanced_engine.find_evidence_for_questions(
        questions=questions,
        main_topic="Main Topic"
    )
    
    assert isinstance(evidence_map, dict)
    assert len(evidence_map) == len(questions)
    assert mock_knowledge_connector.search.await_count == len(questions)

@pytest.mark.asyncio
async def test_generate_evidence_summary(enhanced_engine):
    """Test generating evidence summary."""
    summary = await enhanced_engine.generate_evidence_summary(
        corpus=SAMPLE_CORPUS,
        query="test query",
        include_tables=True
    )
    
    assert isinstance(summary, dict)
    assert 'query' in summary
    assert 'total_sources' in summary
    assert 'sources_by_confidence' in summary
    assert 'tables' in summary

@pytest.mark.asyncio
async def test_generate_personalized_recommendations(enhanced_engine):
    """Test generating personalized recommendations."""
    patient_factors = [
        {"name": "age", "value": "65", "type": "demographic"},
        {"name": "diabetes", "value": "Type 2", "type": "comorbidity"}
    ]
    
    with patch('med_storm.core.engine_enhanced.PersonalizedMedicineEngine') as mock_pm_engine:
        mock_instance = mock_pm_engine.return_value
        mock_instance.generate_recommendations = AsyncMock(return_value=[])
        
        recommendations = await enhanced_engine.generate_personalized_recommendations(
            corpus=SAMPLE_CORPUS,
            patient_factors=patient_factors,
            condition="Test Condition"
        )
        
        assert isinstance(recommendations, dict)
        assert 'patient_factors' in recommendations
        assert 'recommendations' in recommendations
        mock_instance.generate_recommendations.assert_awaited_once()

@pytest.mark.asyncio
async def test_run_enhanced_storm(enhanced_engine):
    """Test running the full enhanced storm process."""
    # Mock the outline generation to return a simple outline
    enhanced_engine.generate_research_outline = AsyncMock(return_value=["Sub Topic 1", "Sub Topic 2"])
    
    # Mock the question generation
    enhanced_engine.generate_research_questions = AsyncMock(return_value=["Question 1", "Question 2"])
    
    # Mock the evidence finding
    enhanced_engine.find_evidence_for_questions = AsyncMock(return_value={
        "Question 1": ("optimized query 1", SAMPLE_CORPUS),
        "Question 2": ("optimized query 2", SAMPLE_CORPUS)
    })
    
    # Run the enhanced storm
    results = await enhanced_engine.run_enhanced_storm(
        topic="Test Topic",
        patient_factors=[{"name": "age", "value": "65", "type": "demographic"}],
        max_concurrent_subtopics=2
    )
    
    # Verify the results
    assert isinstance(results, dict)
    assert 'topic' in results
    assert 'sections' in results
    assert len(results['sections']) == 2  # Should have sections for both sub-topics
    assert 'executive_summary' in results
    assert 'bibliography' in results
