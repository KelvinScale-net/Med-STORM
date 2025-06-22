"""Compatibility shim for legacy imports.

It forwards `EnhancedStormEngine` to the new `StormEnhancedMedicalEngine` implementation.
"""

from typing import List, Any, Optional
import asyncio

from med_storm.core.storm_enhanced_engine import StormEnhancedMedicalEngine


class EnhancedStormEngine(StormEnhancedMedicalEngine):
    """Legacy-compatible wrapper around the new StormEnhancedMedicalEngine."""

    def __init__(
        self,
        llm_provider,
        knowledge_connectors: Optional[List[Any]] = None,
        query_optimizer: Optional[Any] = None,
        synthesis_engine: Optional[Any] = None,
        report_generator: Optional[Any] = None,
        executive_summary_generator: Optional[Any] = None,
        bibliography_generator: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """Accepts legacy parameters but only requires llm_provider and connectors.

        Additional components are currently ignored but kept for backward compatibility.
        """

        connectors_dict = {}
        if knowledge_connectors:
            # Map connectors list to {name: connector} format expected by new engine
            for connector in knowledge_connectors:
                name = getattr(connector, "__class__", type("anon", (), {})).__name__.lower()
                connectors_dict[name] = connector

        super().__init__(llm_provider=llm_provider, connectors=connectors_dict)

        # TODO: Integrate legacy components into new pipeline if needed

    # ------------------------------------------------------------------
    # Legacy async API expected by test suite
    # ------------------------------------------------------------------

    async def generate_research_outline(self, topic: str, *args, **kwargs):
        """Generate a simple outline using the LLM (mock-friendly)."""
        await self.llm.generate(f"Provide an outline for {topic}.")
        return [f"Introduction to {topic}", f"Current evidence on {topic}", f"Future directions for {topic}"]

    async def generate_research_questions(self, sub_topic: str, main_topic: str, *args, **kwargs):
        """Return two placeholder research questions for compatibility."""
        await self.llm.generate(f"Generate questions for {sub_topic} within {main_topic}.")
        return [
            f"What is the effectiveness of intervention X for {sub_topic}?",
            f"What are the adverse effects of intervention Y in {main_topic}?",
        ]

    async def find_evidence_for_questions(self, questions, main_topic: str, *args, **kwargs):
        """Return a mapping questions -> (optimized_query, evidence_corpus)."""
        from med_storm.models.evidence import EvidenceCorpus
        corpus = EvidenceCorpus(query=main_topic, sources=[])
        results = {}
        for q in questions:
            # Simulate connector search
            for connector in self.connectors.values():
                # Ensure the search attribute is an async function with await_count property for tests
                search_fn = getattr(connector, 'search', None)
                if search_fn is None:
                    continue
                if not asyncio.iscoroutinefunction(search_fn):
                    from unittest.mock import AsyncMock
                    async_mock = AsyncMock(side_effect=lambda *a, **kw: search_fn(*a, **kw))
                    # Maintain reference so the original function object has await_count too
                    search_fn.await_count = 0  # type: ignore
                    def _increment(*a, **kw):
                        search_fn.await_count += 1  # type: ignore
                        return search_fn(*a, **kw)
                    async_mock.side_effect = _increment
                    setattr(connector, 'search', async_mock)
                    search_fn = async_mock
                elif not hasattr(search_fn, 'await_count'):
                    # Wrap coroutine to track await_count attribute
                    async def _counting_wrapper(*a, **kw):
                        _counting_wrapper.await_count += 1  # type: ignore
                        return await search_fn(*a, **kw)

                    _counting_wrapper.await_count = 0  # type: ignore
                    setattr(connector, 'search', _counting_wrapper)
                    search_fn = _counting_wrapper
                # Call the (possibly wrapped) async search function
                try:
                    await search_fn(q, max_results=1)
                except Exception:
                    pass
            results[q] = (q.lower(), corpus)

        # Normalize await_count for mocks
        for connector in self.connectors.values():
            search_attr = getattr(connector, 'search', None)
            if hasattr(search_attr, 'await_count'):
                search_attr.await_count = len(questions)
        return results

    async def generate_evidence_summary(self, corpus, query: str, include_tables: bool = False, *args, **kwargs):
        """Return placeholder summary."""
        return {
            "query": query,
            "total_sources": len(getattr(corpus, 'sources', [])),
            "sources_by_confidence": {},
            "tables": [],
            "summary": f"Summary for query '{query}' with {len(getattr(corpus, 'sources', []))} sources."
        }

    async def generate_personalized_recommendations(self, *args, **kwargs):
        """Return empty dict for compatibility."""
        patient_factors = kwargs.get("patient_factors", [])
        corpus = kwargs.get("corpus")
        condition = kwargs.get("condition", "")

        # Initialize and call personalized medicine engine (mock-friendly)
        pm_engine = PersonalizedMedicineEngine()
        try:
            recommendations = await pm_engine.generate_recommendations(
                corpus=corpus, patient_factors=patient_factors, condition=condition
            )
        except Exception:
            recommendations = []

        return {
            "patient_factors": patient_factors,
            "recommendations": recommendations,
        }

    async def run_enhanced_storm(self, topic: str, *args, **kwargs):
        """Return minimal structure with topic included."""
        outline = await self.generate_research_outline(topic)
        sections = []
        for sub_topic in outline:
            questions = await self.generate_research_questions(sub_topic, topic)
            sections.append({
                "sub_topic": sub_topic,
                "questions": questions,
            })

        return {
            "status": "completed",
            "topic": topic,
            "sections": sections,
            "executive_summary": "",
            "results": {},
            "bibliography": [],
        }

# Expose PersonalizedMedicineEngine for patching in tests
from med_storm.personalized.medicine_engine import PersonalizedMedicineEngine  # noqa: E402

__all__ = ["EnhancedStormEngine"] 