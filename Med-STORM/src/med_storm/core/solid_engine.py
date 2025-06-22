"""
🏗️ SOLID CORE ENGINE
Core engine sólido que integra todos los componentes estándar
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..core.interfaces import (
    LLMProvider, KnowledgeConnector, ContentGenerator, 
    EvidenceSource, EvidenceCorpus, PerformanceMode
)
from ..llm.openrouter_provider import UltraFastLLMRouter
from ..connectors.standard_connectors import (
    StandardPubMedConnector, StandardSerperConnector, StandardLocalCorpusConnector
)
from ..synthesis.standard_content_generator import StandardMedicalContentGenerator

logger = logging.getLogger(__name__)


@dataclass
class SolidEngineConfig:
    """Configuration for solid engine"""
    openrouter_api_key: str
    deepseek_api_key: Optional[str] = None
    serper_api_key: Optional[str] = None
    pubmed_email: Optional[str] = None
    pubmed_api_key: Optional[str] = None
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    max_evidence_sources: int = 20
    enable_local_corpus: bool = True
    enable_pubmed: bool = True
    enable_serper: bool = True


@dataclass
class MedicalReport:
    """Standard medical report structure"""
    topic: str
    executive_summary: str
    main_content: str
    clinical_recommendations: str
    evidence_sources: List[EvidenceSource]
    performance_metrics: Dict[str, Any]
    quality_score: float
    generation_time: float


class SolidMedStormEngine:
    """
    🏗️ SOLID MED-STORM ENGINE
    
    Core engine que integra todos los componentes siguiendo interfaces estándar:
    - Ultra-fast LLM routing (OpenRouter + DeepSeek)
    - Multi-source evidence retrieval (PubMed + Serper + Local)
    - Professional content generation
    - Robust error handling and fallbacks
    """
    
    def __init__(self, config: SolidEngineConfig):
        self.config = config
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0.0,
            "evidence_retrieval_time": 0.0,
            "content_generation_time": 0.0
        }
        
        # Initialize components
        self._initialize_components()
        
        logger.info("🚀 Solid Med-STORM Engine initialized")
    
    def _initialize_components(self):
        """Initialize all engine components"""
        
        # 1. LLM Provider (OpenRouter primary, DeepSeek fallback)
        self.llm_provider = UltraFastLLMRouter(
            openrouter_api_key=self.config.openrouter_api_key,
            deepseek_api_key=self.config.deepseek_api_key
        )
        
        # 2. Knowledge Connectors
        self.connectors = []
        
        if self.config.enable_pubmed and self.config.pubmed_email:
            self.connectors.append(
                StandardPubMedConnector(
                    email=self.config.pubmed_email,
                    api_key=self.config.pubmed_api_key
                )
            )
            
        if self.config.enable_serper and self.config.serper_api_key:
            self.connectors.append(
                StandardSerperConnector(api_key=self.config.serper_api_key)
            )
            
        if self.config.enable_local_corpus:
            self.connectors.append(StandardLocalCorpusConnector())
        
        # 3. Content Generator
        self.content_generator = StandardMedicalContentGenerator(self.llm_provider)
        
        logger.info(f"✅ Initialized {len(self.connectors)} connectors")
    
    async def generate_medical_report(
        self, 
        topic: str, 
        **kwargs
    ) -> MedicalReport:
        """
        Generate comprehensive medical report
        
        Args:
            topic: Medical topic to research
            **kwargs: Additional configuration options
            
        Returns:
            MedicalReport: Complete medical report with all sections
        """
        start_time = time.time()
        self.performance_metrics["total_requests"] += 1
        
        try:
            logger.info(f"🔬 Starting medical report generation for: {topic}")
            
            # PHASE 1: Evidence Retrieval (Parallel)
            evidence_start = time.time()
            evidence_sources = await self._retrieve_evidence(topic)
            evidence_time = time.time() - evidence_start
            self.performance_metrics["evidence_retrieval_time"] = evidence_time
            
            logger.info(f"📚 Retrieved {len(evidence_sources)} evidence sources in {evidence_time:.2f}s")
            
            # PHASE 2: Content Generation (Parallel)
            content_start = time.time()
            
            # Generate all content sections in parallel
            content_tasks = [
                self.content_generator.generate_executive_summary(
                    topic, evidence_sources, self.config.performance_mode
                ),
                self.content_generator.generate_main_content(
                    topic, evidence_sources, self.config.performance_mode
                ),
                self.content_generator.generate_clinical_recommendations(
                    topic, evidence_sources, self.config.performance_mode
                )
            ]
            
            executive_summary, main_content, clinical_recommendations = await asyncio.gather(
                *content_tasks, return_exceptions=True
            )
            
            # Handle any exceptions from content generation
            if isinstance(executive_summary, Exception):
                logger.error(f"Executive summary generation failed: {executive_summary}")
                executive_summary = f"Executive summary generation failed for {topic}"
                
            if isinstance(main_content, Exception):
                logger.error(f"Main content generation failed: {main_content}")
                main_content = f"Main content generation failed for {topic}"
                
            if isinstance(clinical_recommendations, Exception):
                logger.error(f"Recommendations generation failed: {clinical_recommendations}")
                clinical_recommendations = f"Clinical recommendations generation failed for {topic}"
            
            content_time = time.time() - content_start
            self.performance_metrics["content_generation_time"] = content_time
            
            # PHASE 3: Quality Assessment
            quality_score = self._calculate_quality_score(
                executive_summary, main_content, clinical_recommendations, evidence_sources
            )
            
            # PHASE 4: Compile Report
            total_time = time.time() - start_time
            
            report = MedicalReport(
                topic=topic,
                executive_summary=executive_summary,
                main_content=main_content,
                clinical_recommendations=clinical_recommendations,
                evidence_sources=evidence_sources,
                performance_metrics={
                    "evidence_retrieval_time": evidence_time,
                    "content_generation_time": content_time,
                    "total_generation_time": total_time,
                    "evidence_sources_count": len(evidence_sources),
                    "llm_provider_stats": self.llm_provider.get_performance_stats()
                },
                quality_score=quality_score,
                generation_time=total_time
            )
            
            self.performance_metrics["successful_requests"] += 1
            self.performance_metrics["average_response_time"] = (
                (self.performance_metrics["average_response_time"] * (self.performance_metrics["successful_requests"] - 1) + total_time) /
                self.performance_metrics["successful_requests"]
            )
            
            logger.info(f"✅ Medical report generated successfully in {total_time:.2f}s (Quality: {quality_score:.1f}/100)")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Medical report generation failed: {e}")
            
            # Return fallback report
            return MedicalReport(
                topic=topic,
                executive_summary=f"Report generation failed for {topic}: {str(e)}",
                main_content="Content generation failed due to system error.",
                clinical_recommendations="Recommendations unavailable due to system error.",
                evidence_sources=[],
                performance_metrics={"error": str(e)},
                quality_score=0.0,
                generation_time=time.time() - start_time
            )
    
    async def _retrieve_evidence(self, topic: str) -> List[EvidenceSource]:
        """Retrieve evidence from all available connectors in parallel"""
        
        if not self.connectors:
            logger.warning("No connectors available for evidence retrieval")
            return []
        
        # Run all connectors in parallel
        connector_tasks = []
        for connector in self.connectors:
            task = self._safe_connector_search(connector, topic)
            connector_tasks.append(task)
        
        # Wait for all connectors to complete
        results = await asyncio.gather(*connector_tasks, return_exceptions=True)
        
        # Combine all evidence sources
        all_evidence = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Connector {i} failed: {result}")
                continue
            
            if isinstance(result, EvidenceCorpus):
                all_evidence.extend(result.sources)
            elif isinstance(result, list):
                all_evidence.extend(result)
        
        # Deduplicate and limit results
        unique_evidence = self._deduplicate_evidence(all_evidence)
        limited_evidence = unique_evidence[:self.config.max_evidence_sources]
        
        return limited_evidence
    
    async def _safe_connector_search(
        self, 
        connector: KnowledgeConnector, 
        topic: str
    ) -> EvidenceCorpus:
        """Safely search with a connector with timeout and error handling"""
        try:
            # Set timeout based on performance mode
            timeout = 60 if self.config.performance_mode == PerformanceMode.GOLD_STANDARD else 30
            
            corpus = await asyncio.wait_for(
                connector.search(topic, max_results=10),
                timeout=timeout
            )
            
            return corpus
            
        except asyncio.TimeoutError:
            logger.warning(f"Connector {type(connector).__name__} timed out")
            return EvidenceCorpus(query=topic, sources=[], total_found=0, retrieval_time=0, source_breakdown={})
        except Exception as e:
            logger.error(f"Connector {type(connector).__name__} failed: {e}")
            return EvidenceCorpus(query=topic, sources=[], total_found=0, retrieval_time=0, source_breakdown={})
    
    def _deduplicate_evidence(self, evidence: List[EvidenceSource]) -> List[EvidenceSource]:
        """Remove duplicate evidence sources"""
        seen_titles = set()
        unique_evidence = []
        
        for source in evidence:
            # Simple deduplication based on title similarity
            title_key = source.title.lower().strip()
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_evidence.append(source)
        
        # Sort by confidence score (highest first)
        unique_evidence.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return unique_evidence
    
    def _calculate_quality_score(
        self, 
        executive_summary: str, 
        main_content: str, 
        recommendations: str, 
        evidence: List[EvidenceSource]
    ) -> float:
        """Calculate quality score for the generated report"""
        
        score = 0.0
        
        # Content quality (40 points)
        if len(executive_summary) > 200:
            score += 15
        if len(main_content) > 500:
            score += 15
        if len(recommendations) > 200:
            score += 10
        
        # Evidence quality (30 points)
        if len(evidence) > 0:
            score += 10
        if len(evidence) >= 5:
            score += 10
        if len([e for e in evidence if e.confidence_score >= 0.8]) > 0:
            score += 10
        
        # Technical quality (30 points)
        if "failed" not in executive_summary.lower():
            score += 10
        if "failed" not in main_content.lower():
            score += 10
        if "failed" not in recommendations.lower():
            score += 10
        
        return min(score, 100.0)
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of all components"""
        
        health_status = {
            "engine_status": "healthy",
            "llm_provider": await self.llm_provider.health_check(),
            "connectors": {},
            "overall_healthy": True
        }
        
        # Check each connector
        for i, connector in enumerate(self.connectors):
            connector_name = type(connector).__name__
            try:
                is_healthy = await asyncio.wait_for(
                    connector.health_check(), 
                    timeout=10
                )
                health_status["connectors"][connector_name] = is_healthy
                if not is_healthy:
                    health_status["overall_healthy"] = False
            except:
                health_status["connectors"][connector_name] = False
                health_status["overall_healthy"] = False
        
        return health_status
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        stats = self.performance_metrics.copy()
        stats["llm_provider_stats"] = self.llm_provider.get_performance_stats()
        stats["connectors_count"] = len(self.connectors)
        stats["success_rate"] = (
            self.performance_metrics["successful_requests"] / 
            max(self.performance_metrics["total_requests"], 1)
        )
        
        return stats 