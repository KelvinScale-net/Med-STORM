"""
🔧 CORE INTERFACES - STANDARD CONTRACTS
Interfaces estándar para evitar inconsistencias de métodos y parámetros
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import time


class PerformanceMode(Enum):
    """Performance modes for different use cases"""
    ULTRA_FAST = "ultra_fast"      # 3-5 min, basic quality
    BALANCED = "balanced"          # 8-12 min, good quality  
    GOLD_STANDARD = "gold"         # 15-20 min, excellent quality


@dataclass
class EvidenceSource:
    """Standard evidence source structure"""
    title: str
    summary: str
    url: str
    source_type: str  # pubmed, serper, local_corpus
    confidence_score: float
    publication_date: Optional[str] = None
    authors: Optional[List[str]] = None
    pmid: Optional[str] = None
    doi: Optional[str] = None


@dataclass
class EvidenceCorpus:
    """Standard evidence corpus structure"""
    query: str
    sources: List[EvidenceSource]
    total_found: int
    retrieval_time: float
    source_breakdown: Dict[str, int]


@dataclass
class ResearchResult:
    """Standard research result structure"""
    topic: str
    executive_summary: str
    content: str
    evidence_sources: List[EvidenceSource]
    quality_score: float
    execution_time: float
    methodology: str
    citations: List[str]


class LLMProvider(ABC):
    """Standard LLM Provider Interface"""
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Standard generate method - NO generate_response variants"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        pass


class KnowledgeConnector(ABC):
    """Standard Knowledge Connector Interface"""
    
    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs
    ) -> EvidenceCorpus:
        """Standard search method returning EvidenceCorpus"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if connector is healthy"""
        pass


class ContentGenerator(ABC):
    """Standard Content Generator Interface"""
    
    @abstractmethod
    async def generate_executive_summary(
        self,
        topic: str,
        evidence: List[EvidenceSource],
        **kwargs
    ) -> str:
        """Generate executive summary"""
        pass
    
    @abstractmethod
    async def generate_main_content(
        self,
        topic: str,
        evidence: List[EvidenceSource],
        **kwargs
    ) -> str:
        """Generate main research content"""
        pass


class MedStormEngine(ABC):
    """Standard Med-STORM Engine Interface"""
    
    @abstractmethod
    async def research(
        self,
        topic: str,
        performance_mode: PerformanceMode = PerformanceMode.BALANCED,
        max_sources: int = 20,
        **kwargs
    ) -> ResearchResult:
        """Main research method"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all components"""
        pass


@dataclass
class PerformanceMetrics:
    """Standard performance tracking"""
    start_time: float
    end_time: Optional[float] = None
    evidence_retrieval_time: float = 0.0
    content_generation_time: float = 0.0
    total_sources: int = 0
    quality_score: float = 0.0
    
    def mark_complete(self):
        """Mark completion and calculate total time"""
        self.end_time = time.time()
    
    @property
    def total_time(self) -> float:
        """Get total execution time"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self):
        self.performance_configs = {
            PerformanceMode.ULTRA_FAST: {
                "max_sources": 10,
                "max_tokens": 500,
                "timeout": 180,  # 3 min
                "enable_process_rewards": False,
                "enable_systematic_review": False
            },
            PerformanceMode.BALANCED: {
                "max_sources": 20,
                "max_tokens": 1000,
                "timeout": 600,  # 10 min
                "enable_process_rewards": False,
                "enable_systematic_review": False
            },
            PerformanceMode.GOLD_STANDARD: {
                "max_sources": 50,
                "max_tokens": 2000,
                "timeout": 1200,  # 20 min
                "enable_process_rewards": True,
                "enable_systematic_review": True
            }
        }
    
    def get_config(self, mode: PerformanceMode) -> Dict[str, Any]:
        """Get configuration for performance mode"""
        return self.performance_configs[mode].copy()


# Global configuration instance
config_manager = ConfigManager() 