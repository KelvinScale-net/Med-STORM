"""
Core components of the Med-STORM application.
"""

# Make key classes available for easier importing
from .engine import StormEngine
from .evidence_manager import EvidenceManager
from .query_optimizer import QueryOptimizer
from .storm_enhanced_engine import StormEnhancedMedicalEngine

# Backward compatibility alias
EnhancedStormEngine = StormEnhancedMedicalEngine

__all__ = [
    "StormEngine",
    "EvidenceManager",
    "QueryOptimizer",
    "EnhancedStormEngine",
]