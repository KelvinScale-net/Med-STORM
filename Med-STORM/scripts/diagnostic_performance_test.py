#!/usr/bin/env python3
"""
🔬 DIAGNOSTIC PERFORMANCE TEST
Pruebas unitarias para identificar cuellos de botella específicos
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, List
import sys
import json

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from med_storm.llm.llm_router import get_llm_router
from med_storm.connectors.pubmed import PubMedConnector
from med_storm.connectors.serper import SerperConnector
from med_storm.connectors.local_corpus import LocalCorpusConnector
from med_storm.synthesis.intelligent_content_generator import IntelligentContentGenerator
from med_storm.evidence.systematic_review_engine import SystematicReviewEngine
from med_storm.evidence.evidence_grading import EvidenceGradingSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComponentPerformanceDiagnostic:
    """🔬 Diagnóstico de rendimiento por componente"""
    
    def __init__(self):
        """Initialize diagnostic suite"""
        self.results = {}
        self.test_topic = "Type 2 Diabetes Treatment"
        
    async def run_full_diagnostic(self) -> Dict[str, Any]:
        """Ejecutar diagnóstico completo de todos los componentes"""
        logger.info("🔬 Iniciando Diagnóstico Completo de Rendimiento")
        
        # Test 1: LLM Router Performance
        await self._test_llm_router_performance()
        
        # Test 2: Connectors Performance
        await self._test_connectors_performance()
        
        # Test 3: Content Generator Performance
        await self._test_content_generator_performance()
        
        # Test 4: Evidence Systems Performance
        await self._test_evidence_systems_performance()
        
        # Test 5: Integration Performance
        await self._test_integration_performance()
        
        # Analyze and report
        self._analyze_bottlenecks()
        
        return self.results
    
    async def _test_llm_router_performance(self):
        """Test LLM Router con diferentes cargas"""
        logger.info("🧠 Testing LLM Router Performance...")
        
        try:
            router = get_llm_router()
            
            # Test 1: Single short request
            start_time = time.time()
            short_response = await router.generate("Define diabetes briefly.", max_tokens=50)
            short_time = time.time() - start_time
            
            # Test 2: Single medium request
            start_time = time.time()
            medium_response = await router.generate(
                "Explain the pathophysiology of type 2 diabetes in detail.", 
                max_tokens=200
            )
            medium_time = time.time() - start_time
            
            # Test 3: Single long request
            start_time = time.time()
            long_response = await router.generate(
                "Provide a comprehensive analysis of type 2 diabetes including causes, symptoms, diagnosis, and treatment options.", 
                max_tokens=500
            )
            long_time = time.time() - start_time
            
            # Test 4: Parallel requests
            start_time = time.time()
            parallel_tasks = [
                router.generate("What is diabetes?", max_tokens=50),
                router.generate("What is hypertension?", max_tokens=50),
                router.generate("What is asthma?", max_tokens=50)
            ]
            parallel_responses = await asyncio.gather(*parallel_tasks, return_exceptions=True)
            parallel_time = time.time() - start_time
            
            # Calculate metrics
            health = router.get_health_status()
            
            self.results["llm_router"] = {
                "short_request_time": short_time,
                "medium_request_time": medium_time,
                "long_request_time": long_time,
                "parallel_time": parallel_time,
                "parallel_success_count": sum(1 for r in parallel_responses if not isinstance(r, Exception)),
                "health_status": health,
                "performance_rating": self._rate_performance("llm", [short_time, medium_time, long_time, parallel_time])
            }
            
            logger.info(f"✅ LLM Router - Short: {short_time:.2f}s, Medium: {medium_time:.2f}s, Long: {long_time:.2f}s, Parallel: {parallel_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ LLM Router test failed: {e}")
            self.results["llm_router"] = {"error": str(e), "performance_rating": "FAILED"}
    
    async def _test_connectors_performance(self):
        """Test performance de cada connector individualmente"""
        logger.info("🔗 Testing Connectors Performance...")
        
        connector_results = {}
        
        # Test PubMed Connector
        try:
            start_time = time.time()
            pubmed = PubMedConnector()
            pubmed_corpus = await pubmed.search(self.test_topic, max_results=5)
            pubmed_time = time.time() - start_time
            
            connector_results["pubmed"] = {
                "search_time": pubmed_time,
                "results_count": len(pubmed_corpus.sources) if pubmed_corpus else 0,
                "performance_rating": self._rate_performance("connector", [pubmed_time])
            }
            logger.info(f"✅ PubMed - {pubmed_time:.2f}s, {len(pubmed_corpus.sources) if pubmed_corpus else 0} results")
            
        except Exception as e:
            logger.error(f"❌ PubMed test failed: {e}")
            connector_results["pubmed"] = {"error": str(e), "performance_rating": "FAILED"}
        
        # Test Serper Connector
        try:
            start_time = time.time()
            serper = SerperConnector()
            serper_corpus = await serper.search(self.test_topic, max_results=5)
            serper_time = time.time() - start_time
            
            connector_results["serper"] = {
                "search_time": serper_time,
                "results_count": len(serper_corpus.sources) if serper_corpus else 0,
                "performance_rating": self._rate_performance("connector", [serper_time])
            }
            logger.info(f"✅ Serper - {serper_time:.2f}s, {len(serper_corpus.sources) if serper_corpus else 0} results")
            
        except Exception as e:
            logger.error(f"❌ Serper test failed: {e}")
            connector_results["serper"] = {"error": str(e), "performance_rating": "FAILED"}
        
        # Test Local Corpus Connector
        try:
            start_time = time.time()
            local_corpus = LocalCorpusConnector()
            local_corpus_result = await local_corpus.search(self.test_topic, max_results=5)
            local_time = time.time() - start_time
            
            connector_results["local_corpus"] = {
                "search_time": local_time,
                "results_count": len(local_corpus_result.sources) if local_corpus_result else 0,
                "performance_rating": self._rate_performance("connector", [local_time])
            }
            logger.info(f"✅ Local Corpus - {local_time:.2f}s, {len(local_corpus_result.sources) if local_corpus_result else 0} results")
            
        except Exception as e:
            logger.error(f"❌ Local Corpus test failed: {e}")
            connector_results["local_corpus"] = {"error": str(e), "performance_rating": "FAILED"}
        
        self.results["connectors"] = connector_results
    
    async def _test_content_generator_performance(self):
        """Test Intelligent Content Generator"""
        logger.info("📝 Testing Content Generator Performance...")
        
        try:
            router = get_llm_router()
            generator = IntelligentContentGenerator(router)
            
            # Test single section generation
            start_time = time.time()
            test_content = await generator._generate_executive_summary(
                self.test_topic, 
                {"complexity": "moderate", "medical_domain": "endocrinology"}, 
                []
            )
            single_section_time = time.time() - start_time
            
            # Test topic analysis
            start_time = time.time()
            topic_analysis = await generator._analyze_topic(self.test_topic)
            analysis_time = time.time() - start_time
            
            self.results["content_generator"] = {
                "single_section_time": single_section_time,
                "topic_analysis_time": analysis_time,
                "content_length": len(test_content) if test_content else 0,
                "performance_rating": self._rate_performance("content", [single_section_time, analysis_time])
            }
            
            logger.info(f"✅ Content Generator - Section: {single_section_time:.2f}s, Analysis: {analysis_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Content Generator test failed: {e}")
            self.results["content_generator"] = {"error": str(e), "performance_rating": "FAILED"}
    
    async def _test_evidence_systems_performance(self):
        """Test Evidence Systems Performance"""
        logger.info("🔬 Testing Evidence Systems Performance...")
        
        evidence_results = {}
        
        # Test Systematic Review Engine
        try:
            start_time = time.time()
            review_engine = SystematicReviewEngine()
            # Test PICO framework creation
            pico_result = review_engine._create_pico_framework(self.test_topic)
            pico_time = time.time() - start_time
            
            evidence_results["systematic_review"] = {
                "pico_creation_time": pico_time,
                "pico_components": len(pico_result.__dict__) if pico_result else 0,
                "performance_rating": self._rate_performance("evidence", [pico_time])
            }
            logger.info(f"✅ Systematic Review - PICO: {pico_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Systematic Review test failed: {e}")
            evidence_results["systematic_review"] = {"error": str(e), "performance_rating": "FAILED"}
        
        # Test Evidence Grading System
        try:
            start_time = time.time()
            grading_system = EvidenceGradingSystem()
            # Test grading with dummy evidence
            dummy_evidence = {"title": "Test Study", "abstract": "Test abstract", "study_type": "RCT"}
            grades = grading_system.grade_evidence_comprehensive([dummy_evidence])
            grading_time = time.time() - start_time
            
            evidence_results["evidence_grading"] = {
                "grading_time": grading_time,
                "grades_generated": len(grades) if grades else 0,
                "performance_rating": self._rate_performance("evidence", [grading_time])
            }
            logger.info(f"✅ Evidence Grading - {grading_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Evidence Grading test failed: {e}")
            evidence_results["evidence_grading"] = {"error": str(e), "performance_rating": "FAILED"}
        
        self.results["evidence_systems"] = evidence_results
    
    async def _test_integration_performance(self):
        """Test integración entre componentes"""
        logger.info("🔗 Testing Integration Performance...")
        
        try:
            # Test simple workflow: Search + Generate
            start_time = time.time()
            
            # Step 1: Get evidence
            pubmed = PubMedConnector()
            evidence = await pubmed.search(self.test_topic, max_results=3)
            
            # Step 2: Generate content
            router = get_llm_router()
            generator = IntelligentContentGenerator(router)
            content = await generator._generate_executive_summary(
                self.test_topic,
                {"complexity": "moderate"},
                evidence.sources if evidence else []
            )
            
            integration_time = time.time() - start_time
            
            self.results["integration"] = {
                "total_workflow_time": integration_time,
                "evidence_sources": len(evidence.sources) if evidence else 0,
                "content_generated": len(content) if content else 0,
                "performance_rating": self._rate_performance("integration", [integration_time])
            }
            
            logger.info(f"✅ Integration - Full workflow: {integration_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            self.results["integration"] = {"error": str(e), "performance_rating": "FAILED"}
    
    def _rate_performance(self, component_type: str, times: List[float]) -> str:
        """Rate performance based on component type and times"""
        max_time = max(times) if times else float('inf')
        
        thresholds = {
            "llm": {"excellent": 10, "good": 30, "acceptable": 60},
            "connector": {"excellent": 5, "good": 15, "acceptable": 30},
            "content": {"excellent": 15, "good": 45, "acceptable": 90},
            "evidence": {"excellent": 5, "good": 15, "acceptable": 30},
            "integration": {"excellent": 30, "good": 90, "acceptable": 180}
        }
        
        threshold = thresholds.get(component_type, thresholds["llm"])
        
        if max_time <= threshold["excellent"]:
            return "EXCELLENT"
        elif max_time <= threshold["good"]:
            return "GOOD"
        elif max_time <= threshold["acceptable"]:
            return "ACCEPTABLE"
        else:
            return "SLOW"
    
    def _analyze_bottlenecks(self):
        """Analyze results to identify bottlenecks"""
        
        # Collect all performance ratings
        bottlenecks = []
        recommendations = []
        
        for component, data in self.results.items():
            if isinstance(data, dict):
                if "performance_rating" in data:
                    rating = data["performance_rating"]
                    if rating in ["SLOW", "FAILED"]:
                        bottlenecks.append({
                            "component": component,
                            "rating": rating,
                            "data": data
                        })
                
                # Check sub-components
                for sub_component, sub_data in data.items():
                    if isinstance(sub_data, dict) and "performance_rating" in sub_data:
                        rating = sub_data["performance_rating"]
                        if rating in ["SLOW", "FAILED"]:
                            bottlenecks.append({
                                "component": f"{component}.{sub_component}",
                                "rating": rating,
                                "data": sub_data
                            })
        
        # Generate recommendations
        if bottlenecks:
            for bottleneck in bottlenecks:
                component = bottleneck["component"]
                rating = bottleneck["rating"]
                
                if "llm_router" in component:
                    recommendations.append("🧠 LLM Router: Considerar timeouts más agresivos o proveedores adicionales")
                elif "connectors" in component:
                    recommendations.append(f"🔗 {component}: Optimizar queries de búsqueda o implementar cache más agresivo")
                elif "content_generator" in component:
                    recommendations.append("📝 Content Generator: Reducir tokens o paralelizar más secciones")
                elif "evidence_systems" in component:
                    recommendations.append(f"🔬 {component}: Evaluar si es crítico para el workflow básico")
                elif "integration" in component:
                    recommendations.append("🔗 Integration: Revisar flujo de datos entre componentes")
        
        # Generate comprehensive report
        self._generate_diagnostic_report(bottlenecks, recommendations)
    
    def _generate_diagnostic_report(self, bottlenecks: List[Dict], recommendations: List[str]):
        """Generate detailed diagnostic report"""
        
        report = f"""
# 🔬 DIAGNOSTIC PERFORMANCE REPORT

## 📊 Component Performance Summary

### 🧠 LLM Router
- **Status**: {self.results.get('llm_router', {}).get('performance_rating', 'Unknown')}
- **Short Request**: {self.results.get('llm_router', {}).get('short_request_time', 0):.2f}s
- **Medium Request**: {self.results.get('llm_router', {}).get('medium_request_time', 0):.2f}s
- **Long Request**: {self.results.get('llm_router', {}).get('long_request_time', 0):.2f}s
- **Parallel Processing**: {self.results.get('llm_router', {}).get('parallel_time', 0):.2f}s

### 🔗 Connectors
"""
        
        # Add connector details
        connectors = self.results.get('connectors', {})
        for name, data in connectors.items():
            if isinstance(data, dict):
                status = data.get('performance_rating', 'Unknown')
                time_taken = data.get('search_time', 0)
                results = data.get('results_count', 0)
                report += f"- **{name.title()}**: {status} - {time_taken:.2f}s ({results} results)\n"
        
        # Add other components
        report += f"""
### 📝 Content Generator
- **Status**: {self.results.get('content_generator', {}).get('performance_rating', 'Unknown')}
- **Section Generation**: {self.results.get('content_generator', {}).get('single_section_time', 0):.2f}s
- **Topic Analysis**: {self.results.get('content_generator', {}).get('topic_analysis_time', 0):.2f}s

### 🔬 Evidence Systems
"""
        
        evidence = self.results.get('evidence_systems', {})
        for name, data in evidence.items():
            if isinstance(data, dict):
                status = data.get('performance_rating', 'Unknown')
                report += f"- **{name.replace('_', ' ').title()}**: {status}\n"
        
        report += f"""
### 🔗 Integration
- **Status**: {self.results.get('integration', {}).get('performance_rating', 'Unknown')}
- **Full Workflow**: {self.results.get('integration', {}).get('total_workflow_time', 0):.2f}s

## 🚨 IDENTIFIED BOTTLENECKS
"""
        
        if bottlenecks:
            for bottleneck in bottlenecks:
                report += f"- **{bottleneck['component']}**: {bottleneck['rating']}\n"
        else:
            report += "✅ No critical bottlenecks identified!\n"
        
        report += "\n## 💡 RECOMMENDATIONS\n"
        
        if recommendations:
            for rec in recommendations:
                report += f"- {rec}\n"
        else:
            report += "✅ System performance is acceptable!\n"
        
        # Priority assessment
        critical_count = len([b for b in bottlenecks if b['rating'] == 'FAILED'])
        slow_count = len([b for b in bottlenecks if b['rating'] == 'SLOW'])
        
        if critical_count > 0:
            priority = "🔴 CRITICAL - Immediate action required"
        elif slow_count > 2:
            priority = "🟡 HIGH - Performance optimization needed"
        elif slow_count > 0:
            priority = "🟢 MEDIUM - Minor optimizations recommended"
        else:
            priority = "✅ LOW - System performing well"
        
        report += f"""
## 🎯 PRIORITY ASSESSMENT
{priority}

**Critical Issues**: {critical_count}
**Performance Issues**: {slow_count}
**Total Components Tested**: {len(self.results)}

---
*Generated by Med-STORM Diagnostic Performance Test*
        """
        
        print(report)
        
        # Save detailed results
        results_path = Path("output/diagnostic_results.json")
        results_path.parent.mkdir(exist_ok=True)
        results_path.write_text(json.dumps(self.results, indent=2))
        
        # Save report
        report_path = Path("output/diagnostic_performance_report.md")
        report_path.write_text(report)
        
        logger.info(f"📄 Diagnostic report saved to: {report_path}")
        logger.info(f"📊 Detailed results saved to: {results_path}")

async def main():
    """Run the diagnostic performance test suite"""
    diagnostic = ComponentPerformanceDiagnostic()
    results = await diagnostic.run_full_diagnostic()
    
    # Calculate overall system health
    failed_components = sum(1 for comp in results.values() 
                          if isinstance(comp, dict) and 
                          (comp.get("performance_rating") == "FAILED" or
                           any(sub.get("performance_rating") == "FAILED" 
                               for sub in comp.values() if isinstance(sub, dict))))
    
    slow_components = sum(1 for comp in results.values() 
                         if isinstance(comp, dict) and 
                         (comp.get("performance_rating") == "SLOW" or
                          any(sub.get("performance_rating") == "SLOW" 
                              for sub in comp.values() if isinstance(sub, dict))))
    
    if failed_components > 0:
        print(f"\n🔴 CRITICAL: {failed_components} components have failures!")
    elif slow_components > 2:
        print(f"\n🟡 WARNING: {slow_components} components are slow!")
    else:
        print("\n🟢 SYSTEM HEALTH: Good performance overall!")
    
    return results

if __name__ == "__main__":
    asyncio.run(main()) 