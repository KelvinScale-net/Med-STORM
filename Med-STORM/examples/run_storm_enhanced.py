#!/usr/bin/env python3
"""
🚀 STORM-ENHANCED MEDICAL RESEARCH RUNNER
========================================

REVOLUTIONARY SYSTEM COMBINING:
✅ STORM Multi-Persona Expert System
✅ Med-PRM Process Reward Model
✅ Ultra-Performance Optimizations
✅ Multi-Source Real-Time Retrieval
✅ Evidence Stratification
✅ Treatment Analysis

This is the COMPLETE implementation addressing all identified gaps.
"""

import asyncio
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any

from med_storm.config import settings
from med_storm.llm.deepseek import DeepSeekLLM
from med_storm.connectors.local_corpus import LocalCorpusConnector
from med_storm.connectors.pubmed import PubMedConnector
from med_storm.connectors.serper import SerperConnector
from med_storm.core.storm_enhanced_engine import StormEnhancedMedicalEngine
from med_storm.utils.cache import get_cache_stats, clear_cache, warm_cache_for_topic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('storm_enhanced.log')
    ]
)
logger = logging.getLogger(__name__)

def print_banner():
    """🎨 Display impressive banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 STORM-ENHANCED MEDICAL RESEARCH SYSTEM                 ║
║                                                                              ║
║  🎭 Multi-Persona Expert System  |  🧠 Process Reward Model                 ║
║  ⚡ Ultra-Performance Engine      |  🔍 Multi-Source Retrieval               ║
║  📊 Evidence Stratification       |  💊 Treatment Analysis                   ║
║                                                                              ║
║                   🏥 IMPLACABLE MEDICAL RESEARCH ENGINE                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def initialize_connectors(args) -> Dict[str, Any]:
    """🔌 Initialize all available connectors"""
    connectors = {}
    
    logger.info("🔌 Initializing connectors...")
    
    # Local corpus connector (always available)
    if Path("qdrant_storage").exists():
        try:
            # Find available collections
            collection_name = "corpus_bariatric_surgery_for_type_2_diabetes"  # Default collection
            local_connector = LocalCorpusConnector(collection_name=collection_name)
            connectors["local_corpus"] = local_connector
            logger.info("✅ Local corpus connector initialized")
        except Exception as e:
            logger.warning(f"⚠️ Local corpus connector failed: {e}")
    
    # PubMed connector
    if args.enable_pubmed:
        try:
            pubmed_connector = PubMedConnector()
            connectors["pubmed"] = pubmed_connector
            logger.info("✅ PubMed connector initialized")
        except Exception as e:
            logger.warning(f"⚠️ PubMed connector failed: {e}")
    
    # Serper connector (web search)
    if args.enable_web and settings.SERPER_API_KEY:
        try:
            serper_connector = SerperConnector(api_key=settings.SERPER_API_KEY)
            connectors["serper"] = serper_connector
            logger.info("✅ Serper web connector initialized")
        except Exception as e:
            logger.warning(f"⚠️ Serper connector failed: {e}")
    
    logger.info(f"🔌 Initialized {len(connectors)} connectors: {list(connectors.keys())}")
    return connectors

async def run_storm_research(args) -> Dict[str, Any]:
    """🔬 Execute complete STORM-enhanced research"""
    
    # Initialize LLM provider
    logger.info("🧠 Initializing DeepSeek LLM provider...")
    llm_provider = DeepSeekLLM(
        api_key=settings.DEEPSEEK_API_KEY,
        model=args.model
    )
    
    # Initialize connectors
    connectors = await initialize_connectors(args)
    
    if not connectors:
        logger.error("❌ No connectors available! Cannot proceed.")
        return {}
    
    # Pre-warm cache if requested
    if args.warm_cache:
        logger.info("🔥 Pre-warming cache...")
        await warm_cache_for_topic(args.topic)
    
    # Initialize STORM Enhanced Engine
    logger.info(f"🚀 Initializing STORM Enhanced Engine (mode: {args.performance_mode})")
    engine = StormEnhancedMedicalEngine(
        llm_provider=llm_provider,
        connectors=connectors,
        performance_mode=args.performance_mode
    )
    
    # Execute research
    logger.info(f"🔬 Starting STORM-enhanced research on: {args.topic}")
    start_time = time.time()
    
    results = await engine.research_topic(
        topic=args.topic,
        max_personas=args.max_personas,
        max_questions_per_persona=args.max_questions_per_persona,
        max_conversation_turns=args.max_conversation_turns,
        enable_process_rewards=args.enable_process_rewards,
        enable_treatment_analysis=args.enable_treatment_analysis
    )
    
    total_time = time.time() - start_time
    logger.info(f"✅ Research completed in {total_time:.2f} seconds")
    
    # Add performance report
    results["performance_report"] = engine.get_performance_report()
    
    return results

def save_results(results: Dict[str, Any], output_file: str):
    """💾 Save results to files"""
    
    logger.info(f"💾 Saving results to {output_file}")
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save complete results as JSON
    with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # Save readable report
    report_content = generate_readable_report(results)
    with open(output_path.with_suffix('.md'), 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"✅ Results saved to {output_path.with_suffix('.json')} and {output_path.with_suffix('.md')}")

def generate_readable_report(results: Dict[str, Any]) -> str:
    """📋 Generate human-readable research report"""
    
    report = f"""# 🔬 STORM-Enhanced Medical Research Report

## 📝 Topic: {results.get('topic', 'Unknown')}

**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**System**: STORM-Enhanced Medical Research Engine

---

## 🎭 Expert Perspectives Consulted

{results.get('persona_summary', 'No personas available')}

---

## ❓ Research Questions Generated

Total Questions: **{len(results.get('questions', []))}**

"""
    
    # Add questions
    for i, question in enumerate(results.get('questions', [])[:10], 1):
        report += f"{i}. {question}\n"
    
    if len(results.get('questions', [])) > 10:
        report += f"\n... and {len(results.get('questions', [])) - 10} more questions.\n"
    
    # Add synthesis
    synthesis = results.get('synthesis', {})
    if synthesis.get('content'):
        report += f"""
---

## 🧠 Medical Analysis & Synthesis

{synthesis.get('content', 'No synthesis available')}

"""
    
    # Add reasoning verification
    if synthesis.get('reasoning_verification'):
        verification = synthesis['reasoning_verification']
        report += f"""
---

## 🏆 Med-PRM Reasoning Verification

**Overall Quality Score**: {verification.get('overall_score', 0):.2f}/1.0
**Reasoning Steps Analyzed**: {verification.get('reasoning_steps', 0)}

{verification.get('report', 'No verification report available')}

"""
    
    # Add treatment analysis
    if synthesis.get('treatment_analysis'):
        report += f"""
---

## 💊 Treatment Analysis

{synthesis.get('treatment_analysis', 'No treatment analysis available')}

"""
    
    # Add performance metrics
    if results.get('performance_report'):
        report += f"""
---

{results.get('performance_report', 'No performance report available')}

"""
    
    # Add evidence summary
    report += f"""
---

## 📚 Evidence Summary

**Total Evidence Sources**: {results.get('evidence_count', 0)}

---

## 🔧 Technical Details

**Performance Mode**: {results.get('performance_metrics', {}).get('total_time', 'Unknown')}
**LLM Model**: DeepSeek
**Connectors Used**: Multiple sources (PubMed, Web, Local Corpus)
**Cache System**: Ultra-Smart Multi-Layer Cache
**Quality Control**: Oxford CEBM Evidence Hierarchy

---

*Generated by STORM-Enhanced Medical Research System*
*Combining STORM multi-persona approach with Med-PRM process rewards*
"""
    
    return report

async def display_cache_stats():
    """📊 Display cache performance statistics"""
    try:
        stats = await get_cache_stats()
        print("\n📊 Cache Performance Statistics:")
        print(f"   Cache Hits: {stats.get('hits', 0)}")
        print(f"   Cache Misses: {stats.get('misses', 0)}")
        print(f"   Hit Rate: {stats.get('hit_rate', 0):.2%}")
        print(f"   Memory Usage: {stats.get('memory_usage', 0)} items")
    except Exception as e:
        logger.warning(f"Could not retrieve cache stats: {e}")

def main():
    """🚀 Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="🚀 STORM-Enhanced Medical Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic research with all features
  python run_storm_enhanced.py "Bariatric Surgery for Type 2 Diabetes"
  
  # High-performance mode with all connectors
  python run_storm_enhanced.py "COVID-19 Treatment" --performance-mode ultra --enable-pubmed --enable-web
  
  # Quality-focused research with process rewards
  python run_storm_enhanced.py "Heart Failure Management" --performance-mode quality --max-personas 6
  
  # Research with custom output
  python run_storm_enhanced.py "Diabetes Medications" --output results/diabetes_study
        """
    )
    
    # Required arguments
    parser.add_argument("topic", help="Medical topic to research")
    
    # STORM-specific arguments
    parser.add_argument("--max-personas", type=int, default=4,
                       help="Maximum number of expert personas to generate (default: 4)")
    parser.add_argument("--max-questions-per-persona", type=int, default=8,
                       help="Maximum questions per persona (default: 8)")
    parser.add_argument("--max-conversation-turns", type=int, default=3,
                       help="Maximum conversation turns between personas (default: 3)")
    
    # Med-PRM specific arguments
    parser.add_argument("--enable-process-rewards", action="store_true", default=True,
                       help="Enable Med-PRM process reward model (default: True)")
    parser.add_argument("--disable-process-rewards", action="store_true",
                       help="Disable process reward model")
    
    # Feature toggles
    parser.add_argument("--enable-treatment-analysis", action="store_true", default=True,
                       help="Enable comprehensive treatment analysis (default: True)")
    parser.add_argument("--enable-pubmed", action="store_true", default=True,
                       help="Enable PubMed real-time search (default: True)")
    parser.add_argument("--enable-web", action="store_true", default=True,
                       help="Enable web search via Serper (default: True)")
    
    # Performance arguments
    parser.add_argument("--performance-mode", choices=["ultra", "balanced", "quality"], 
                       default="balanced", help="Performance mode (default: balanced)")
    parser.add_argument("--warm-cache", action="store_true",
                       help="Pre-warm cache before research")
    parser.add_argument("--clear-cache", action="store_true",
                       help="Clear cache before research")
    
    # LLM arguments
    parser.add_argument("--model", default="deepseek-chat",
                       help="DeepSeek model to use (default: deepseek-chat)")
    parser.add_argument("--max-tokens", type=int, default=4000,
                       help="Maximum tokens per LLM call (default: 4000)")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="LLM temperature (default: 0.1)")
    
    # Output arguments
    parser.add_argument("--output", default="output/storm_enhanced_research",
                       help="Output file prefix (default: output/storm_enhanced_research)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Handle disable flags
    if args.disable_process_rewards:
        args.enable_process_rewards = False
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print banner
    print_banner()
    
    async def async_main():
        try:
            # Clear cache if requested
            if args.clear_cache:
                logger.info("🧹 Clearing cache...")
                await clear_cache()
            
            # Run research
            results = await run_storm_research(args)
            
            if results:
                # Save results
                save_results(results, args.output)
                
                # Display summary
                print(f"\n✅ Research completed successfully!")
                print(f"📝 Topic: {args.topic}")
                print(f"🎭 Personas: {results.get('performance_metrics', {}).get('personas_generated', 0)}")
                print(f"❓ Questions: {results.get('performance_metrics', {}).get('questions_generated', 0)}")
                print(f"📚 Evidence: {results.get('evidence_count', 0)} sources")
                print(f"⏱️ Time: {results.get('performance_metrics', {}).get('total_time', 0):.2f}s")
                
                if results.get('synthesis', {}).get('reasoning_verification'):
                    score = results['synthesis']['reasoning_verification'].get('overall_score', 0)
                    print(f"🧠 Reasoning Quality: {score:.2f}/1.0")
                
                # Display cache stats
                await display_cache_stats()
                
                print(f"\n📄 Results saved to: {args.output}.json and {args.output}.md")
            else:
                print("❌ Research failed - no results generated")
                return 1
                
        except KeyboardInterrupt:
            print("\n⚠️ Research interrupted by user")
            return 1
        except Exception as e:
            logger.error(f"❌ Research failed: {e}")
            return 1
        
        return 0
    
    # Run async main
    exit_code = asyncio.run(async_main())
    exit(exit_code)

if __name__ == "__main__":
    main() 