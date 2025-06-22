#!/usr/bin/env python3
"""
🚀 REVOLUTIONARY HYBRID MED-STORM RUNNER
========================================

FEATURES:
1. MULTI-SOURCE REAL-TIME RETRIEVAL: PubMed + Web + Local Corpus
2. EVIDENCE QUALITY STRATIFICATION: Oxford CEBM hierarchy
3. AUTOMATED BIAS ASSESSMENT: Risk evaluation for all sources
4. COMPREHENSIVE QUALITY REPORTING: Detailed evidence assessment
5. DYNAMIC CORPUS EXPANSION: Auto-population for new topics

USAGE:
    python examples/run_hybrid_storm.py "Topic" [corpus_name] [--options]

EXAMPLES:
    # New topic with real-time retrieval only
    python examples/run_hybrid_storm.py "COVID-19 Long Haul Syndrome"
    
    # Existing topic with hybrid approach
    python examples/run_hybrid_storm.py "Bariatric Surgery for Type 2 Diabetes" corpus_bariatric_surgery_for_type_2_diabetes
    
    # High-quality evidence only
    python examples/run_hybrid_storm.py "Heart Failure Treatment" --min-evidence-level LEVEL_1B --min-year 2020
"""

import asyncio
import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.med_storm.core.hybrid_engine import HybridMedStormEngine, EvidenceLevel
from src.med_storm.llm.deepseek_llm import DeepSeekLLM
from src.med_storm.synthesis.engine import IntelligentSynthesisEngine
from src.med_storm.synthesis.report_generator import ReportGenerator
from src.med_storm.synthesis.executive_summary_generator import ExecutiveSummaryGenerator
from src.med_storm.synthesis.bibliography_generator import BibliographyGenerator
from src.med_storm.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridPerformanceMonitor:
    """📊 Advanced performance monitoring for hybrid engine"""
    
    def __init__(self):
        self.start_time = time.time()
        self.stage_times = {}
        self.current_stage = None
        
    def start_stage(self, stage_name: str):
        """Start timing a stage"""
        if self.current_stage:
            self.end_stage(self.current_stage)
        self.current_stage = stage_name
        self.stage_times[stage_name] = {'start': time.time()}
        logger.info(f"🚀 Starting stage: {stage_name}")
        
    def end_stage(self, stage_name: str):
        """End timing a stage"""
        if stage_name in self.stage_times and 'start' in self.stage_times[stage_name]:
            self.stage_times[stage_name]['end'] = time.time()
            duration = self.stage_times[stage_name]['end'] - self.stage_times[stage_name]['start']
            self.stage_times[stage_name]['duration'] = duration
            logger.info(f"✅ Completed stage: {stage_name} in {duration:.2f}s")
        self.current_stage = None
        
    def get_total_time(self) -> float:
        """Get total elapsed time"""
        return time.time() - self.start_time
        
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        total_time = self.get_total_time()
        
        report = f"""
## 📊 Hybrid Engine Performance Report

### Overall Performance
- **Total Execution Time**: {total_time:.2f} seconds
- **Average Stage Time**: {sum(s.get('duration', 0) for s in self.stage_times.values()) / len(self.stage_times):.2f}s

### Stage Breakdown
{chr(10).join([f"- **{stage}**: {data.get('duration', 0):.2f}s" for stage, data in self.stage_times.items()])}

### Performance Analysis
- **Initialization Efficiency**: {'Excellent' if self.stage_times.get('initialization', {}).get('duration', 0) < 5 else 'Good'}
- **Search Performance**: {'Excellent' if self.stage_times.get('multi_source_search', {}).get('duration', 0) < 60 else 'Good'}
- **Synthesis Speed**: {'Excellent' if self.stage_times.get('synthesis', {}).get('duration', 0) < 30 else 'Good'}
"""
        return report

async def initialize_hybrid_engine(args) -> HybridMedStormEngine:
    """🔧 Initialize the hybrid Med-STORM engine with all components"""
    
    # Validate API key
    if not settings.DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
    
    # Initialize LLM provider
    llm_provider = DeepSeekLLM(api_key=settings.DEEPSEEK_API_KEY)
    
    # Initialize synthesis components
    synthesis_engine = IntelligentSynthesisEngine(
        llm_provider=llm_provider,
        max_context_tokens=60000
    )
    
    report_generator = ReportGenerator(llm_provider=llm_provider)
    executive_summary_generator = ExecutiveSummaryGenerator(llm_provider=llm_provider)
    bibliography_generator = BibliographyGenerator()
    
    # Parse evidence level
    evidence_level_map = {
        'LEVEL_1A': EvidenceLevel.LEVEL_1A,
        'LEVEL_1B': EvidenceLevel.LEVEL_1B,
        'LEVEL_2A': EvidenceLevel.LEVEL_2A,
        'LEVEL_2B': EvidenceLevel.LEVEL_2B,
        'LEVEL_3B': EvidenceLevel.LEVEL_3B,
        'LEVEL_4': EvidenceLevel.LEVEL_4,
        'LEVEL_5': EvidenceLevel.LEVEL_5,
    }
    
    min_evidence_level = evidence_level_map.get(args.min_evidence_level, EvidenceLevel.LEVEL_4)
    
    # Initialize hybrid engine
    engine = HybridMedStormEngine(
        llm_provider=llm_provider,
        synthesis_engine=synthesis_engine,
        report_generator=report_generator,
        executive_summary_generator=executive_summary_generator,
        bibliography_generator=bibliography_generator,
        enable_real_time_pubmed=args.enable_pubmed,
        enable_web_search=args.enable_web,
        enable_local_corpus=args.enable_local,
        min_evidence_level=min_evidence_level,
        min_publication_year=args.min_year
    )
    
    # Set local corpus if provided
    if args.corpus_name and args.enable_local:
        engine.set_local_corpus(args.corpus_name)
        logger.info(f"✅ Local corpus set: {args.corpus_name}")
    
    return engine

def save_report(report: str, topic: str, output_dir: str = "output") -> str:
    """💾 Save the generated report to file"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_topic = safe_topic.replace(' ', '_').lower()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"hybrid_storm_{safe_topic}_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)
    
    # Save report
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"📄 Report saved to: {filepath}")
    return filepath

async def main():
    """🚀 Main function to run the Hybrid Med-STORM engine"""
    
    parser = argparse.ArgumentParser(
        description="Hybrid Med-STORM: Multi-Source Medical Research Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument("topic", help="Research topic to investigate")
    parser.add_argument("corpus_name", nargs='?', help="Local corpus collection name (optional)")
    
    # Source configuration
    parser.add_argument("--enable-pubmed", action="store_true", default=True, 
                       help="Enable real-time PubMed searches (default: True)")
    parser.add_argument("--disable-pubmed", dest="enable_pubmed", action="store_false",
                       help="Disable PubMed searches")
    parser.add_argument("--enable-web", action="store_true", default=True,
                       help="Enable web searches on trusted domains (default: True)")
    parser.add_argument("--disable-web", dest="enable_web", action="store_false",
                       help="Disable web searches")
    parser.add_argument("--enable-local", action="store_true", default=True,
                       help="Enable local corpus searches (default: True)")
    parser.add_argument("--disable-local", dest="enable_local", action="store_false",
                       help="Disable local corpus searches")
    
    # Quality control
    parser.add_argument("--min-evidence-level", choices=[
        'LEVEL_1A', 'LEVEL_1B', 'LEVEL_2A', 'LEVEL_2B', 'LEVEL_3B', 'LEVEL_4', 'LEVEL_5'
    ], default='LEVEL_4', help="Minimum evidence level to include (default: LEVEL_4)")
    parser.add_argument("--min-year", type=int, default=2018,
                       help="Minimum publication year (default: 2018)")
    
    # Performance settings
    parser.add_argument("--max-results", type=int, default=10,
                       help="Maximum results per question (default: 10)")
    parser.add_argument("--no-quality-assessment", action="store_true",
                       help="Skip quality assessment report generation")
    
    # Output settings
    parser.add_argument("--output-dir", default="output",
                       help="Output directory for reports (default: output)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize performance monitor
    monitor = HybridPerformanceMonitor()
    
    try:
        print("🚀 HYBRID MED-STORM ENGINE STARTING")
        print(f"📋 Topic: {args.topic}")
        print(f"🔍 Sources: PubMed={'✅' if args.enable_pubmed else '❌'}, Web={'✅' if args.enable_web else '❌'}, Local={'✅' if args.enable_local else '❌'}")
        print(f"📊 Quality: Min Level={args.min_evidence_level}, Min Year={args.min_year}")
        print("=" * 60)
        
        # STAGE 1: Initialize engine
        monitor.start_stage("initialization")
        engine = await initialize_hybrid_engine(args)
        monitor.end_stage("initialization")
        
        # STAGE 2: Run hybrid STORM
        monitor.start_stage("hybrid_storm_execution")
        results = await engine.run_hybrid_storm(
            topic=args.topic,
            max_results_per_question=args.max_results,
            include_quality_assessment=not args.no_quality_assessment
        )
        monitor.end_stage("hybrid_storm_execution")
        
        # STAGE 3: Save and display results
        monitor.start_stage("results_processing")
        
        # Save the report
        report_path = save_report(results["final_report"], args.topic, args.output_dir)
        
        # Display performance metrics
        performance_metrics = results["performance_metrics"]
        print("\n" + "=" * 60)
        print("🎉 HYBRID STORM COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Performance Summary:")
        print(f"   Total Time: {performance_metrics['total_time']:.2f}s")
        print(f"   Questions Generated: {performance_metrics['total_questions']}")
        print(f"   Sources Found: {performance_metrics['total_sources']}")
        print(f"   Connectors Used: {performance_metrics['connectors_used']}")
        print(f"   Questions/Second: {performance_metrics['total_questions'] / performance_metrics['total_time']:.2f}")
        
        # Display quality assessment summary
        if results.get("quality_assessment"):
            print(f"\n📋 Quality Assessment:")
            quality_lines = results["quality_assessment"].split('\n')[:10]  # First 10 lines
            for line in quality_lines:
                if line.strip():
                    print(f"   {line.strip()}")
        
        # Display file information
        print(f"\n📄 Report Details:")
        print(f"   File: {report_path}")
        print(f"   Size: {os.path.getsize(report_path)} bytes")
        print(f"   Lines: {len(results['final_report'].split(chr(10)))}")
        
        # Display performance report
        print(monitor.generate_performance_report())
        
        monitor.end_stage("results_processing")
        
        print(f"\n🚀 Total execution time: {monitor.get_total_time():.2f}s")
        print("✨ Hybrid Med-STORM session completed successfully!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n💥 Error occurred: {e}")
        logger.exception("Detailed error information:")
        return 1

if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 