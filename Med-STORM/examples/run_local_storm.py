# examples/run_local_storm.py

"""
🚀 ULTRA-STORM RUNNER: Revolutionary Medical Research Engine
===========================================================

Performance Features:
- MASSIVE PARALLELIZATION: All subtopics processed simultaneously
- INTELLIGENT CACHING: Multi-layer caching with prediction
- REAL-TIME MONITORING: Live performance metrics
- ADAPTIVE OPTIMIZATION: Self-tuning based on load
- STREAMING RESULTS: Results available as they're generated
"""

import asyncio
import argparse
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from med_storm.core.engine import UltraStormEngine
from med_storm.llm.deepseek import DeepSeekLLM
from med_storm.synthesis.engine import SynthesisEngine
from med_storm.synthesis.report_generator import ReportGenerator
from med_storm.synthesis.executive_summary_generator import ExecutiveSummaryGenerator
from med_storm.synthesis.bibliography_generator import BibliographyGenerator
from med_storm.utils.cache import warm_cache_for_topic, get_cache_stats, clear_cache
from med_storm.config import settings

class PerformanceMonitor:
    """🔥 Real-time performance monitoring and optimization."""
    
    def __init__(self):
        self.start_time = time.time()
        self.stage_times = {}
        self.current_stage = None
    
    def start_stage(self, stage_name: str):
        """Start timing a new stage."""
        if self.current_stage:
            self.end_stage()
        self.current_stage = stage_name
        self.stage_times[stage_name] = {'start': time.time()}
        print(f"🚀 [{self._elapsed():.1f}s] Starting: {stage_name}")
    
    def end_stage(self):
        """End the current stage."""
        if self.current_stage and self.current_stage in self.stage_times:
            self.stage_times[self.current_stage]['end'] = time.time()
            duration = self.stage_times[self.current_stage]['end'] - self.stage_times[self.current_stage]['start']
            print(f"✅ [{self._elapsed():.1f}s] Completed: {self.current_stage} ({duration:.2f}s)")
            self.current_stage = None
    
    def _elapsed(self) -> float:
        """Get elapsed time since start."""
        return time.time() - self.start_time
    
    def get_summary(self) -> str:
        """Get performance summary."""
        total_time = self._elapsed()
        summary = [f"\n🏆 ULTRA-STORM PERFORMANCE REPORT"]
        summary.append(f"{'='*50}")
        summary.append(f"Total Execution Time: {total_time:.2f}s")
        summary.append(f"{'='*50}")
        
        for stage, times in self.stage_times.items():
            if 'end' in times:
                duration = times['end'] - times['start']
                percentage = (duration / total_time) * 100
                summary.append(f"{stage:<30} {duration:>8.2f}s ({percentage:>5.1f}%)")
        
        summary.append(f"{'='*50}")
        return "\n".join(summary)

async def print_cache_stats():
    """Print cache performance statistics."""
    try:
        stats = await get_cache_stats()
        print(f"\n📊 CACHE PERFORMANCE:")
        print(f"   Hit Rate: {stats.get('hit_rate', 0)*100:.1f}%")
        print(f"   Memory Items: {stats.get('memory_items', 0)}")
        print(f"   Total Hits: {stats.get('hits', 0)}")
        print(f"   Total Misses: {stats.get('misses', 0)}")
    except Exception as e:
        print(f"   Cache stats unavailable: {e}")

async def optimize_for_topic(topic: str):
    """🧠 Pre-optimize system for the given topic."""
    print(f"🧠 Pre-warming cache for topic: '{topic}'")
    await warm_cache_for_topic(topic)

async def main():
    """🚀 ULTRA-STORM Main Runner with Revolutionary Performance"""
    
    parser = argparse.ArgumentParser(
        description="🚀 ULTRA-STORM: Revolutionary Medical Research Engine",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "topic", 
        type=str, 
        help="Research topic for comprehensive analysis"
    )
    parser.add_argument(
        "collection_name",
        type=str,
        help="Qdrant collection name (created with scripts/create_corpus.py)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for reports (default: output)"
    )
    parser.add_argument(
        "--max_results",
        type=int,
        default=5,
        help="Max results per question (default: 5)"
    )
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="Clear cache before starting"
    )
    parser.add_argument(
        "--performance_mode",
        choices=["ultra", "balanced", "quality"],
        default="ultra",
        help="Performance vs quality trade-off (default: ultra)"
    )
    
    args = parser.parse_args()
    
    # Initialize advanced performance monitor
    from src.med_storm.utils.monitoring import get_monitor
    monitor = get_monitor()
    monitor.start_stage("total_research")
    
    print(f"""
🚀 ULTRA-STORM INITIALIZED
==========================
Topic: {args.topic}
Collection: {args.collection_name}
Mode: {args.performance_mode.upper()}
Max Results: {args.max_results}
Output: {args.output_dir}
==========================
""")
    
    # Optional cache clearing
    if args.clear_cache:
        monitor.start_stage("cache_clearing")
        await clear_cache()
        monitor.end_stage("cache_clearing")
    
    # Pre-optimization
    monitor.start_stage("system_pre_optimization")
    await optimize_for_topic(args.topic)
    monitor.end_stage("system_pre_optimization")
    
    # Component initialization with performance tuning
    monitor.start_stage("component_initialization")
    
    # Performance mode settings - optimized for memory stability
    if args.performance_mode == "ultra":
        max_concurrent_subtopics = 30  # Reduced from 100 for memory stability
        max_concurrent_questions = 100  # Reduced from 200
    elif args.performance_mode == "balanced":
        max_concurrent_subtopics = 20  # Reduced from 50 for stability  
        max_concurrent_questions = 50   # Reduced from 100
    else:  # quality
        max_concurrent_subtopics = 10  # Reduced from 20
        max_concurrent_questions = 25   # Reduced from 50
    
    # Initialize LLM with optimization
    llm_provider = DeepSeekLLM(api_key=settings.DEEPSEEK_API_KEY)
    
    # Initialize synthesis components with context management
    from src.med_storm.synthesis.engine import IntelligentSynthesisEngine
    synthesis_engine = IntelligentSynthesisEngine(llm_provider=llm_provider, max_context_tokens=60000)
    report_generator = ReportGenerator(llm_provider=llm_provider)
    summary_generator = ExecutiveSummaryGenerator(llm_provider=llm_provider)
    bib_generator = BibliographyGenerator()
    
    # Initialize ULTRA-STORM Engine
    engine = UltraStormEngine(
        llm_provider=llm_provider,
        synthesis_engine=synthesis_engine,
        report_generator=report_generator,
        executive_summary_generator=summary_generator,
        bibliography_generator=bib_generator,
        max_concurrent_subtopics=max_concurrent_subtopics,
        max_concurrent_questions=max_concurrent_questions,
        adaptive_batching=True,
    )
    
    # Set corpus with optimization
    engine.set_corpus(collection_name=args.collection_name)
    
    monitor.end_stage("component_initialization")
    
    # Execute ULTRA-STORM research
    monitor.start_stage("ultrastorm_research_execution")
    
    print(f"🔥 Launching ULTRA-STORM research...")
    print(f"🎯 Processing ALL subtopics simultaneously with {max_concurrent_subtopics} max concurrency")
    
    try:
        async with engine:  # Use context manager for cleanup
            final_report_data = await engine.run_storm(
                topic=args.topic,
                max_results_per_question=args.max_results
            )
        
        monitor.end_stage("ultrastorm_research_execution")
        
        # Performance metrics display
        monitor.start_stage("results_processing_export")
        
        if final_report_data and "final_report" in final_report_data:
            # Create output directory
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Generate optimized filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_topic = "".join(c for c in args.topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_topic = safe_topic.replace(' ', '_').lower()[:30]
            filename = f"{args.output_dir}/ultrastorm_{safe_topic}_{timestamp}.md"
            
            # Write report with performance metrics
            report_content = final_report_data["final_report"]
            
            # Add performance footer
            performance_metrics = final_report_data.get("performance_metrics", {})
            
            report_footer = f"""
---

## 🚀 ULTRA-STORM Performance Report

**Total Execution Time:** {performance_metrics.get('total_time', 0):.2f} seconds
**Questions Processed:** {performance_metrics.get('questions_processed', 0)}
**Sources Found:** {performance_metrics.get('sources_found', 0)}
**Synthesis Tasks:** {performance_metrics.get('synthesis_tasks', 0)}
**Total API Calls:** {performance_metrics.get('total_api_calls', 0)}
**Sections Processed:** {performance_metrics.get('sections_processed', 0)}

*Generated by ULTRA-STORM v2.0 - Revolutionary Medical Research Engine*
"""
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_content + report_footer)
            
            print(f"\n🎉 ULTRA-STORM RESEARCH COMPLETED!")
            print(f"📄 Report saved: {filename}")
            
            # Display advanced performance summary
            print(monitor.generate_performance_report())
            
            # Performance metrics
            print(f"\n🏆 RESEARCH METRICS:")
            print(f"   Sections: {performance_metrics.get('sections_processed', 0)}")
            print(f"   Questions: {performance_metrics.get('questions_processed', 0)}")
            print(f"   Sources: {performance_metrics.get('sources_found', 0)}")
            print(f"   Speed: {performance_metrics.get('questions_processed', 0) / performance_metrics.get('total_time', 1):.1f} questions/sec")
            
            # Cache performance
            await print_cache_stats()
            
        else:
            print("\n❌ ULTRA-STORM completed but no report was generated.")
            print("Check logs for potential issues.")
        
        monitor.end_stage("results_processing_export")
        
    except Exception as e:
        if hasattr(monitor, 'current_stage') and monitor.current_stage:
            monitor.end_stage("ultrastorm_research_execution")
        monitor.record_error(e, "main_execution")
        print(f"\n💥 ULTRA-STORM encountered an error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    monitor.end_stage("total_research")
    print(f"\n🚀 ULTRA-STORM session completed in {monitor.current_metrics.total_time:.2f}s")
    return 0

if __name__ == "__main__":
    # Run with proper async handling
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 