"""
🚀 ULTRA-STORM Advanced Monitoring and Metrics System
Real-time performance tracking and system health monitoring
"""
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """📊 Comprehensive performance metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Timing metrics
    total_time: float = 0.0
    outline_time: float = 0.0
    question_generation_time: float = 0.0
    evidence_search_time: float = 0.0
    synthesis_time: float = 0.0
    report_generation_time: float = 0.0
    
    # Throughput metrics
    questions_processed: int = 0
    sources_found: int = 0
    synthesis_tasks: int = 0
    api_calls: int = 0
    sections_processed: int = 0
    
    # Quality metrics
    average_confidence: float = 0.0
    high_confidence_sources: int = 0
    cache_hit_rate: float = 0.0
    
    # Error metrics
    errors_encountered: int = 0
    timeout_errors: int = 0
    api_errors: int = 0
    synthesis_errors: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'timing': {
                'total_time': self.total_time,
                'outline_time': self.outline_time,
                'question_generation_time': self.question_generation_time,
                'evidence_search_time': self.evidence_search_time,
                'synthesis_time': self.synthesis_time,
                'report_generation_time': self.report_generation_time
            },
            'throughput': {
                'questions_processed': self.questions_processed,
                'sources_found': self.sources_found,
                'synthesis_tasks': self.synthesis_tasks,
                'api_calls': self.api_calls,
                'sections_processed': self.sections_processed,
                'questions_per_second': self.questions_processed / max(self.total_time, 0.1),
                'sources_per_question': self.sources_found / max(self.questions_processed, 1)
            },
            'quality': {
                'average_confidence': self.average_confidence,
                'high_confidence_sources': self.high_confidence_sources,
                'cache_hit_rate': self.cache_hit_rate
            },
            'errors': {
                'total_errors': self.errors_encountered,
                'timeout_errors': self.timeout_errors,
                'api_errors': self.api_errors,
                'synthesis_errors': self.synthesis_errors,
                'error_rate': self.errors_encountered / max(self.api_calls, 1)
            }
        }

@dataclass
class SystemHealth:
    """🏥 System health status"""
    status: str = "healthy"  # healthy, warning, critical
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    queue_size: int = 0
    last_error: Optional[str] = None
    uptime: float = 0.0

class UltraStormMonitor:
    """🔍 ULTRA-STORM Advanced Performance Monitor"""
    
    def __init__(self, max_history: int = 1000):
        self.start_time = time.time()
        self.max_history = max_history
        
        # Real-time metrics
        self.current_metrics = PerformanceMetrics()
        self.metrics_history: deque = deque(maxlen=max_history)
        
        # Performance tracking
        self.stage_timers: Dict[str, float] = {}
        self.stage_history: Dict[str, List[float]] = defaultdict(list)
        
        # Error tracking
        self.error_log: deque = deque(maxlen=100)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # System health
        self.health = SystemHealth()
        
        # Real-time stats
        self.rolling_window = deque(maxlen=60)  # Last 60 seconds
        
        logger.info("🚀 ULTRA-STORM Monitor initialized")
    
    def start_stage(self, stage_name: str) -> None:
        """⏰ Start timing a stage"""
        self.stage_timers[stage_name] = time.time()
        logger.debug(f"🏁 Started stage: {stage_name}")
    
    def end_stage(self, stage_name: str) -> float:
        """⏹️ End timing a stage and return duration"""
        if stage_name not in self.stage_timers:
            logger.warning(f"Stage '{stage_name}' was not started")
            return 0.0
        
        duration = time.time() - self.stage_timers[stage_name]
        self.stage_history[stage_name].append(duration)
        
        # Update current metrics based on stage
        if stage_name == "outline_generation":
            self.current_metrics.outline_time = duration
        elif stage_name == "question_generation":
            self.current_metrics.question_generation_time = duration
        elif stage_name == "evidence_search":
            self.current_metrics.evidence_search_time = duration
        elif stage_name == "synthesis":
            self.current_metrics.synthesis_time = duration
        elif stage_name == "report_generation":
            self.current_metrics.report_generation_time = duration
        
        logger.info(f"✅ Completed stage: {stage_name} in {duration:.2f}s")
        return duration
    
    def record_api_call(self, success: bool = True, error_type: Optional[str] = None) -> None:
        """📞 Record an API call"""
        self.current_metrics.api_calls += 1
        
        if not success:
            self.current_metrics.errors_encountered += 1
            if error_type:
                self.error_counts[error_type] += 1
                if error_type == "timeout":
                    self.current_metrics.timeout_errors += 1
                elif error_type == "api_error":
                    self.current_metrics.api_errors += 1
    
    def record_synthesis(self, success: bool = True) -> None:
        """🔬 Record a synthesis operation"""
        self.current_metrics.synthesis_tasks += 1
        if not success:
            self.current_metrics.synthesis_errors += 1
    
    def record_evidence(self, source_count: int, avg_confidence: float = 0.0) -> None:
        """📚 Record evidence search results"""
        self.current_metrics.sources_found += source_count
        if avg_confidence >= 0.8:
            self.current_metrics.high_confidence_sources += source_count
        
        # Update rolling average confidence
        total_sources = self.current_metrics.sources_found
        if total_sources > 0:
            self.current_metrics.average_confidence = (
                (self.current_metrics.average_confidence * (total_sources - source_count) + 
                 avg_confidence * source_count) / total_sources
            )
    
    def record_questions(self, question_count: int) -> None:
        """❓ Record question processing"""
        self.current_metrics.questions_processed += question_count
    
    def record_sections(self, section_count: int) -> None:
        """📝 Record section processing"""
        self.current_metrics.sections_processed += section_count
    
    def record_error(self, error: Exception, context: str = "") -> None:
        """❌ Record an error with context"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'message': str(error),
            'context': context
        }
        self.error_log.append(error_entry)
        self.current_metrics.errors_encountered += 1
        self.health.last_error = f"{error_entry['error_type']}: {error_entry['message']}"
        logger.error(f"🚨 Error recorded: {error_entry}")
    
    def update_cache_stats(self, hit_rate: float) -> None:
        """💾 Update cache performance stats"""
        self.current_metrics.cache_hit_rate = hit_rate
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """📊 Get real-time performance statistics"""
        current_time = time.time()
        total_time = current_time - self.start_time
        
        self.current_metrics.total_time = total_time
        self.health.uptime = total_time
        
        return {
            'current_metrics': self.current_metrics.to_dict(),
            'system_health': {
                'status': self.health.status,
                'uptime': self.health.uptime,
                'last_error': self.health.last_error
            },
            'stage_averages': {
                stage: sum(times) / len(times) if times else 0.0
                for stage, times in self.stage_history.items()
            },
            'error_summary': dict(self.error_counts),
            'performance_trends': self._calculate_trends()
        }
    
    def _calculate_trends(self) -> Dict[str, str]:
        """📈 Calculate performance trends"""
        trends = {}
        
        # Questions per second trend
        if self.current_metrics.total_time > 0:
            qps = self.current_metrics.questions_processed / self.current_metrics.total_time
            if qps > 5.0:
                trends['throughput'] = "excellent"
            elif qps > 2.0:
                trends['throughput'] = "good"
            elif qps > 1.0:
                trends['throughput'] = "moderate"
            else:
                trends['throughput'] = "slow"
        
        # Error rate trend
        if self.current_metrics.api_calls > 0:
            error_rate = self.current_metrics.errors_encountered / self.current_metrics.api_calls
            if error_rate < 0.01:
                trends['reliability'] = "excellent"
            elif error_rate < 0.05:
                trends['reliability'] = "good"
            elif error_rate < 0.1:
                trends['reliability'] = "moderate"
            else:
                trends['reliability'] = "poor"
        
        # Cache efficiency trend
        if self.current_metrics.cache_hit_rate > 0.8:
            trends['cache_efficiency'] = "excellent"
        elif self.current_metrics.cache_hit_rate > 0.5:
            trends['cache_efficiency'] = "good"
        elif self.current_metrics.cache_hit_rate > 0.2:
            trends['cache_efficiency'] = "moderate"
        else:
            trends['cache_efficiency'] = "poor"
        
        return trends
    
    def generate_performance_report(self) -> str:
        """📋 Generate a comprehensive performance report"""
        stats = self.get_real_time_stats()
        
        report = f"""
🚀 ULTRA-STORM PERFORMANCE REPORT
{'='*50}
⏱️  Total Runtime: {stats['current_metrics']['timing']['total_time']:.2f}s
📊 Questions Processed: {stats['current_metrics']['throughput']['questions_processed']}
📚 Sources Found: {stats['current_metrics']['throughput']['sources_found']}
🔬 Synthesis Tasks: {stats['current_metrics']['throughput']['synthesis_tasks']}
📞 API Calls: {stats['current_metrics']['throughput']['api_calls']}

🏃‍♀️ PERFORMANCE METRICS:
   Questions/sec: {stats['current_metrics']['throughput']['questions_per_second']:.2f}
   Sources/question: {stats['current_metrics']['throughput']['sources_per_question']:.2f}
   Cache hit rate: {stats['current_metrics']['quality']['cache_hit_rate']:.1%}
   Average confidence: {stats['current_metrics']['quality']['average_confidence']:.2f}

🎯 STAGE TIMINGS:
   Outline: {stats['current_metrics']['timing']['outline_time']:.2f}s
   Questions: {stats['current_metrics']['timing']['question_generation_time']:.2f}s
   Evidence: {stats['current_metrics']['timing']['evidence_search_time']:.2f}s
   Synthesis: {stats['current_metrics']['timing']['synthesis_time']:.2f}s
   Report: {stats['current_metrics']['timing']['report_generation_time']:.2f}s

❌ ERROR SUMMARY:
   Total errors: {stats['current_metrics']['errors']['total_errors']}
   Error rate: {stats['current_metrics']['errors']['error_rate']:.1%}
   API errors: {stats['current_metrics']['errors']['api_errors']}
   Synthesis errors: {stats['current_metrics']['errors']['synthesis_errors']}

📈 PERFORMANCE TRENDS:
   Throughput: {stats['performance_trends'].get('throughput', 'unknown')}
   Reliability: {stats['performance_trends'].get('reliability', 'unknown')}
   Cache efficiency: {stats['performance_trends'].get('cache_efficiency', 'unknown')}

🏥 SYSTEM HEALTH: {stats['system_health']['status'].upper()}
"""
        return report
    
    def save_metrics(self, filepath: str) -> None:
        """💾 Save metrics to file"""
        try:
            stats = self.get_real_time_stats()
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            logger.info(f"📊 Metrics saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def reset_metrics(self) -> None:
        """🔄 Reset all metrics"""
        self.current_metrics = PerformanceMetrics()
        self.stage_timers.clear()
        self.error_log.clear()
        self.error_counts.clear()
        self.start_time = time.time()
        logger.info("🔄 Metrics reset")

# Global monitor instance
_global_monitor: Optional[UltraStormMonitor] = None

def get_monitor() -> UltraStormMonitor:
    """Get the global monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = UltraStormMonitor()
    return _global_monitor

def reset_monitor() -> None:
    """Reset the global monitor"""
    global _global_monitor
    _global_monitor = None 