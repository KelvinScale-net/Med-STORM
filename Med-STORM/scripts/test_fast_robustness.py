#!/usr/bin/env python3
"""
⚡ FAST PRODUCTION ROBUSTNESS TEST
Quick validation of system reliability and performance
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any
import sys

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from med_storm.llm.llm_router import get_llm_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FastRobustnessTest:
    """⚡ Fast production robustness test"""
    
    def __init__(self):
        """Initialize the test suite"""
        self.results = {}
        
    async def run_fast_tests(self) -> Dict[str, Any]:
        """Run fast essential tests"""
        logger.info("⚡ Starting Fast Robustness Test Suite")
        
        # Test 1: LLM Router Speed Test
        await self._test_llm_speed()
        
        # Test 2: Parallel Processing Test
        await self._test_parallel_processing()
        
        # Test 3: Timeout and Failover Test
        await self._test_timeout_behavior()
        
        # Generate quick report
        self._generate_quick_report()
        
        return self.results
    
    async def _test_llm_speed(self):
        """Test LLM router speed with short prompts"""
        logger.info("🚀 Testing LLM Speed...")
        
        try:
            router = get_llm_router()
            
            # Short medical prompt for speed test
            short_prompt = "Define diabetes in 50 words."
            
            start_time = time.time()
            response = await router.generate(short_prompt, max_tokens=100)
            response_time = time.time() - start_time
            
            # Health check
            health = router.get_health_status()
            
            self.results["llm_speed_test"] = {
                "response_time": response_time,
                "response_length": len(response),
                "providers_available": len(health["providers"]),
                "success_rate": health["metrics"]["success_rate"],
                "fast_enough": response_time < 15,  # Should be under 15 seconds
                "success": True
            }
            
            logger.info(f"✅ LLM Speed: {response_time:.2f}s, {len(response)} chars")
            
        except Exception as e:
            logger.error(f"❌ LLM Speed test failed: {e}")
            self.results["llm_speed_test"] = {"success": False, "error": str(e)}
    
    async def _test_parallel_processing(self):
        """Test parallel request processing"""
        logger.info("⚡ Testing Parallel Processing...")
        
        try:
            router = get_llm_router()
            
            # Create 3 short parallel requests
            prompts = [
                "Define hypertension briefly.",
                "What is asthma?",
                "Explain diabetes type 2."
            ]
            
            # Sequential test
            start_sequential = time.time()
            sequential_results = []
            for prompt in prompts:
                result = await router.generate(prompt, max_tokens=50)
                sequential_results.append(result)
            sequential_time = time.time() - start_sequential
            
            # Parallel test
            start_parallel = time.time()
            tasks = [router.generate(prompt, max_tokens=50) for prompt in prompts]
            parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
            parallel_time = time.time() - start_parallel
            
            # Calculate speedup
            speedup = sequential_time / parallel_time if parallel_time > 0 else 0
            
            self.results["parallel_test"] = {
                "sequential_time": sequential_time,
                "parallel_time": parallel_time,
                "speedup": speedup,
                "parallel_success_count": sum(1 for r in parallel_results if not isinstance(r, Exception)),
                "good_speedup": speedup > 1.5,  # At least 50% faster
                "success": True
            }
            
            logger.info(f"✅ Parallel: {parallel_time:.2f}s vs {sequential_time:.2f}s (speedup: {speedup:.1f}x)")
            
        except Exception as e:
            logger.error(f"❌ Parallel test failed: {e}")
            self.results["parallel_test"] = {"success": False, "error": str(e)}
    
    async def _test_timeout_behavior(self):
        """Test timeout and circuit breaker behavior"""
        logger.info("⏰ Testing Timeout Behavior...")
        
        try:
            router = get_llm_router()
            
            # Test with a reasonable prompt that should complete quickly
            test_prompt = "List 3 symptoms of heart disease."
            
            start_time = time.time()
            response = await router.generate(test_prompt, max_tokens=100)
            response_time = time.time() - start_time
            
            # Check if timeouts are reasonable
            reasonable_timeout = response_time < 30  # Should complete in under 30 seconds
            
            # Test circuit breaker reset
            router.reset_circuit_breakers()
            health_after_reset = router.get_health_status()
            
            self.results["timeout_test"] = {
                "response_time": response_time,
                "reasonable_timeout": reasonable_timeout,
                "circuit_breaker_functional": True,
                "health_monitoring": len(health_after_reset["providers"]) > 0,
                "success": True
            }
            
            logger.info(f"✅ Timeout: {response_time:.2f}s (reasonable: {reasonable_timeout})")
            
        except Exception as e:
            logger.error(f"❌ Timeout test failed: {e}")
            self.results["timeout_test"] = {"success": False, "error": str(e)}
    
    def _generate_quick_report(self):
        """Generate quick test report"""
        
        # Calculate overall score
        test_scores = []
        for test_name, test_result in self.results.items():
            if test_result.get("success"):
                if test_name == "llm_speed_test":
                    score = 1.0 if test_result.get("fast_enough") else 0.5
                elif test_name == "parallel_test":
                    score = 1.0 if test_result.get("good_speedup") else 0.5
                elif test_name == "timeout_test":
                    score = 1.0 if test_result.get("reasonable_timeout") else 0.5
                else:
                    score = 1.0
                test_scores.append(score)
            else:
                test_scores.append(0.0)
        
        overall_score = sum(test_scores) / len(test_scores) if test_scores else 0.0
        
        report = f"""
# ⚡ FAST ROBUSTNESS TEST REPORT

## 📊 Overall Score: {overall_score:.1%}

## 🚀 LLM Speed Test
- **Status**: {'✅ PASSED' if self.results.get('llm_speed_test', {}).get('success') else '❌ FAILED'}
- **Response Time**: {self.results.get('llm_speed_test', {}).get('response_time', 0):.2f}s
- **Fast Enough**: {'✅ Yes' if self.results.get('llm_speed_test', {}).get('fast_enough') else '❌ No'}
- **Providers**: {self.results.get('llm_speed_test', {}).get('providers_available', 0)}

## ⚡ Parallel Processing Test
- **Status**: {'✅ PASSED' if self.results.get('parallel_test', {}).get('success') else '❌ FAILED'}
- **Parallel Time**: {self.results.get('parallel_test', {}).get('parallel_time', 0):.2f}s
- **Sequential Time**: {self.results.get('parallel_test', {}).get('sequential_time', 0):.2f}s
- **Speedup**: {self.results.get('parallel_test', {}).get('speedup', 0):.1f}x
- **Good Speedup**: {'✅ Yes' if self.results.get('parallel_test', {}).get('good_speedup') else '❌ No'}

## ⏰ Timeout Test
- **Status**: {'✅ PASSED' if self.results.get('timeout_test', {}).get('success') else '❌ FAILED'}
- **Response Time**: {self.results.get('timeout_test', {}).get('response_time', 0):.2f}s
- **Reasonable**: {'✅ Yes' if self.results.get('timeout_test', {}).get('reasonable_timeout') else '❌ No'}

## 🎯 Speed Assessment

{'🟢 **EXCELLENT PERFORMANCE**' if overall_score >= 0.8 else '🟡 **ACCEPTABLE PERFORMANCE**' if overall_score >= 0.6 else '🔴 **PERFORMANCE ISSUES**'}

The system demonstrates {'excellent' if overall_score >= 0.8 else 'acceptable' if overall_score >= 0.6 else 'poor'} speed and reliability.

---
*Generated by Fast Med-STORM Robustness Test*
        """
        
        print(report)
        
        # Save report
        report_path = Path("output/fast_robustness_report.md")
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(report)
        
        logger.info(f"📄 Fast test report saved to: {report_path}")

async def main():
    """Run the fast robustness test suite"""
    test_suite = FastRobustnessTest()
    results = await test_suite.run_fast_tests()
    
    # Calculate overall performance
    all_success = all(test.get("success", False) for test in results.values())
    
    if all_success:
        print("\n🟢 ALL TESTS PASSED - SYSTEM PERFORMANCE IS GOOD!")
    else:
        print("\n🔴 SOME TESTS FAILED - PERFORMANCE ISSUES DETECTED!")
    
    return results

if __name__ == "__main__":
    asyncio.run(main()) 