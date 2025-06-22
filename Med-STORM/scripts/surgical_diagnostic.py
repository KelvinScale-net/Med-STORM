#!/usr/bin/env python3
"""
🔬 SURGICAL DIAGNOSTIC WITH TIMEOUTS
Identificar exactamente dónde está el problema con timeouts estrictos
"""

import asyncio
import time
import sys
import signal
from pathlib import Path
from typing import Optional, Any

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

async def test_with_timeout(coro, timeout_seconds: int, test_name: str):
    """Test con timeout estricto"""
    print(f"🔬 Testing {test_name} (timeout: {timeout_seconds}s)...")
    
    start_time = time.time()
    
    try:
        # Set alarm signal
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        result = await coro
        elapsed = time.time() - start_time
        
        # Cancel alarm
        signal.alarm(0)
        
        print(f"✅ {test_name}: {elapsed:.2f}s - SUCCESS")
        return {"success": True, "time": elapsed, "result": result}
        
    except TimeoutError:
        elapsed = time.time() - start_time
        print(f"🔴 {test_name}: TIMEOUT after {elapsed:.2f}s - FAILED")
        return {"success": False, "time": elapsed, "error": "TIMEOUT"}
        
    except Exception as e:
        elapsed = time.time() - start_time
        signal.alarm(0)  # Cancel alarm
        print(f"❌ {test_name}: ERROR after {elapsed:.2f}s - {str(e)[:100]}")
        return {"success": False, "time": elapsed, "error": str(e)}

async def surgical_test_llm():
    """Test LLM con timeout de 30s"""
    from med_storm.llm.llm_router import get_llm_router
    
    router = get_llm_router()
    response = await router.generate("Define diabetes in 20 words.", max_tokens=50)
    return len(response)

async def surgical_test_pubmed():
    """Test PubMed con timeout de 60s"""
    from med_storm.connectors.pubmed import PubMedConnector
    
    pubmed = PubMedConnector()
    corpus = await pubmed.search("diabetes", max_results=3)
    return len(corpus.sources) if corpus else 0

async def surgical_test_serper():
    """Test Serper con timeout de 60s"""
    from med_storm.connectors.serper import SerperConnector
    
    serper = SerperConnector()
    corpus = await serper.search("diabetes", max_results=3)
    return len(corpus.sources) if corpus else 0

async def surgical_test_local_corpus():
    """Test Local Corpus con timeout de 60s"""
    from med_storm.connectors.local_corpus import LocalCorpusConnector
    
    # Usar el nombre de colección correcto
    local = LocalCorpusConnector(collection_name="corpus_bariatric_surgery_for_type_2_diabetes")
    corpus = await local.search("diabetes", max_results=3)
    return len(corpus.sources) if corpus else 0

async def surgical_test_content_generator():
    """Test Content Generator con timeout de 120s"""
    from med_storm.synthesis.intelligent_content_generator import IntelligentContentGenerator
    
    generator = IntelligentContentGenerator()
    content = await generator.generate_comprehensive_content(
        topic="Diabetes Test",
        target_length=500  # Reducido para velocidad
    )
    return content.get("word_count", 0)

async def surgical_test_storm_engine():
    """Test STORM Engine básico con timeout de 300s (5 min)"""
    from med_storm.core.storm_enhanced_engine import StormEnhancedMedicalEngine
    from med_storm.llm.deepseek import DeepSeekLLM
    from med_storm.connectors.pubmed import PubMedConnector
    from med_storm.connectors.serper import SerperConnector
    from med_storm.connectors.local_corpus import LocalCorpusConnector
    
    # Setup minimal engine
    llm = DeepSeekLLM()
    connectors = {
        "pubmed": PubMedConnector(),
        "serper": SerperConnector(),
        "local": LocalCorpusConnector(collection_name="corpus_bariatric_surgery_for_type_2_diabetes")
    }
    
    engine = StormEnhancedMedicalEngine(
        llm_provider=llm,
        connectors=connectors,
        performance_mode="ultra"  # Máxima velocidad
    )
    
    # Test básico sin funciones revolucionarias
    results = await engine.research_topic(
        topic="Diabetes Test",
        max_personas=2,  # Reducido
        max_questions_per_persona=2,  # Reducido
        max_conversation_turns=1,  # Reducido
        enable_process_rewards=False,  # Desactivado
        enable_treatment_analysis=False  # Desactivado
    )
    
    return results.get("evidence_count", 0)

async def main():
    """Ejecutar diagnóstico quirúrgico con timeouts"""
    print("🔬 SURGICAL DIAGNOSTIC WITH TIMEOUTS")
    print("=" * 50)
    
    results = {}
    
    # Test 1: LLM (30s timeout)
    results["llm"] = await test_with_timeout(
        surgical_test_llm(), 30, "LLM Router"
    )
    
    # Test 2: PubMed (60s timeout)
    results["pubmed"] = await test_with_timeout(
        surgical_test_pubmed(), 60, "PubMed Connector"
    )
    
    # Test 3: Serper (60s timeout)
    results["serper"] = await test_with_timeout(
        surgical_test_serper(), 60, "Serper Connector"
    )
    
    # Test 4: Local Corpus (60s timeout)
    results["local_corpus"] = await test_with_timeout(
        surgical_test_local_corpus(), 60, "Local Corpus"
    )
    
    # Test 5: Content Generator (120s timeout)
    results["content_generator"] = await test_with_timeout(
        surgical_test_content_generator(), 120, "Content Generator"
    )
    
    # Test 6: STORM Engine básico (300s timeout = 5 min)
    results["storm_engine"] = await test_with_timeout(
        surgical_test_storm_engine(), 300, "STORM Engine Basic"
    )
    
    # ANÁLISIS DE RESULTADOS
    print("\n" + "=" * 50)
    print("🎯 SURGICAL ANALYSIS")
    print("=" * 50)
    
    failed_components = []
    slow_components = []
    working_components = []
    
    for component, result in results.items():
        if not result["success"]:
            if result.get("error") == "TIMEOUT":
                failed_components.append(f"🔴 {component}: TIMEOUT after {result['time']:.1f}s")
            else:
                failed_components.append(f"❌ {component}: ERROR - {result.get('error', 'Unknown')}")
        elif result["time"] > 60:
            slow_components.append(f"🟡 {component}: SLOW ({result['time']:.1f}s)")
        else:
            working_components.append(f"✅ {component}: FAST ({result['time']:.1f}s)")
    
    print("\n🔴 FAILED COMPONENTS:")
    for comp in failed_components:
        print(f"   {comp}")
    
    print("\n🟡 SLOW COMPONENTS:")
    for comp in slow_components:
        print(f"   {comp}")
    
    print("\n✅ WORKING COMPONENTS:")
    for comp in working_components:
        print(f"   {comp}")
    
    # RECOMENDACIONES ESPECÍFICAS
    print("\n💡 SURGICAL RECOMMENDATIONS:")
    
    if len(failed_components) > 0:
        print("🔴 CRITICAL: Hay componentes que fallan completamente")
        print("   → Necesitan arreglo inmediato antes de continuar")
    
    if "storm_engine" in [comp.split(":")[0].split()[-1] for comp in failed_components]:
        print("🚨 STORM Engine falla - problema en el flujo principal")
        print("   → Revisar configuración de personas/preguntas")
        print("   → Verificar timeouts internos")
    
    if "content_generator" in [comp.split(":")[0].split()[-1] for comp in failed_components]:
        print("📝 Content Generator falla - prompts muy largos")
        print("   → Reducir tokens por sección")
        print("   → Simplificar prompts")
    
    if len(working_components) >= 4:
        print("✅ GOOD NEWS: Los componentes básicos funcionan")
        print("   → El problema está en configuraciones específicas")
    
    # TIEMPO TOTAL ESTIMADO
    total_time = sum(r["time"] for r in results.values())
    print(f"\n⏱️  TIEMPO TOTAL DE TESTS: {total_time:.1f}s")
    
    if total_time > 600:  # 10 minutos
        print("🔴 SYSTEM TOO SLOW - Optimización crítica necesaria")
    elif total_time > 300:  # 5 minutos
        print("🟡 SYSTEM SLOW - Optimización recomendada")
    else:
        print("🟢 SYSTEM SPEED ACCEPTABLE")

if __name__ == "__main__":
    asyncio.run(main()) 