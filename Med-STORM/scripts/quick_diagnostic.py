#!/usr/bin/env python3
"""
🔬 QUICK DIAGNOSTIC TEST
Identificar cuellos de botella rápidamente
"""

import asyncio
import time
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

async def test_llm_speed():
    """Test básico de velocidad del LLM"""
    print("🧠 Testing LLM Speed...")
    
    try:
        from med_storm.llm.llm_router import get_llm_router
        
        router = get_llm_router()
        
        # Test 1: Request corto
        start = time.time()
        response1 = await router.generate("Define diabetes in 20 words.", max_tokens=50)
        time1 = time.time() - start
        
        # Test 2: Request mediano
        start = time.time()
        response2 = await router.generate("Explain type 2 diabetes treatment options.", max_tokens=200)
        time2 = time.time() - start
        
        # Test 3: Request largo
        start = time.time()
        response3 = await router.generate("Provide comprehensive analysis of diabetes management including medications, lifestyle, and monitoring.", max_tokens=500)
        time3 = time.time() - start
        
        print(f"✅ LLM Results:")
        print(f"   Short (50 tokens): {time1:.2f}s")
        print(f"   Medium (200 tokens): {time2:.2f}s") 
        print(f"   Long (500 tokens): {time3:.2f}s")
        
        if time3 > 60:
            print("🔴 LLM is SLOW - this is likely the bottleneck!")
        elif time3 > 30:
            print("🟡 LLM is moderate - could be optimized")
        else:
            print("🟢 LLM is FAST - not the bottleneck")
            
        return {"short": time1, "medium": time2, "long": time3}
        
    except Exception as e:
        print(f"❌ LLM Test Failed: {e}")
        return None

async def test_connectors_speed():
    """Test velocidad de connectors"""
    print("\n🔗 Testing Connectors Speed...")
    
    results = {}
    
    # Test PubMed
    try:
        start = time.time()
        # Import y create inline para evitar errores de sintaxis
        exec("""
from med_storm.connectors.pubmed import PubMedConnector
pubmed = PubMedConnector()
""", globals())
        
        # Test simple sin hacer la búsqueda real
        time_taken = time.time() - start
        results["pubmed_init"] = time_taken
        print(f"✅ PubMed init: {time_taken:.2f}s")
        
    except Exception as e:
        print(f"❌ PubMed Failed: {e}")
        results["pubmed_init"] = None
    
    # Test Serper
    try:
        from med_storm.connectors.serper import SerperConnector
        start = time.time()
        serper = SerperConnector()
        time_taken = time.time() - start
        results["serper_init"] = time_taken
        print(f"✅ Serper init: {time_taken:.2f}s")
        
    except Exception as e:
        print(f"❌ Serper Failed: {e}")
        results["serper_init"] = None
    
    # Test Local Corpus
    try:
        from med_storm.connectors.local_corpus import LocalCorpusConnector
        start = time.time()
        local = LocalCorpusConnector()
        time_taken = time.time() - start
        results["local_init"] = time_taken
        print(f"✅ Local Corpus init: {time_taken:.2f}s")
        
    except Exception as e:
        print(f"❌ Local Corpus Failed: {e}")
        results["local_init"] = None
    
    return results

async def test_content_generator():
    """Test content generator speed"""
    print("\n�� Testing Content Generator...")
    
    try:
        from med_storm.llm.llm_router import get_llm_router
        from med_storm.synthesis.intelligent_content_generator import IntelligentContentGenerator
        
        router = get_llm_router()
        generator = IntelligentContentGenerator(router)
        
        # Test topic analysis
        start = time.time()
        analysis = await generator._analyze_topic("Type 2 Diabetes")
        analysis_time = time.time() - start
        
        print(f"✅ Topic Analysis: {analysis_time:.2f}s")
        
        if analysis_time > 30:
            print("🔴 Content Generator is SLOW!")
        elif analysis_time > 15:
            print("🟡 Content Generator is moderate")
        else:
            print("🟢 Content Generator is FAST")
            
        return {"analysis": analysis_time}
        
    except Exception as e:
        print(f"❌ Content Generator Failed: {e}")
        return None

async def main():
    """Ejecutar diagnóstico rápido"""
    print("🔬 QUICK DIAGNOSTIC TEST - Identificando cuellos de botella\n")
    
    # Test 1: LLM Speed
    llm_results = await test_llm_speed()
    
    # Test 2: Connectors
    connector_results = await test_connectors_speed()
    
    # Test 3: Content Generator
    content_results = await test_content_generator()
    
    # Análisis final
    print(f"\n{'='*50}")
    print("🎯 DIAGNÓSTICO FINAL")
    print(f"{'='*50}")
    
    # Identificar el peor rendimiento
    slow_components = []
    
    if llm_results and llm_results.get("long", 0) > 60:
        slow_components.append("�� LLM Router (>60s para requests largos)")
    
    if connector_results:
        for name, time_val in connector_results.items():
            if time_val and time_val > 10:
                slow_components.append(f"🔗 {name} (>{time_val:.1f}s)")
    
    if content_results and content_results.get("analysis", 0) > 30:
        slow_components.append("📝 Content Generator (>30s)")
    
    if slow_components:
        print("🔴 COMPONENTES LENTOS IDENTIFICADOS:")
        for comp in slow_components:
            print(f"   - {comp}")
        print("\n💡 RECOMENDACIÓN: Estos componentes necesitan optimización prioritaria")
    else:
        print("🟢 NO se identificaron cuellos de botella obvios")
        print("💡 El problema puede estar en la configuración o en operaciones específicas")
    
    # Estimación de tiempo total
    if llm_results:
        estimated_total = llm_results.get("long", 0) * 6  # 6 secciones de contenido
        print(f"\n⏱️  ESTIMACIÓN: Con configuración actual, un reporte completo tomaría ~{estimated_total/60:.1f} minutos")
        
        if estimated_total > 1800:  # 30 minutos
            print("🔴 ESTO EXPLICA LAS 3 HORAS - El LLM está extremadamente lento!")

if __name__ == "__main__":
    asyncio.run(main())
