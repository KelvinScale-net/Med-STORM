#!/usr/bin/env python3
"""
🔬 CORRECTED DIAGNOSTIC TEST
Test con los constructores correctos
"""

import asyncio
import time
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

async def test_content_generator_corrected():
    """Test content generator con constructor correcto"""
    print("\n📝 Testing Content Generator (Corrected)...")
    
    try:
        from med_storm.synthesis.intelligent_content_generator import IntelligentContentGenerator
        
        # Constructor correcto - sin argumentos
        start = time.time()
        generator = IntelligentContentGenerator()
        init_time = time.time() - start
        
        print(f"✅ Content Generator init: {init_time:.2f}s")
        
        # Test topic analysis
        start = time.time()
        analysis = await generator._analyze_topic("Type 2 Diabetes")
        analysis_time = time.time() - start
        
        print(f"✅ Topic Analysis: {analysis_time:.2f}s")
        
        # Test single section
        start = time.time()
        section = await generator._generate_executive_summary(
            "Type 2 Diabetes", 
            {"complexity": "moderate", "specialty": "endocrinology"}, 
            []
        )
        section_time = time.time() - start
        
        print(f"✅ Single Section: {section_time:.2f}s")
        
        if section_time > 120:  # 2 minutos
            print("🔴 AQUÍ ESTÁ EL PROBLEMA! Las secciones toman >2 minutos")
        elif section_time > 60:
            print("🟡 Las secciones son lentas (~1 minuto cada una)")
        else:
            print("🟢 Las secciones son rápidas")
            
        # Estimación de tiempo total para el reporte completo
        estimated_full_report = section_time * 6  # 6 secciones
        print(f"⏱️  ESTIMACIÓN TOTAL: {estimated_full_report/60:.1f} minutos para reporte completo")
        
        if estimated_full_report > 1800:  # 30 minutos
            print("🔴 ESTO EXPLICA LAS 3 HORAS!")
        
        return {
            "init": init_time, 
            "analysis": analysis_time, 
            "section": section_time,
            "estimated_total": estimated_full_report
        }
        
    except Exception as e:
        print(f"❌ Content Generator Failed: {e}")
        return None

async def test_revolutionary_engine():
    """Test si el problema está en el revolutionary engine"""
    print("\n🚀 Testing Revolutionary Engine Components...")
    
    try:
        # Test Evidence Systems
        from med_storm.evidence.systematic_review_engine import SystematicReviewEngine
        start = time.time()
        review_engine = SystematicReviewEngine()
        review_init = time.time() - start
        print(f"✅ Systematic Review Engine init: {review_init:.2f}s")
        
        # Test Evidence Grading
        from med_storm.evidence.evidence_grading import EvidenceGradingSystem
        start = time.time()
        grading = EvidenceGradingSystem()
        grading_init = time.time() - start
        print(f"✅ Evidence Grading init: {grading_init:.2f}s")
        
        if review_init > 10 or grading_init > 10:
            print("🔴 Evidence systems son lentos de inicializar")
        else:
            print("🟢 Evidence systems se inicializan rápido")
            
        return {"review_init": review_init, "grading_init": grading_init}
        
    except Exception as e:
        print(f"❌ Revolutionary Engine Failed: {e}")
        return None

async def main():
    """Ejecutar diagnóstico corregido"""
    print("🔬 CORRECTED DIAGNOSTIC TEST - Identificando el verdadero problema\n")
    
    # Test 1: Content Generator corregido
    content_results = await test_content_generator_corrected()
    
    # Test 2: Revolutionary components
    revolutionary_results = await test_revolutionary_engine()
    
    # Análisis final
    print(f"\n{'='*60}")
    print("🎯 DIAGNÓSTICO FINAL CORREGIDO")
    print(f"{'='*60}")
    
    if content_results:
        section_time = content_results.get("section", 0)
        estimated_total = content_results.get("estimated_total", 0)
        
        print(f"📊 MÉTRICAS CLAVE:")
        print(f"   - Tiempo por sección: {section_time:.1f}s")
        print(f"   - Tiempo estimado total: {estimated_total/60:.1f} minutos")
        
        if estimated_total > 3600:  # 1 hora
            print("\n🔴 PROBLEMA IDENTIFICADO:")
            print("   Las secciones individuales son extremadamente lentas")
            print("   Probable causa: Prompts muy largos o timeouts inadecuados")
            print("\n💡 SOLUCIÓN:")
            print("   1. Reducir complejidad de prompts")
            print("   2. Acortar tokens por sección")
            print("   3. Verificar configuración de timeouts")
        else:
            print("\n🟢 El Content Generator no es el problema principal")
    
    print(f"\n🎯 CONCLUSIÓN:")
    if content_results and content_results.get("estimated_total", 0) > 3600:
        print("El problema está en la generación de contenido - cada sección toma demasiado tiempo")
    else:
        print("El problema puede estar en otra parte del flujo o en la configuración")

if __name__ == "__main__":
    asyncio.run(main())
