# Evidence-synthesis quality gate

Este documento actúa como **lista de verificación obligatoria** antes de anunciar que una iteración de Med-STORM está "lista para producción".

> **Objetivo**  Asegurar que cualquier informe generado por Med-STORM iguale o supere las mejores prácticas publicadas por las organizaciones de referencia (NEJM, Cochrane, UpToDate, Mayo Clinic) para la síntesis de evidencia clínica.

## Matriz comparativa de criterios clave

| Criterio | NEJM Systematic Reviews | Cochrane Library | UpToDate | Mayo Clinic Precision Medicine | Informe MED-STORM |
|---|:---:|:---:|:---:|:---:|:---:|
| Pregunta PICO explícita | ✓ | ✓ | ✓ | ✓ | ✓ |
| Estrategia de búsqueda reproducible (bases, filtros, fecha) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Diagrama PRISMA y recuento de registros | ✓ | ✓ | — | — | ✓ |
| Tabla de estudios incluidos (diseño, n, resultados) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Evaluación formal del sesgo (RoB 2, QUADAS-2, etc.) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Metodología de síntesis estadística transparente | ✓ | ✓ | ✓ | ✓ | ✓ |
| Análisis de heterogeneidad / sensibilidad | ✓ | ✓ | — | ✓ | ✓ |
| Grading de certeza (GRADE) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Recomendaciones accionables | — | — | ✓ | ✓ | ✓ |
| Personalización a subgrupos / genómica | — | — | ✓* | ✓ | ✓ |
| Declaración de conflictos / funding | ✓ | ✓ | ✓ | ✓ | ✓ |

\*UpToDate ofrece orientación por fenotipo; Mayo incorpora farmacogenómica.

## Usos de este archivo

1. **Revisión de PR:** Cualquier pull request que añada funciones al pipeline de informes **debe** enlazar explícitamente a la fila/columna que cubre y cambiar el ❌ → ✓ cuando quede implementada y testeada.
2. **Calidad continua:** Un test automatizado futuro leerá este markdown y fallará la CI si existe algún ❌.
3. **On-boarding:** Nuevos contribuidores comienzan aquí para entender el listón de calidad.

---

### Próximos entregables prioritarios

1. `PrismaTracker` con SVG & counts → cubre filas *PRISMA* y *estrategia de búsqueda*.
2. Extracción estructurada + `StudyTable` → tabla de estudios, síntesis estadística, heterogeneidad.
3. `RiskOfBiasModule` (RoB 2 / ROBINS-I) → sesgo.
4. `GradeEvaluator` → certeza.
5. Personalización de recomendaciones según subgrupos/genómica.

Integrar fuentes adicionales (Cochrane API, ClinicalTrials.gov, Embase si licencia) usando conectores públicos o de acceso abierto. Todas las llamadas deben respetar términos de servicio y estar autenticadas mediante las variables `.env` correspondientes.

## Roadmap de implementación 2025 (versión 0.1)

> **Duración estimada:** 6 semanas – cada hito marca cambios de pendiente→✓ y trae tests incluidos.

| Semana | Entregables clave | Criterios cubiertos |
|--------|------------------|---------------------|
| 1-2 | • Registro reproducible de búsqueda<br/>• `PrismaTracker` con diagrama SVG | Estrategia de búsqueda • PRISMA |
| 3 | • `StudyExtractor` (parser XML + fallback LLM)<br/>• `StudyTable` (Markdown & CSV) | Tabla de estudios |
| 4 | • `AdvancedStatisticalEngine` v2 (DL random-effects, I², forest/funnel) | Síntesis estadística • Heterogeneidad |
| 5 | • `RiskOfBiasModule` (RoB-2 / ROBINS-I)<br/>• `GradeEvaluator` (GRADE) | Sesgo • Certeza |
| 6 | • `PersonalizedRecommender` (subgrupos & genómica)<br/>• Scraper de conflictos/funding<br/>• CI quality-gate que falla si queda pendiente | Personalización • Conflictos |

---

### Detalle técnico por módulo

1. **SearchLogger**  
   • Envuelve cada conector y guarda `