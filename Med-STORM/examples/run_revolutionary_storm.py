#!/usr/bin/env python3
"""
🚀 REVOLUTIONARY MED-STORM RUNNER
Runner que demuestra las capacidades revolucionarias del sistema que SUPERAN la competencia
"""

import asyncio
import sys
import os
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from med_storm.core.storm_enhanced_engine import StormEnhancedMedicalEngine
from med_storm.llm.openrouter import OpenRouterLLM
from med_storm.llm.deepseek import DeepSeekLLM
# from med_storm.llm.openrouter import OpenRouterLLM  # Ready for when we have valid API key
from med_storm.connectors.pubmed import PubMedConnector
from med_storm.connectors.serper import SerperConnector
from med_storm.connectors.local_corpus import LocalCorpusConnector
from med_storm.config import settings

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_report_to_file(report: Dict[str, Any], topic: str, report_type: str = "revolutionary") -> str:
    """
    Save report to markdown file in output directory
    
    Args:
        report: The report dictionary to save
        topic: The topic/title for the report
        report_type: Type of report (enhanced, revolutionary, etc.)
    
    Returns:
        Path to the saved file
    """
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_topic = safe_topic.replace(' ', '_')
    filename = f"{safe_topic}_{report_type}_report_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)
    
    # Generate markdown content
    markdown_content = generate_markdown_report(report, topic, report_type)
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    logger.info(f"📄 Report saved to: {filepath}")
    return filepath


def generate_markdown_report(report: Dict[str, Any], topic: str, report_type: str) -> str:
    """
    Generates a professional, data-driven medical report in Markdown format.
    This function dynamically builds the report based on the actual content
    provided in the report dictionary, avoiding placeholder values.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # --- Report Header ---
    markdown = f"""# Med-STORM {report_type.title()} Report: {topic}
**Generated:** {timestamp}
**Report ID:** {report.get('id', 'N/A')}
---

"""

    # --- Executive Summary ---
    if summary := report.get('executive_summary'):
        markdown += f"""## Executive Summary
{summary}
---
"""

    # --- Synthesis / Main Body ---
    # This is the core content of the report
    if synthesis := report.get('synthesis'):
        if isinstance(synthesis, dict) and (content := synthesis.get('content')):
             markdown += f"""## Comprehensive Analysis
{content}
---
"""
        elif isinstance(synthesis, str):
             markdown += f"""## Comprehensive Analysis
{synthesis}
---
"""

    # --- Systematic Review Section ---
    if sr := report.get('systematic_review'):
        if sr.get('status') == 'completed' and sr.get('studies_included', 0) > 0:
            markdown += f"""## Systematic Review (PRISMA 2020)
**Status:** ✅ Completed
**Certainty of Evidence:** {sr.get('overall_certainty', 'Not Assessed')}

### PRISMA Flow
- **Records Identified from Databases:** {sr.get('records_identified', 'N/A')}
- **Studies Included in Synthesis:** {sr.get('studies_included', 'N/A')}

"""
            if detailed_results := sr.get('results'):
                markdown += f"### Key Findings\n{detailed_results}\n"
            markdown += "---\n"

    # --- Evidence Sources ---
    if evidence_sources := report.get('evidence_sources'):
        if isinstance(evidence_sources, list) and len(evidence_sources) > 0:
            markdown += "## Evidence Sources\n\n"
            markdown += "| # | Title | URL | Confidence |\n"
            markdown += "|---|---|---|---|\n"
            for i, source in enumerate(evidence_sources, 1):
                if isinstance(source, dict):
                    src_dict = source
                else:
                    # Pydantic model -> dict
                    src_dict = source.dict()
                title = src_dict.get('title', 'N/A').replace('|', '')
                url = src_dict.get('url', '#')
                conf = src_dict.get('confidence_score', src_dict.get('score', 0))
                confidence = f"{conf * 100:.0f}%" if isinstance(conf, (int, float)) else 'N/A'
                markdown += f"| {i} | {title} | {url} | {confidence} |\n"
            markdown += "\n---\n"

    # --- Personalized Medicine ---
    if pm := report.get('personalized_medicine'):
        if pm.get('status') != 'Not conducted' and (recommendations := pm.get('recommendations')):
            markdown += f"## 🧬 Personalized Medicine Recommendations\n"
            for i, rec in enumerate(recommendations, 1):
                markdown += f"""
### Recommendation {i}: {rec.get('intervention', 'N/A')}
- **Predicted Efficacy:** {rec.get('efficacy_prediction', 'N/A')}
- **Patient Subgroup:** {rec.get('patient_subgroup', 'General Population')}
- **Rationale:** {rec.get('rationale', 'N/A')}
"""
            markdown += "\n---\n"

    # --- Quality & Performance Metrics ---
    markdown += "## Report Metrics\n"
    
    if quality := report.get('quality_metrics'):
        markdown += f"- **Quality Score:** {quality.get('overall_score', 'N/A')}/100\n"

    if perf := report.get('performance_metrics'):
        def safe_format_time(value, default='N/A'):
            if isinstance(value, (int, float)):
                return f"{value:.2f}s"
            return str(default)
        
        total_time = safe_format_time(perf.get('total_generation_time'))
        markdown += f"- **Total Generation Time:** {total_time}\n"

    # --- Methodology ---
    markdown += """
---
## Methodology
This report was generated using the Med-STORM Revolutionary Evidence Synthesis Engine. The process combines multi-expert perspective generation (STORM) with a systematic, evidence-based medical analysis framework. Key data sources include PubMed/MEDLINE, Serper Web Search, and a local medical corpus.
"""

    return markdown


async def main():
    """Función principal para demostrar capacidades revolucionarias"""
    
    print("🚀 REVOLUTIONARY MED-STORM DEMONSTRATION")
    print("="*80)
    
    # --- Configuration ---
    # Simplified to use only OpenRouter as per the new architecture
    llm_provider = OpenRouterLLM(
        api_key=settings.OPENROUTER_API_KEY,
        model="google/gemini-2.5-flash"
    )

    connectors = {
        "pubmed": PubMedConnector(),
        "serper": SerperConnector(api_key=settings.SERPER_API_KEY),
        "local_corpus": LocalCorpusConnector(
            collection_name=getattr(settings, 'QDRANT_COLLECTION_NAME', 'med_storm_corpus'),
            qdrant_url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
        )
    }

    storm_engine = StormEnhancedMedicalEngine(
        llm_provider=llm_provider,
        connectors=connectors
    )

    # --- Topic and Patient Profile ---
    # topic = "Efficacy and safety of SGLT2 inhibitors in type 2 diabetes with established cardiovascular disease"
    topic = "Cardiovascular Risk Management in Diabetes"

    patient_profile = {
        "age": 65,
        "sex": "male",
        "comorbidities": ["hypertension", "dyslipidemia"],
        "lifestyle_factors": ["sedentary", "high-carbohydrate diet"]
    }

    print(f"🔬 Researching Topic: {topic}")
    print(f"👤 Patient Profile: {patient_profile}")
    print("="*80)

    try:
        # --- Run the Engine ---
        # Using the new, simplified `run` method
        final_report = await storm_engine.run(
            topic=topic,
            patient_profile=patient_profile,
            enable_systematic_review=True,
            enable_advanced_statistics=True,
            enable_personalized_medicine=True
        )

        # --- Save the Report ---
        if final_report:
            save_report_to_file(final_report, topic, "revolutionary")
            print("\n✅ Revolutionary report generated successfully.")
        else:
            print("\n❌ Report generation failed.")

    except Exception as e:
        logger.error(f"An error occurred during the revolutionary run: {e}", exc_info=True)
        print(f"\n❌ An error occurred: {e}")

    print("="*80)
    print("🏁 Demonstration complete.")


if __name__ == '__main__':
    # Ensure settings are loaded
    if not all([settings.OPENROUTER_API_KEY, settings.SERPER_API_KEY]):
        print("🔥 CRITICAL ERROR: API Keys are not configured.")
        print("Please check your .env file or environment variables for OPENROUTER_API_KEY and SERPER_API_KEY.")
        sys.exit(1)
        
    asyncio.run(main())