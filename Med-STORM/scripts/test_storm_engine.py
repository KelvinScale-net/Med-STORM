import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Adjust path to import from the project's source
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from med_storm.core.engine import StormEngine
from med_storm.llm.openai import OpenAILLM
from med_storm.llm.cache import CachingLLM
from med_storm.connectors.pubmed import PubMedConnector
from med_storm.connectors.serper import SerperConnector
from med_storm.core.query_optimizer import QueryOptimizer
from med_storm.synthesis.engine import SynthesisEngine
from med_storm.synthesis.report_generator import ReportGenerator
from med_storm.synthesis.executive_summary_generator import ExecutiveSummaryGenerator
from med_storm.synthesis.bibliography_generator import BibliographyGenerator
from med_storm.config import settings

async def main():
    """Main function to test the full Med-STORM report generation pipeline."""
    print("Testing Med-STORM Engine: Full Report Generation...")

    # -- DEBUG: Print environment information --
    print("\n[DEBUG] Environment Variables:")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Environment file path: {os.path.abspath('.env')}")
    print(f"Environment file exists: {os.path.exists('.env')}")
    
    # Print all environment variables that start with DEEPSEEK_, SERPER_, or PUBMED_
    print("\n[DEBUG] Environment Variables:")
    for key, value in os.environ.items():
        if any(prefix in key.upper() for prefix in ['DEEPSEEK', 'SERPER', 'PUBMED']):
            print(f"{key} = {'*' * 8 if 'KEY' in key or 'API' in key else value}")
    print(f"DEEPSEEK_API_KEY loaded: {bool(settings.deepseek_api_key)}")
    print(f"SERPER_API_KEY loaded: {bool(settings.serper_api_key)}")
    print(f"PUBMED_API_KEY loaded: {bool(settings.pubmed_api_key)}")
    # -- END DEBUG --
    
    if not settings.deepseek_api_key or not settings.serper_api_key:
        print("\nERROR: DEEPSEEK_API_KEY or SERPER_API_KEY is not set in the .env file.")
        print("Please make sure the .env file exists in the project root and contains valid API keys.")
        return

    try:
        # 1. Setup the components
        # Use OpenAILLM with a caching layer to optimize costs
        openai_llm = OpenAILLM(api_key=settings.openai_api_key)
        llm = CachingLLM(llm_provider=openai_llm)
        optimizer = QueryOptimizer(llm_provider=llm)
        synthesizer = SynthesisEngine(llm_provider=llm)
        report_generator = ReportGenerator(llm_provider=llm)
        summary_generator = ExecutiveSummaryGenerator(llm_provider=llm)
        bib_generator = BibliographyGenerator()
        
        # 2. Define knowledge sources
        pubmed_kc = PubMedConnector()
        web_kc = SerperConnector(trusted_domains=[
            "who.int", "cdc.gov", "nih.gov", "mayoclinic.org", "nice.org.uk"
        ])
        
        # 3. Instantiate the full engine
        engine = StormEngine(
            llm_provider=llm, 
            knowledge_connectors=[pubmed_kc, web_kc],
            query_optimizer=optimizer,
            synthesis_engine=synthesizer,
            report_generator=report_generator,
            executive_summary_generator=summary_generator,
            bibliography_generator=bib_generator
        )

        # 4. Run the full STORM process
        topic = "Management of Type 2 Diabetes in Adults"
        print(f"\nGenerating report for: {topic}")
        full_report = await engine.run_full_storm(topic)

        # 5. Display the final, structured report
        print("\n" + "="*80)
        print(" " * 25 + "COMPLETED CLINICAL REVIEW")
        print("="*80)

        print(f"\nTOPIC: {full_report['topic']}\n")

        print("\n" + "-"*80)
        print(" " * 30 + "EXECUTIVE SUMMARY")
        print("-" * 80)
        print(full_report['executive_summary'])

        for sub_topic, chapter_content in full_report['chapters'].items():
            print("\n" + "-"*80)
            print(f" " * 20 + f"CHAPTER: {sub_topic}")
            print("-" * 80)
            print(chapter_content)

        print("\n" + "-"*80)
        print(" " * 32 + "BIBLIOGRAPHY")
        print("-" * 80)
        print(full_report['bibliography'])

        # 6. Save the report to a file
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"{topic.replace(' ', '_').lower()}.md"
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Clinical Review: {full_report['topic']}\n\n")
            f.write("## Executive Summary\n\n")
            f.write(full_report['executive_summary'] + "\n\n")
            
            # Write each chapter
            for sub_topic, chapter_content in full_report['chapters'].items():
                f.write(f"## {sub_topic}\n\n")
                f.write(chapter_content + "\n\n")
                
            f.write("## Bibliography\n\n")
            f.write(full_report['bibliography'] + "\n")

        print("\n" + "="*80)
        print(f" " * 15 + f"Med-STORM Full Report Generation Successful!")
        print(f" " * 18 + f"Report saved to: {file_path}")
        print("="*80)

    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")

if __name__ == "__main__":
    asyncio.run(main())
