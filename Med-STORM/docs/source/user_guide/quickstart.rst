.. _quickstart:

Quickstart Guide
===============

This guide will help you get started with Med-STORM for evidence-based medical knowledge synthesis.

Basic Usage
-----------

1. **Import and Initialize**

   .. code-block:: python

      from med_storm import MedSTORM
      import asyncio

      async def main():
          # Initialize with default settings
          med_storm = MedSTORM()
          
          # Search for evidence
          query = "What are the latest treatments for type 2 diabetes?"
          evidence = await med_storm.search(query, max_results=5)
          
          # Generate a report
          report = await med_storm.generate_report(evidence)
          print(report.summary)
          
          # Save the full report
          report.save("diabetes_treatment_report.html")

      # Run the async function
      asyncio.run(main())


2. **Search for Evidence**

   .. code-block:: python

      # Search with specific parameters
      evidence = await med_storm.search(
          "efficacy of statins in primary prevention",
          max_results=10,
          min_confidence=0.7,
          publication_years=(2015, 2025),
          evidence_types=["RCT", "meta-analysis"]
      )

3. **Analyze Evidence**

   .. code-block:: python

      # Get a summary of the evidence
      summary = await med_storm.analyze_evidence(evidence)
      print(f"Found {len(summary.studies)} studies")
      print(f"Publication years: {summary.years}")
      print(f"Confidence scores: {summary.confidence_scores}")

4. **Generate Reports**

   .. code-block:: python

      # Generate different report formats
      report = await med_storm.generate_report(
          evidence,
          format="html",  # or "markdown", "pdf"
          include_sources=True,
          confidence_threshold=0.6
      )
      
      # Save to a file
      report.save("report.html")
      
      # Or get the content directly
      print(report.content)

Advanced Usage
-------------

1. **Custom Search**

   .. code-block:: python

      from med_storm.connectors import PubMedConnector, ClinicalTrialsConnector
      
      # Create custom connector instances
      pubmed = PubMedConnector()
      clinical_trials = ClinicalTrialsConnector()
      
      # Search with specific connectors
      pubmed_results = await pubmed.search("covid-19 treatment", max_results=5)
      trial_results = await clinical_trials.search("covid-19 vaccine", status="recruiting")

2. **Evidence Processing Pipeline**

   .. code-block:: python

      from med_storm.core import EvidenceProcessor
      
      processor = EvidenceProcessor()
      
      # Process evidence with custom pipeline
      processed = await processor.process(
          evidence,
          steps=[
              "validate_sources",
              "extract_key_findings",
              "assess_quality",
              "resolve_contradictions"
          ]
      )

3. **Using Custom Models**

   .. code-block:: python

      from med_storm.models import EvidenceCorpus, EvidenceSource
      
      # Create custom evidence
      source = EvidenceSource(
          title="Custom Study on Treatment X",
          authors=["Researcher A", "Researcher B"],
          publication_year=2024,
          journal="Journal of Medical Research",
          url="https://example.com/study",
          confidence=0.85
      )
      
      corpus = EvidenceCorpus(query="custom query", sources=[source])

Command Line Interface
---------------------

Med-STORM also provides a command-line interface for common tasks:

.. code-block:: bash

   # Search for evidence
   med-storm search "treatment for hypertension" --max-results 5 --output results.json
   
   # Generate a report
   med-storm report results.json --format html --output report.html
   
   # View available commands and options
   med-storm --help

Next Steps
----------

- :ref:`configuration` - Configure advanced settings
- :ref:`examples` - More example use cases
- :ref:`api_reference` - Detailed API documentation
