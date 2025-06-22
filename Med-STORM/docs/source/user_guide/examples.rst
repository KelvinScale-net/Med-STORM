.. _examples:

Examples
========

This page provides practical examples of how to use Med-STORM for various evidence synthesis tasks.

.. contents:: Table of Contents
   :depth: 3
   :local:
   :backlinks: top

Basic Usage
-----------

Searching for Evidence
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from med_storm import MedSTORM
   import asyncio
   
   async def search_evidence():
       # Initialize Med-STORM
       med_storm = MedSTORM()
       
       # Search for evidence
       results = await med_storm.search(
           "effectiveness of statins in primary prevention",
           max_results=5,
           min_confidence=0.7,
           evidence_types=["RCT", "meta-analysis"]
       )
       
       # Print results
       print(f"Found {len(results.sources)} sources")
       for source in results.sources:
           print(f"- {source.title} ({source.publication_year})")
           print(f"  Confidence: {source.confidence_score:.2f}")
   
   # Run the async function
   asyncio.run(search_evidence())

Generating a Report
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from med_storm import MedSTORM
   import asyncio
   
   async def generate_evidence_report():
       # Initialize Med-STORM
       med_storm = MedSTORM()
       
       # Search for evidence
       evidence = await med_storm.search(
           "latest treatments for type 2 diabetes",
           max_results=10
       )
       
       # Generate a report
       report = await med_storm.generate_report(
           evidence,
           format="html",
           include_sources=True
       )
       
       # Save the report
       report.save("diabetes_treatment_report.html")
       print("Report generated successfully!")
   
   # Run the async function
   asyncio.run(generate_evidence_report())

Advanced Usage
-------------

Custom Evidence Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from med_storm import MedSTORM
   from med_storm.core import EvidenceProcessor
   import asyncio
   
   async def custom_pipeline():
       # Initialize Med-STORM
       med_storm = MedSTORM()
       
       # Search for evidence
       evidence = await med_storm.search(
           "efficacy of immunotherapy in lung cancer",
           max_results=15
       )
       
       # Create a custom processing pipeline
       processor = EvidenceProcessor()
       
       # Define processing steps
       processed = await processor.process(
           evidence,
           steps=[
               "validate_sources",
               "extract_key_findings",
               "assess_quality",
               "resolve_contradictions"
           ]
       )
       
       # Generate a report from processed evidence
       report = await med_storm.generate_report(
           processed,
           format="markdown",
           confidence_threshold=0.65
       )
       
       # Save the report
       report.save("lung_cancer_immunotherapy_report.md")
   
   # Run the async function
   asyncio.run(custom_pipeline())

Batch Processing Multiple Queries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from med_storm import MedSTORM
   import asyncio
   
   async def process_queries():
       # List of queries to process
       queries = [
           "treatment for hypertension in elderly",
           "latest guidelines for asthma management",
           "effectiveness of cognitive behavioral therapy for depression"
       ]
       
       # Initialize Med-STORM
       med_storm = MedSTORM()
       
       # Process each query
       for query in queries:
           print(f"Processing query: {query}")
           
           # Search for evidence
           evidence = await med_storm.search(
               query,
               max_results=5,
               min_confidence=0.6
           )
           
           # Generate a report
           report = await med_storm.generate_report(
               evidence,
               format="html"
           )
           
           # Save the report
           filename = f"report_{query[:30].replace(' ', '_').lower()}.html"
           report.save(filename)
           print(f"Saved report to {filename}")
   
   # Run the async function
   asyncio.run(process_queries())

Real-world Examples
------------------

Systematic Literature Review
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how to use Med-STORM to assist with a systematic literature review:

1. **Define Research Question**:
   - Use PICO framework (Population, Intervention, Comparison, Outcome)
   - Example: "In adults with type 2 diabetes (P), does metformin (I) compared to sulfonylureas (C) reduce cardiovascular events (O)?"

2. **Search Strategy**:
   - Use specific search terms and Boolean operators
   - Filter by study type, publication date, and language

3. **Evidence Collection**:
   - Search multiple databases (PubMed, ClinicalTrials.gov)
   - Export results for screening

4. **Screening and Data Extraction**:
   - Use Med-STORM to screen titles/abstracts
   - Extract key data from full-text articles

5. **Synthesis and Reporting**:
   - Generate evidence tables
   - Create PRISMA flow diagram
   - Write systematic review sections

Clinical Decision Support
~~~~~~~~~~~~~~~~~~~~~~~

Med-STORM can be used to create a clinical decision support system:

1. **Integrate with EHR**:
   - Connect to electronic health records
   - Extract patient-specific information

2. **Generate Patient-specific Evidence**:
   - Formulate queries based on patient data
   - Retrieve relevant evidence
   - Filter by patient characteristics

3. **Present Recommendations**:
   - Generate structured reports
   - Include confidence levels
   - Link to source documents

4. **Continuous Updates**:
   - Set up alerts for new evidence
   - Update recommendations as needed

Troubleshooting
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **No Results Found**
   - Check your internet connection
   - Verify API keys are set correctly
   - Try a broader search query
   - Check if the source is available and responding

2. **API Rate Limits**
   - Reduce the number of concurrent requests
   - Implement retry logic with exponential backoff
   - Consider using a paid plan with higher limits

3. **Memory Issues**
   - Process evidence in smaller batches
   - Close unused connections and files
   - Use streaming for large datasets

4. **Report Generation Failures**
   - Check if all required fields are present
   - Verify write permissions for the output directory
   - Try a different output format

Getting Help
-----------

If you encounter issues not covered here:

1. Check the :ref:`faq` section
2. Review the :ref:`api_reference`
3. Open an issue on our `GitHub repository <https://github.com/your-username/med-storm/issues>`_
