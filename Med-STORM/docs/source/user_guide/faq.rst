.. _faq:

Frequently Asked Questions (FAQ)
===============================

This page answers common questions about using Med-STORM.

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: top

General
-------

What is Med-STORM?
~~~~~~~~~~~~~~~~~~
Med-STORM is an open-source tool for evidence-based medical knowledge synthesis. It helps researchers and healthcare professionals find, analyze, and synthesize medical evidence from various sources.

What does STORM stand for?
~~~~~~~~~~~~~~~~~~~~~~~~~
STORM stands for "Systematic Tool for Organized Research in Medicine."

Is Med-STORM free to use?
~~~~~~~~~~~~~~~~~~~~~~~~
Yes, Med-STORM is open-source and free to use under the MIT License. However, some features may require API keys from third-party services that might have their own pricing.

Installation
-----------

What are the system requirements?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Python 3.9 or higher
- 4GB RAM (8GB recommended)
- Internet connection for API access
- Approximately 100MB disk space

How do I install Med-STORM?
~~~~~~~~~~~~~~~~~~~~~~~~~~
See the :ref:`installation` guide for detailed instructions.

I'm getting dependency conflicts. What should I do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Try creating a fresh virtual environment and installing Med-STORM there:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install --upgrade pip setuptools wheel
   pip install med-storm

Usage
-----

How do I get started with Med-STORM?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Check out our :ref:`quickstart` guide for a quick introduction.

What search engines/databases does Med-STORM support?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Med-STORM supports multiple sources organized by evidence tiers:

1. **Tier 1 (Maximum Confidence)**:
   - PubMed/MEDLINE
   - ClinicalTrials.gov

2. **Tier 2 (High Confidence)**:
   - Google Scholar (via Serper API)
   - Medical guidelines (WHO, CDC, etc.)

3. **Tier 3 (Supplementary)**:
   - Conference proceedings
   - Preprints
   - Expert opinions

How do I interpret the confidence scores?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Confidence scores range from 0 to 1, where:

- 0.8-1.0: High confidence (Tier 1 sources, well-designed studies)
- 0.6-0.79: Medium confidence (Tier 2 sources, some limitations)
- 0.4-0.59: Low confidence (Tier 3 sources, significant limitations)
- <0.4: Very low confidence (use with caution)

Can I customize the evidence synthesis process?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Yes, you can customize:
- Search parameters (date ranges, study types, etc.)
- Evidence processing pipeline
- Report format and content
- Confidence thresholds

Troubleshooting
--------------

I'm getting API key errors. What should I do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Make sure you've set the required API keys in your `.env` file
2. Verify the keys are correct and have the necessary permissions
3. Check if you've exceeded any rate limits
4. Ensure your subscription is active if using paid APIs

Why am I not getting any results?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Check your internet connection
2. Verify your search query is well-formed
3. Try a broader search
4. Check if the source is available and responding
5. Look for error messages in the console output

How can I improve search results?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Use specific, well-defined search terms
2. Include synonyms and alternative phrasings
3. Use Boolean operators (AND, OR, NOT)
4. Filter by publication date, study type, etc.
5. Adjust confidence thresholds

Advanced
-------

Can I add custom data sources?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Yes, you can create custom connectors by subclassing `BaseConnector`. See the :ref:`developer_guide` for more information.

How does Med-STORM handle conflicting evidence?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Med-STORM uses several strategies:
1. Source hierarchy (Tier 1 > Tier 2 > Tier 3)
2. Study design and quality assessment
3. Sample size and statistical power
4. Publication date (newer evidence preferred)
5. Consensus among multiple sources

Can I use Med-STORM for non-medical research?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
While designed for medical research, Med-STORM can be adapted for other evidence synthesis tasks. However, you may need to customize the evidence assessment criteria and sources.

Performance & Scaling
-------------------

How can I improve performance for large searches?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Increase the number of concurrent requests (if your API limits allow)
2. Use caching to avoid redundant API calls
3. Process evidence in smaller batches
4. Use more specific search queries to reduce result sets

Is there a way to run Med-STORM in parallel?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Yes, you can use Python's `asyncio` to run multiple searches or analyses in parallel. See the :ref:`examples` section for sample code.

How much data can Med-STORM handle?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Med-STORM is designed to handle thousands of documents, but performance depends on:
- Available system resources
- API rate limits
- Complexity of evidence processing
- Output format and detail level

Getting Help
-----------

Where can I report bugs or request features?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Please open an issue on our `GitHub repository <https://github.com/your-username/med-storm/issues>`_.

Is there a community forum or chat?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Not currently, but you can start discussions in the GitHub Discussions section of our repository.

Can I contribute to Med-STORM?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Yes! We welcome contributions. See our :ref:`contributing` guide for more information.

Legal & Compliance
-----------------

Is Med-STORM HIPAA compliant?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Med-STORM itself doesn't store or process protected health information (PHI). However, if you integrate it with systems containing PHI, you'll need to ensure proper safeguards are in place.

What are the data privacy implications?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Med-STORM doesn't store your search queries or results by default
- API providers may log requests according to their privacy policies
- For sensitive queries, consider using local models or self-hosted solutions

Can I use Med-STORM for commercial purposes?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Yes, Med-STORM is licensed under the MIT License, which allows for commercial use. However, some third-party APIs may have their own usage restrictions.
