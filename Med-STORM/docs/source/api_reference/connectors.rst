.. _api_connectors:

Connectors
==========

This module contains the connector classes for various evidence sources used by Med-STORM, organized by evidence confidence tiers.

.. note::
   All connectors implement the base :class:`~med_storm.connectors.base.BaseConnector` interface,
   ensuring consistent behavior across different evidence sources.

Base Classes
------------

.. automodule:: med_storm.connectors.base
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Tier 1: Maximum Confidence Sources
--------------------------------

PubMed Connector
~~~~~~~~~~~~~~~~

.. automodule:: med_storm.connectors.pubmed
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Example Usage

   .. code-block:: python

      from med_storm.connectors import PubMedConnector
      import asyncio

      async def search_pubmed():
          pubmed = PubMedConnector()
          results = await pubmed.search(
              "covid-19 treatment",
              max_results=5,
              sort="relevance",
              publication_date="5 years"
          )
          return results

      # Run the async function
      results = asyncio.run(search_pubmed())

ClinicalTrials.gov Connector
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: med_storm.connectors.clinicaltrials
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Example Usage

   .. code-block:: python

      from med_storm.connectors import ClinicalTrialsConnector
      import asyncio

      async def search_clinical_trials():
          ct = ClinicalTrialsConnector()
          results = await ct.search(
              "covid-19 vaccine",
              status="recruiting",
              study_type="interventional"
          )
          return results

      # Run the async function
      results = asyncio.run(search_clinical_trials())

Tier 2: High Confidence Sources
-----------------------------

Web Search Connector
~~~~~~~~~~~~~~~~~~~

.. automodule:: med_storm.connectors.web_search
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Example Usage

   .. code-block:: python

      from med_storm.connectors import WebSearchConnector
      import asyncio

      async def search_web():
          web = WebSearchConnector()
          results = await web.search(
              "latest WHO guidelines on diabetes",
              site="who.int",
              max_results=3
          )
          return results

      # Run the async function
      results = asyncio.run(search_web())

Guideline Connector
~~~~~~~~~~~~~~~~~~

.. automodule:: med_storm.connectors.guidelines
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Example Usage

   .. code-block:: python

      from med_storm.connectors import GuidelineConnector
      import asyncio

      async def get_guidelines():
          guidelines = GuidelineConnector()
          results = await guidelines.search(
              "hypertension management",
              organization="who,esha,acc"
          )
          return results

      # Run the async function
      results = asyncio.run(get_guidelines())

Tier 3: Supplementary Sources
---------------------------

Secondary Source Connector
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: med_storm.connectors.secondary
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Example Usage

   .. code-block:: python

      from med_storm.connectors import SecondarySourceConnector
      import asyncio

      async def search_secondary():
          secondary = SecondarySourceConnector()
          results = await secondary.search(
              "emerging treatments for alzheimer's",
              source_types=["conference", "preprint", "expert_opinion"]
          )
          return results

      # Run the async function
      results = asyncio.run(search_secondary())

Creating Custom Connectors
-------------------------

To create a custom connector, inherit from :class:`~med_storm.connectors.base.BaseConnector`
and implement the required methods:

.. code-block:: python

   from typing import List, Optional, Dict, Any
   from med_storm.connectors.base import BaseConnector, EvidenceCorpus
   from med_storm.models.evidence import EvidenceSource
   import aiohttp

   class MyCustomConnector(BaseConnector):
       """Custom connector for a specific evidence source."""
       
       def __init__(self, api_key: Optional[str] = None):
           super().__init__(source_name="my_custom_source")
           self.api_key = api_key or os.getenv("MY_CUSTOM_API_KEY")
           self.base_url = "https://api.custom-source.com/v1"
       
       async def search(
           self,
           query: str,
           max_results: int = 10,
           **kwargs
       ) -> EvidenceCorpus:
           """Search the custom source for evidence."""
           # Implementation here
           pass
       
       async def _fetch_evidence(
           self,
           evidence_id: str
       ) -> Optional[EvidenceSource]:
           """Fetch detailed evidence by ID."""
           # Implementation here
           pass

For more details, see the :ref:`developer_guide`.
