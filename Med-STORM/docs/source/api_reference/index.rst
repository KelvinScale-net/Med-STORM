.. _api_reference:

API Reference
============

This section contains the complete API reference for Med-STORM, organized by module.

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   connectors
   core
   models
   utils

Base Classes
------------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:
   
   med_storm.connectors.base
   med_storm.models.base
   med_storm.core.base

Connectors
----------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   
   med_storm.connectors.pubmed
   med_storm.connectors.clinicaltrials
   med_storm.connectors.web_search
   med_storm.connectors.guidelines
   med_storm.connectors.secondary

Core Modules
------------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   
   med_storm.core.processor
   med_storm.core.synthesis
   med_storm.core.query
   med_storm.core.report

Data Models
-----------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   
   med_storm.models.evidence
   med_storm.models.sources
   med_storm.models.reports

Utilities
---------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   
   med_storm.utils.text
   med_storm.utils.cache
   med_storm.utils.config
   med_storm.utils.logging

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
