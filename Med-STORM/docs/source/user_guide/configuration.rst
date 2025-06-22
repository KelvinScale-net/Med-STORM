.. _configuration:

Configuration Guide
==================

This guide explains how to configure Med-STORM to suit your needs.

Configuration Methods
--------------------

1. **Environment Variables**

   Set these in your `.env` file or export them in your shell:

   .. code-block:: bash

      # Required API Keys
      DEEPSEEK_API_KEY=your_deepseek_api_key
      SERPER_API_KEY=your_serper_api_key
      
      # Cache Settings
      CACHE_ENABLED=true
      CACHE_DIR=./.llm_cache
      CACHE_TTL=86400  # seconds (1 day)
      
      # Logging
      LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
      LOG_FILE=med_storm.log
      
      # Performance
      MAX_CONCURRENT_REQUESTS=5
      REQUEST_TIMEOUT=30  # seconds

2. **Configuration File**

   Create a `config.yaml` in your working directory:

   .. code-block:: yaml

      # API Configuration
      api:
        deepseek:
          api_key: ${DEEPSEEK_API_KEY}
          base_url: https://api.deepseek.com/v1
          model: deepseek-chat
          temperature: 0.7
          max_tokens: 2000
        
        serper:
          api_key: ${SERPER_API_KEY}
          base_url: https://google.serper.dev
      
      # Search Settings
      search:
        default_max_results: 10
        min_confidence: 0.6
        max_retries: 3
        retry_delay: 2  # seconds
      
      # Evidence Processing
      evidence:
        default_evidence_types:
          - RCT
          - meta-analysis
          - systematic-review
        min_publication_year: 2015
        required_fields:
          - title
          - authors
          - publication_year
          - abstract
      
      # Report Generation
      report:
        default_format: html
        include_sources: true
        confidence_threshold: 0.6
        max_sources: 50

3. **Programmatic Configuration**

   .. code-block:: python

      from med_storm import MedSTORM
      from med_storm.config import settings
      
      # Update settings
      settings.update({
          "api.deepseek.temperature": 0.5,
          "search.default_max_results": 15,
          "evidence.min_publication_year": 2020
      })
      
      # Initialize with custom settings
      med_storm = MedSTORM(config={
          "api": {
              "deepseek": {
                  "model": "deepseek-chat-pro"
              }
          }
      })

Configuration Options
-------------------

API Configuration
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   
   * - Option
     - Type
     - Default
     - Description
   * - api.deepseek.api_key
     - String
     - ""
     - DeepSeek API key
   * - api.deepseek.base_url
     - String
     - "https://api.deepseek.com/v1"
     - Base URL for DeepSeek API
   * - api.deepseek.model
     - String
     - "deepseek-chat"
     - Default model to use
   * - api.deepseek.temperature
     - Float
     - 0.7
     - Sampling temperature (0-2)
   * - api.serper.api_key
     - String
     - ""
     - Serper API key
   * - api.serper.base_url
     - String
     - "https://google.serper.dev"
     - Base URL for Serper API

Search Settings
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   
   * - Option
     - Type
     - Default
     - Description
   * - search.default_max_results
     - Integer
     - 10
     - Default number of results to return
   * - search.min_confidence
     - Float
     - 0.6
     - Minimum confidence score (0-1)
   * - search.max_retries
     - Integer
     - 3
     - Maximum number of retry attempts
   * - search.retry_delay
     - Integer
     - 2
     - Delay between retries in seconds

Evidence Processing
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   
   * - Option
     - Type
     - Default
     - Description
   * - evidence.default_evidence_types
     - List[String]
     - ["RCT", "meta-analysis", "systematic-review"]
     - Default evidence types to include
   * - evidence.min_publication_year
     - Integer
     - 2015
     - Minimum publication year
   * - evidence.required_fields
     - List[String]
     - ["title", "authors", "publication_year", "abstract"]
     - Required fields for evidence sources

Report Generation
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   
   * - Option
     - Type
     - Default
     - Description
   * - report.default_format
     - String
     - "html"
     - Default report format (html, markdown, pdf)
   * - report.include_sources
     - Boolean
     - true
     - Whether to include sources in the report
   * - report.confidence_threshold
     - Float
     - 0.6
     - Minimum confidence score to include (0-1)
   * - report.max_sources
     - Integer
     - 50
     - Maximum number of sources to include

Caching
~~~~~~

.. list-table::
   :header-rows: 1
   
   * - Option
     - Type
     - Default
     - Description
   * - cache.enabled
     - Boolean
     - true
     - Whether to enable caching
   * - cache.dir
     - String
     - "./.llm_cache"
     - Cache directory
   * - cache.ttl
     - Integer
     - 86400
     - Cache time-to-live in seconds

Logging
~~~~~~~

.. list-table::
   :header-rows: 1
   
   * - Option
     - Type
     - Default
     - Description
   * - logging.level
     - String
     - "INFO"
     - Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   * - logging.file
     - String
     - ""
     - Log file path (empty for console only)
   * - logging.format
     - String
     - "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
     - Log message format

Performance Tuning
----------------

1. **Concurrent Requests**

   .. code-block:: yaml

      # config.yaml
      max_concurrent_requests: 10  # Adjust based on your API rate limits
      request_timeout: 30  # seconds

2. **Caching**

   Enable and configure caching to reduce API calls:

   .. code-block:: yaml

      # config.yaml
      cache:
        enabled: true
        dir: ~/.llm_cache/med-storm
        ttl: 86400  # 1 day in seconds

3. **Batch Processing**

   For processing multiple queries:

   .. code-block:: python

      from med_storm import MedSTORM
      import asyncio
      
      async def process_queries(queries):
          med_storm = MedSTORM()
          tasks = [med_storm.search(query) for query in queries]
          return await asyncio.gather(*tasks)
      
      queries = ["treatment for diabetes", "latest hypertension guidelines"]
      results = asyncio.run(process_queries(queries))

Environment-Specific Configuration
--------------------------------

1. **Development**

   Create a `config/development.yaml`:

   .. code-block:: yaml

      # Development-specific settings
      logging:
        level: DEBUG
        file: med_storm_dev.log
      
      api:
        deepseek:
          base_url: http://localhost:8000  # Mock server

2. **Production**

   Create a `config/production.yaml`:

   .. code-block:: yaml

      # Production settings
      logging:
        level: WARNING
        file: /var/log/med_storm/app.log
      
      cache:
        enabled: true
        dir: /var/cache/med-storm
      
      api:
        deepseek:
          base_url: https://api.deepseek.com/v1

Load the appropriate configuration based on the environment:

.. code-block:: python

   import os
   from med_storm import MedSTORM
   
   env = os.getenv("ENVIRONMENT", "development")
   config_file = f"config/{env}.yaml"
   
   med_storm = MedSTORM(config_file=config_file)

Verification
-----------

Verify your configuration:

.. code-block:: python

   from med_storm.config import settings
   
   # Print current configuration
   print("Current configuration:")
   print(settings)
   
   # Check if required settings are present
   try:
       settings.validate()
       print("Configuration is valid!")
   except ValueError as e:
       print(f"Configuration error: {e}")
