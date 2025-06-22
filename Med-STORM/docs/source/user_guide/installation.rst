.. _installation:

Installation Guide
==================

This guide will help you install Med-STORM and its dependencies.

Prerequisites
------------

- Python 3.9 or higher
- pip (Python package manager)
- Git (for development installations)

Installation Methods
-------------------

1. **Install from PyPI (Recommended for Users)**

   .. code-block:: bash

      pip install med-storm

2. **Install from Source (For Developers)**

   .. code-block:: bash

      # Clone the repository
      git clone https://github.com/your-username/med-storm.git
      cd med-storm
      
      # Create and activate a virtual environment
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      
      # Install in development mode with all dependencies
      pip install -e .[dev]

3. **Using Docker (Alternative)**

   .. code-block:: bash

      docker pull yourusername/med-storm:latest
      docker run -it --rm yourusername/med-storm

Configuration
------------

1. **Environment Variables**

   Create a `.env` file in your project root with the following variables:

   .. code-block:: bash

      # Required API Keys
      DEEPSEEK_API_KEY=your_deepseek_api_key
      SERPER_API_KEY=your_serper_api_key
      
      # Optional Configuration
      CACHE_DIR=./.llm_cache
      LOG_LEVEL=INFO

2. **Verification**

   Verify your installation by running:

   .. code-block:: bash

      python -c "import med_storm; print('Med-STORM version:', med_storm.__version__)"

Troubleshooting
--------------

1. **Dependency Conflicts**

   If you encounter dependency conflicts, try:

   .. code-block:: bash

      pip install --upgrade pip setuptools wheel
      pip install -r requirements.txt

2. **API Key Issues**

   Ensure your API keys are correctly set in the `.env` file and have the necessary permissions.

3. **Memory Issues**

   For large evidence sets, you might need to increase Python's memory limit or use a machine with more RAM.

Next Steps
----------

- :ref:`quickstart` - Get started with basic usage
- :ref:`configuration` - Advanced configuration options
- :ref:`examples` - Example use cases and code snippets
