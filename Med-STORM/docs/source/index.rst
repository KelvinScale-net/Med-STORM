.. Med-STORM documentation master file
   ==================================

   Welcome to Med-STORM's documentation!

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: container pb-3

   .. grid-item::
      :class: sd-fs-5 text-center

      .. link-button:: user_guide/installation
         :type: ref
         :text: Get Started
         :classes: btn-outline-primary btn-block stretched-link

      .. rst-class:: mt-2

      New to Med-STORM? Start here to install and configure the tool.

   .. grid-item::
      :class: sd-fs-5 text-center

      .. link-button:: user_guide/quickstart
         :type: ref
         :text: Quickstart
         :classes: btn-outline-primary btn-block stretched-link

      .. rst-class:: mt-2

      Learn the basics with our quickstart guide.

   .. grid-item::
      :class: sd-fs-5 text-center

      .. link-button:: api_reference/index
         :type: ref
         :text: API Reference
         :classes: btn-outline-primary btn-block stretched-link

      .. rst-class:: mt-2

      Detailed API documentation for developers.

..
   Include the README content
.. include:: ../../README.md
   :parser: myst_parser.sphinx_
   :start-after: <!-- start-doc-overview -->
   :end-before: <!-- end-doc-overview -->


.. toctree::
   :maxdepth: 2
   :caption: "User Guide"
   :hidden:
   :glob:

   
   user_guide/*
   :exclude:
      user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: "Developer Guide"
   :hidden:
   :glob:
   
   developer_guide/*
   developer_roadmap
   contributing
   code_of_conduct

.. toctree::
   :maxdepth: 2
   :caption: "API Reference"
   :hidden:
   :glob:
   
   api_reference/*
   :exclude:
      api_reference/index

.. toctree::
   :maxdepth: 2
   :caption: "Additional Resources"
   :hidden:
   
   acknowledgments
   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

..
   Add custom CSS and JavaScript
   =============================
   
   .. raw:: html
   
      <link rel="stylesheet" href="_static/custom.css" type="text/css" />
      <script type="text/javascript" src="_static/custom.js"></script>
