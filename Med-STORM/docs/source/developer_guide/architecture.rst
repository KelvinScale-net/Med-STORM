.. _architecture:

System Architecture
===================

This document outlines the high-level architecture of Med-STORM, focusing on the evidence-based approach and technical decisions made during development.

.. contents:: Table of Contents
   :depth: 3
   :local:
   :backlinks: top

Overview
--------

Med-STORM is designed as a modular, extensible system for evidence-based medical knowledge synthesis. The architecture follows a pipeline pattern where data flows through several stages of processing, each responsible for a specific aspect of the evidence synthesis process.

.. mermaid::
   :align: center
   :caption: High-Level System Architecture

   graph TD
       A[User Query] --> B[Query Understanding]
       B --> C[Evidence Collection]
       C --> D[Evidence Processing]
       D --> E[Evidence Synthesis]
       E --> F[Report Generation]
       
       subgraph Evidence Collection
           C1[PubMed API]
           C2[ClinicalTrials.gov]
           C3[Google Scholar]
           C4[Medical Guidelines]
       end
       
       subgraph Evidence Processing
           D1[Source Validation]
           D2[Evidence Grading]
           D3[Contradiction Detection]
       end

Evidence Collection Layer
------------------------

The evidence collection layer is responsible for gathering information from various sources with different levels of confidence.

Trust Tiers
~~~~~~~~~~~

1. **Tier 1 - Maximum Confidence**
   - PubMed/MEDLINE
   - ClinicalTrials.gov
   - Characteristics:
     - Peer-reviewed
     - Clinical trials and meta-analyses
     - Directly from authoritative sources

2. **Tier 2 - High Confidence**
   - Google Scholar (peer-reviewed results)
   - Medical organization guidelines (WHO, CDC, etc.)
   - Medical textbooks
   - Characteristics:
     - Peer-reviewed or vetted by experts
     - From reputable organizations
     - May include some pre-prints

3. **Tier 3 - Variable Confidence**
   - High-quality secondary sources
   - Conference proceedings
   - Characteristics:
     - Clearly labeled confidence level
     - Used for supplementary information
     - Requires verification

API Integrations
~~~~~~~~~~~~~~~

- **PubMed E-utilities**: Primary source for peer-reviewed medical literature
- **Serper API**: Web search with medical domain focus
- **DeepSeek API**: LLM for content analysis and synthesis
- **ClinicalTrials.gov API**: For ongoing and completed clinical trials

Evidence Processing Pipeline
---------------------------

The evidence processing pipeline is responsible for evaluating and grading the collected evidence.

1. **Source Validation**
   - Verify source authenticity
   - Check for retractions
   - Validate publication metadata

2. **Evidence Grading**
   - Apply GRADE methodology
   - Consider study design and limitations
   - Evaluate risk of bias

3. **Contradiction Detection**
   - Identify conflicting evidence
   - Resolve contradictions using predefined rules
   - Maintain transparency about conflicts

Technical Stack
--------------

- **Core Language**: Python 3.9+
- **Asynchronous Framework**: asyncio
- **LLM Integration**: DeepSeek API with OpenAI compatibility layer
- **Search**: Serper API, PubMed E-utilities
- **Data Processing**: Pandas, NumPy
- **Web Framework**: FastAPI (for future web interface)
- **Testing**: pytest, pytest-cov
- **Documentation**: Sphinx, ReadTheDocs

Data Models
-----------

The system uses several core data models to represent evidence and its metadata:

.. mermaid::
   :align: center
   :caption: Core Data Models

   classDiagram
       class EvidenceSource {
           +str source_id
           +str title
           +str authors
           +int publication_year
           +str journal
           +str url
           +SourceTier tier
           +float confidence_score
           +list[EvidenceType] evidence_types
           +dict metadata
       }
       
       class EvidenceCorpus {
           +str query
           +list[EvidenceSource] sources
           +datetime retrieval_date
           +add_source()
           +filter_by_tier()
           +to_dataframe()
       }
       
       class EvidenceReport {
           +str query
           +str summary
           +list[KeyFinding] key_findings
           +list[Recommendation] recommendations
           +EvidenceCorpus sources
           +generate_markdown()
           +generate_pdf()
       }

Security and Privacy
-------------------

- All API keys are stored in environment variables
- No patient data is stored or processed
- All data is processed locally when possible
- HTTPS is enforced for all API communications

Performance Considerations
-------------------------

- Asynchronous I/O for API calls
- Caching of frequent queries
- Rate limiting and retry logic
- Batch processing for large datasets

Future Extensions
----------------

- Support for additional evidence sources
- Integration with electronic health records (EHR)
- Automated literature review generation
- Clinical decision support system integration

For more detailed technical specifications, see the :ref:`API Reference <api_reference>`.
